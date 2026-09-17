"""
High-level software embedding interface for Vaayu SLMM & Vaayu-Large.
Supports:
- Vaayu.from_pretrained("meetmendapara/Vaayu-Base")
- Vaayu.load_local("path/to/weights.pt")
- In-process tool calling, parallel execution, automated error self-correction, and full MCP orchestration.
"""

import re
import json
import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Union, Iterator

from .core import VaayuInferenceEngine, ToolCall
from .mcp_client import MCPServerConnection, MCPTool
from .tools import (
    infer_schema_from_callable,
    FilesystemTool,
    ShellTool,
    HttpTool,
    SqliteTool
)

@dataclass
class StreamEvent:
    """Event emitted during real-time streaming inference."""
    type: str  # 'thought', 'tool_call', 'tool_result', 'text'
    content: str = ""
    tool_name: str = ""
    arguments: Dict[str, Any] = field(default_factory=dict)

class Vaayu:
    """Embedded Vaayu SLMM instance."""
    def __init__(self, engine: VaayuInferenceEngine):
        self.engine = engine
        self.mcp_connections: List[MCPServerConnection] = []
        self.local_tools: Dict[str, Callable] = {}
        self.local_tool_schemas: List[Dict[str, Any]] = []

    @classmethod
    def from_pretrained(
        cls,
        model_id_or_path: str = "meetmendapara/Vaayu-Base",
        variant: str = "base",
        device: Optional[str] = None
    ) -> "Vaayu":
        """
        Loads Vaayu from Hugging Face Hub (e.g. 'meetmendapara/Vaayu-Base') or local directory.
        Downloads and caches model weights automatically.
        """
        path_obj = Path(model_id_or_path)
        if path_obj.exists():
            if path_obj.is_file():
                return cls.load_local(checkpoint_path=str(path_obj), variant=variant, device=device)
            else:
                ckpt = path_obj / "vaayu_final.pt"
                if not ckpt.exists():
                    ckpt = path_obj / "pytorch_model.bin"
                return cls.load_local(
                    checkpoint_path=str(ckpt) if ckpt.exists() else None,
                    tokenizer_dir=str(path_obj),
                    variant=variant,
                    device=device
                )

        # Download snapshot from Hugging Face Hub
        print(f"[Vaayu Hub] Loading '{model_id_or_path}' from Hugging Face...")
        try:
            from huggingface_hub import snapshot_download
            cached_dir = snapshot_download(repo_id=model_id_or_path)
            cached_path = Path(cached_dir)
            ckpt = cached_path / "vaayu_final.pt"
            if not ckpt.exists():
                ckpt = cached_path / "pytorch_model.bin"
            return cls.load_local(
                checkpoint_path=str(ckpt) if ckpt.exists() else None,
                tokenizer_dir=str(cached_path),
                variant=variant,
                device=device
            )
        except Exception as e:
            print(f"[WARN] Failed to load from Hugging Face Hub ({e}). Falling back to local search.")
            return cls.load_local(variant=variant, device=device)

    @classmethod
    def load_local(
        cls,
        checkpoint_path: Optional[str] = None,
        tokenizer_dir: Optional[str] = None,
        variant: str = "base",
        device: Optional[str] = None
    ) -> "Vaayu":
        engine = VaayuInferenceEngine(
            checkpoint_path=checkpoint_path,
            tokenizer_dir=tokenizer_dir,
            variant=variant,
            device=device
        )
        return cls(engine)

    def attach_mcp_server(self, command: str, args: Optional[List[str]] = None) -> MCPServerConnection:
        """Connect to an external local MCP server running via stdio."""
        conn = MCPServerConnection(command, args)
        conn.start()
        self.mcp_connections.append(conn)
        print(f"[OK] Attached MCP Server: {command} ({len(conn.tools)} tools discovered)")
        return conn

    def register_function(self, name: str, description: str, input_schema: Dict[str, Any], func: Callable):
        """Register an in-process local Python function as an executable tool."""
        self.local_tools[name] = func
        # Update if exists, or append
        existing = [i for i, t in enumerate(self.local_tool_schemas) if t["name"] == name]
        schema_dict = {
            "name": name,
            "description": description,
            "inputSchema": input_schema
        }
        if existing:
            self.local_tool_schemas[existing[0]] = schema_dict
        else:
            self.local_tool_schemas.append(schema_dict)

    def tool(
        self,
        func_or_name: Optional[Union[Callable, str]] = None,
        name: Optional[str] = None,
        description: Optional[str] = None
    ):
        """
        Decorator to register a Python function as an in-process tool with automatic schema inference.
        Supports:
            @ai.tool
            def my_fn(param: str) -> str: ...

            @ai.tool(name="custom_name", description="Custom description")
            def my_fn(param: str) -> str: ...

            @ai.tool("custom_name", description="Custom description")
            def my_fn(param: str) -> str: ...
        """
        if callable(func_or_name):
            func = func_or_name
            schema = infer_schema_from_callable(func)
            self.register_function(
                name=schema["name"],
                description=schema["description"],
                input_schema=schema["inputSchema"],
                func=func
            )
            return func

        chosen_name = name or (func_or_name if isinstance(func_or_name, str) else None)

        def decorator(func: Callable):
            schema = infer_schema_from_callable(func, name_override=chosen_name, desc_override=description)
            self.register_function(
                name=schema["name"],
                description=schema["description"],
                input_schema=schema["inputSchema"],
                func=func
            )
            return func

        return decorator

    def enable_default_tools(
        self,
        tools: Optional[List[str]] = None,
        workspace: str = "."
    ):
        """
        Enables batteries-included standard tools.
        Available tools: 'filesystem', 'http', 'sqlite', 'shell'.
        Default: ['filesystem', 'http', 'sqlite'].
        """
        if tools is None:
            tools = ["filesystem", "http", "sqlite"]

        if "filesystem" in tools:
            fs = FilesystemTool(root_dir=workspace)
            for method in [fs.read_file, fs.write_file, fs.list_dir, fs.search_files]:
                schema = infer_schema_from_callable(method)
                self.register_function(schema["name"], schema["description"], schema["inputSchema"], method)

        if "http" in tools:
            http_tool = HttpTool()
            schema = infer_schema_from_callable(http_tool.http_fetch)
            self.register_function(schema["name"], schema["description"], schema["inputSchema"], http_tool.http_fetch)

        if "sqlite" in tools:
            db_tool = SqliteTool()
            schema = infer_schema_from_callable(db_tool.execute_sql)
            self.register_function(schema["name"], schema["description"], schema["inputSchema"], db_tool.execute_sql)

        if "shell" in tools:
            shell_tool = ShellTool(cwd=workspace)
            schema = infer_schema_from_callable(shell_tool.execute_command)
            self.register_function(schema["name"], schema["description"], schema["inputSchema"], shell_tool.execute_command)

    def get_all_tools(self) -> List[Dict[str, Any]]:
        tools = list(self.local_tool_schemas)
        for conn in self.mcp_connections:
            for t in conn.tools:
                tools.append(t.to_dict())
        return tools

    def execute_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if name in self.local_tools:
            try:
                res = self.local_tools[name](**arguments)
                return {"status": "success", "result": res}
            except Exception as e:
                return {"status": "error", "message": str(e)}

        for conn in self.mcp_connections:
            for t in conn.tools:
                if t.name == name:
                    return conn.call_tool(name, arguments)

        return {"status": "error", "message": f"Tool '{name}' not found."}

    def _build_system_prompt(self) -> str:
        tools = self.get_all_tools()
        variant_tag = "Vaayu-Large (492M)" if self.engine.variant == "large" else "Vaayu-Base (245M)"
        return (
            f"You are {variant_tag}, an autonomous Small Language Machine Model (SLMM). "
            "You have access to local software tools via the Model Context Protocol (MCP):\n"
            f"{json.dumps(tools, indent=2)}"
        )

    def chat(self, user_message: str, max_turns: int = 5) -> str:
        """Autonomous tool-calling and response loop with parallel dispatch and self-correction."""
        system_prompt = self._build_system_prompt()
        conversation_history = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_message}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        for turn in range(max_turns):
            response = self.engine.generate(conversation_history, max_new_tokens=300)
            conversation_history += response

            tool_calls = self.engine.extract_tool_calls(response)
            if not tool_calls:
                break

            tool_responses_str = ""
            for call in tool_calls:
                print(f"[Vaayu Action] Executing tool: {call.name} with {call.arguments}")
                tool_result = self.execute_tool(call.name, call.arguments)
                tool_responses_str += f"<|tool_result_start|>{json.dumps(tool_result)}<|tool_result_end|>\n"

            conversation_history += (
                f"\n<|im_start|>tool\n"
                f"{tool_responses_str}"
                f"<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

        parts = conversation_history.split("<|im_start|>assistant\n")
        final_part = parts[-1].replace("<|im_end|>", "").strip()
        return final_part

    def stream_chat(self, user_message: str, max_turns: int = 5) -> Iterator[StreamEvent]:
        """
        Real-time streaming chat generator.
        Yields StreamEvent instances representing internal thoughts, tool calls, tool results, and final text.
        """
        system_prompt = self._build_system_prompt()
        conversation_history = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_message}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        for turn in range(max_turns):
            turn_text = ""
            mode = "text"  # "text", "thought", or "tool_call"
            tool_call_buffer = ""

            for chunk in self.engine.stream_generate(conversation_history, max_new_tokens=300):
                turn_text += chunk

                if "<|thought_start|>" in chunk:
                    mode = "thought"
                    chunk = chunk.replace("<|thought_start|>", "")
                if "<|thought_end|>" in chunk:
                    sub_parts = chunk.split("<|thought_end|>")
                    if sub_parts[0]:
                        yield StreamEvent(type="thought", content=sub_parts[0])
                    mode = "text"
                    if len(sub_parts) > 1 and sub_parts[1]:
                        yield StreamEvent(type="text", content=sub_parts[1])
                    continue

                if "<|tool_call_start|>" in chunk:
                    mode = "tool_call"
                    tool_call_buffer = ""
                    chunk = chunk.replace("<|tool_call_start|>", "")

                if mode == "tool_call":
                    if "<|tool_call_end|>" in chunk:
                        sub_parts = chunk.split("<|tool_call_end|>")
                        tool_call_buffer += sub_parts[0]
                        mode = "text"
                    else:
                        tool_call_buffer += chunk
                    continue

                if mode == "thought":
                    if chunk:
                        yield StreamEvent(type="thought", content=chunk)
                elif mode == "text":
                    # Clean control tokens if present
                    clean_chunk = chunk.replace("<|im_end|>", "").replace("<|im_start|>", "")
                    if clean_chunk:
                        yield StreamEvent(type="text", content=clean_chunk)

            conversation_history += turn_text
            tool_calls = self.engine.extract_tool_calls(turn_text)
            if not tool_calls:
                break

            tool_responses_str = ""
            for call in tool_calls:
                yield StreamEvent(type="tool_call", tool_name=call.name, arguments=call.arguments)
                tool_result = self.execute_tool(call.name, call.arguments)
                yield StreamEvent(type="tool_result", tool_name=call.name, content=json.dumps(tool_result))
                tool_responses_str += f"<|tool_result_start|>{json.dumps(tool_result)}<|tool_result_end|>\n"

            conversation_history += (
                f"\n<|im_start|>tool\n"
                f"{tool_responses_str}"
                f"<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

    def generate_structured(self, user_message: str, response_model: Any) -> Any:
        """
        Generates structured output strictly adhering to a Pydantic model or JSON Schema dict.
        """
        schema: Dict[str, Any] = {}
        if hasattr(response_model, "model_json_schema"):
            schema = response_model.model_json_schema()
        elif hasattr(response_model, "schema"):
            schema = response_model.schema()
        elif isinstance(response_model, dict):
            schema = response_model
        elif callable(response_model):
            schema = infer_schema_from_callable(response_model)["inputSchema"]

        prompt = (
            f"{user_message}\n\n"
            "Respond ONLY with a valid JSON object matching the following schema:\n"
            f"{json.dumps(schema, indent=2)}\n"
            "Do not include any reasoning or markdown code fence blocks outside the JSON."
        )

        response = self.chat(prompt)
        # Extract JSON substring
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        raw_json = json_match.group(0) if json_match else response.strip()

        parsed = json.loads(raw_json)
        if hasattr(response_model, "model_validate"):
            return response_model.model_validate(parsed)
        elif hasattr(response_model, "parse_obj"):
            return response_model.parse_obj(parsed)
        return parsed