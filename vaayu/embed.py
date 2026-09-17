"""
High-level software embedding interface for Vaayu SLMM & Vaayu-Large.
Supports:
- Vaayu.from_pretrained("meetmendapara/Vaayu-Base")
- Vaayu.load_local("path/to/weights.pt")
- In-process tool calling, parallel execution, automated error self-correction, and full MCP orchestration.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable

from .core import VaayuInferenceEngine, ToolCall
from .mcp_client import MCPServerConnection, MCPTool

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
        self.local_tool_schemas.append({
            "name": name,
            "description": description,
            "inputSchema": input_schema
        })

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

    def chat(self, user_message: str, max_turns: int = 5) -> str:
        """Autonomous tool-calling and response loop with parallel dispatch and self-correction."""
        tools = self.get_all_tools()
        variant_tag = "Vaayu-Large (492M)" if self.engine.variant == "large" else "Vaayu-Base (245M)"
        system_prompt = (
            f"You are {variant_tag}, an autonomous Small Language Machine Model (SLMM). "
            "You have access to local software tools via the Model Context Protocol (MCP):\n"
            f"{json.dumps(tools, indent=2)}"
        )

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