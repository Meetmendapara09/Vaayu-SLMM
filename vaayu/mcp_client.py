"""
Model Context Protocol (MCP) native client bridge for Vaayu SLMM.
Enables connecting to any local software or external agent via stdio or HTTP/SSE.
"""

import json
import subprocess
import threading
import queue
from typing import Dict, Any, List, Optional

class MCPTool:
    def __init__(self, name: str, description: str, input_schema: Dict[str, Any]):
        self.name = name
        self.description = description
        self.input_schema = input_schema

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema
        }

class MCPServerConnection:
    """Manages an active stdio transport connection to a local MCP server."""
    def __init__(self, command: str, args: Optional[List[str]] = None, env: Optional[Dict[str, str]] = None):
        self.command = command
        self.args = args or []
        self.env = env
        self.process = None
        self.request_id = 0
        self.response_queues: Dict[int, queue.Queue] = {}
        self.tools: List[MCPTool] = []

    def start(self):
        full_cmd = [self.command] + self.args
        self.process = subprocess.Popen(
            full_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=self.env
        )
        self.listener_thread = threading.Thread(target=self._listen, daemon=True)
        self.listener_thread.start()
        self._initialize()

    def _listen(self):
        while self.process and self.process.poll() is None:
            line = self.process.stdout.readline()
            if not line:
                break
            line_str = line.strip()
            if not line_str:
                continue
            try:
                msg = json.loads(line_str)
                req_id = msg.get("id")
                if req_id in self.response_queues:
                    self.response_queues[req_id].put(msg)
            except json.JSONDecodeError:
                pass

    def send_request(self, method: str, params: Optional[Dict[str, Any]] = None, timeout: float = 10.0) -> Dict[str, Any]:
        self.request_id += 1
        req_id = self.request_id
        q = queue.Queue()
        self.response_queues[req_id] = q

        payload = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params or {}
        }
        raw_req = json.dumps(payload) + "\n"
        self.process.stdin.write(raw_req)
        self.process.stdin.flush()

        try:
            resp = q.get(timeout=timeout)
            return resp.get("result", {})
        finally:
            del self.response_queues[req_id]

    def _initialize(self):
        init_params = {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "roots": {"listChanged": True},
                "sampling": {}
            },
            "clientInfo": {
                "name": "Vaayu-SLMM-Client",
                "version": "1.0.0"
            }
        }
        res = self.send_request("initialize", init_params)
        self.refresh_tools()

    def refresh_tools(self) -> List[MCPTool]:
        res = self.send_request("tools/list")
        raw_tools = res.get("tools", [])
        self.tools = [
            MCPTool(
                name=t.get("name"),
                description=t.get("description", ""),
                input_schema=t.get("inputSchema", {})
            )
            for t in raw_tools
        ]
        return self.tools

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        params = {"name": name, "arguments": arguments}
        return self.send_request("tools/call", params)

    def close(self):
        if self.process:
            self.process.terminate()
            self.process = None