"""
OpenAI-compatible HTTP REST API Server for Vaayu SLMM.
Implements:
- GET /v1/models
- POST /v1/chat/completions (Non-streaming & Server-Sent Events (SSE) streaming)
- GET /health
Zero external dependencies (uses standard library http.server).
"""

import sys
import json
import time
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any, List, Optional, Union

from .embed import Vaayu

class VaayuHTTPHandler(BaseHTTPRequestHandler):
    vaayu_instance: Optional[Vaayu] = None

    def _set_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def do_OPTIONS(self):
        self.send_response(200)
        self._set_cors_headers()
        self.end_headers()

    def do_GET(self):
        if self.path in ("/", "/health"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._set_cors_headers()
            self.end_headers()
            variant = self.vaayu_instance.engine.variant if self.vaayu_instance else "unknown"
            res = {
                "status": "ok",
                "service": "Vaayu SLMM Inference Server",
                "version": "1.1.0",
                "variant": variant
            }
            self.wfile.write(json.dumps(res, indent=2).encode("utf-8"))
            return

        if self.path.startswith("/v1/models"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._set_cors_headers()
            self.end_headers()
            res = {
                "object": "list",
                "data": [
                    {
                        "id": "vaayu-base",
                        "object": "model",
                        "created": 1726500000,
                        "owned_by": "meetmendapara",
                        "permission": []
                    },
                    {
                        "id": "vaayu-large",
                        "object": "model",
                        "created": 1726500000,
                        "owned_by": "meetmendapara",
                        "permission": []
                    }
                ]
            }
            self.wfile.write(json.dumps(res, indent=2).encode("utf-8"))
            return

        self.send_response(404)
        self.send_header("Content-Type", "application/json")
        self._set_cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps({"error": "Endpoint not found"}).encode("utf-8"))

    def do_POST(self):
        if not self.path.startswith("/v1/chat/completions"):
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self._set_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Endpoint not found"}).encode("utf-8"))
            return

        content_len = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(content_len).decode("utf-8")
        try:
            payload = json.loads(raw_body)
        except Exception as e:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self._set_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"error": f"Invalid JSON body: {e}"}).encode("utf-8"))
            return

        messages: List[Dict[str, str]] = payload.get("messages", [])
        tools: List[Dict[str, Any]] = payload.get("tools", [])
        stream: bool = payload.get("stream", False)
        max_tokens: int = payload.get("max_tokens", 300)
        model_id = payload.get("model", "vaayu-base")

        # Construct prompt from messages
        prompt = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"

        req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created_time = int(time.time())

        if stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self._set_cors_headers()
            self.end_headers()

            for chunk in self.vaayu_instance.engine.stream_generate(prompt, max_new_tokens=max_tokens):
                clean_chunk = chunk.replace("<|im_end|>", "").replace("<|im_start|>", "")
                if not clean_chunk:
                    continue
                sse_data = {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": model_id,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": clean_chunk},
                            "finish_reason": None
                        }
                    ]
                }
                self.wfile.write(f"data: {json.dumps(sse_data)}\n\n".encode("utf-8"))
                self.wfile.flush()

            # End of stream chunk
            end_chunk = {
                "id": req_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": model_id,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }
                ]
            }
            self.wfile.write(f"data: {json.dumps(end_chunk)}\n\n".encode("utf-8"))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return

        # Non-streaming response
        output_text = self.vaayu_instance.engine.generate(prompt, max_new_tokens=max_tokens)
        tool_calls = self.vaayu_instance.engine.extract_tool_calls(output_text)

        finish_reason = "stop"
        formatted_tool_calls = None

        if tool_calls:
            finish_reason = "tool_calls"
            formatted_tool_calls = []
            for call in tool_calls:
                formatted_tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments)
                    }
                })

        clean_text = output_text.replace("<|im_end|>", "").strip()
        message_dict = {
            "role": "assistant",
            "content": clean_text
        }
        if formatted_tool_calls:
            message_dict["tool_calls"] = formatted_tool_calls

        response_body = {
            "id": req_id,
            "object": "chat.completion",
            "created": created_time,
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "message": message_dict,
                    "finish_reason": finish_reason
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(output_text.split()),
                "total_tokens": len(prompt.split()) + len(output_text.split())
            }
        }

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self._set_cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps(response_body, indent=2).encode("utf-8"))

    def log_message(self, format, *args):
        # Override to suppress default noisy request logging in console
        sys.stderr.write(f"[Vaayu HTTP Server] {self.address_string()} - {format % args}\n")

def start_server(
    model: Union[str, Vaayu] = "meetmendapara/Vaayu-Base",
    host: str = "0.0.0.0",
    port: int = 8000,
    variant: str = "base",
    device: Optional[str] = None
):
    """Starts the local OpenAI-compatible HTTP server."""
    if isinstance(model, Vaayu):
        ai = model
    else:
        print(f"[Vaayu Server] Loading model '{model}'...")
        ai = Vaayu.from_pretrained(model, variant=variant, device=device)

    VaayuHTTPHandler.vaayu_instance = ai
    server_address = (host, port)
    httpd = HTTPServer(server_address, VaayuHTTPHandler)

    print(f"==================================================")
    print(f" Vaayu SLMM OpenAI-Compatible Server Running")
    print(f" URL: http://{host if host != '0.0.0.0' else '127.0.0.1'}:{port}")
    print(f" Endpoints:")
    print(f"   - GET  /health")
    print(f"   - GET  /v1/models")
    print(f"   - POST /v1/chat/completions (streaming & non-streaming)")
    print(f" Compatible with: OpenAI Python SDK, LangChain, LlamaIndex, Ollama-UI")
    print(f"==================================================")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[Vaayu Server] Shutting down...")
        httpd.server_close()
