"""
Automated unit test suite for Vaayu SDK features:
- In-process tool schema inference & @ai.tool decorator
- Batteries-included standard tools (Filesystem, Shell, SQLite, HTTP)
- Real-time streaming events (StreamEvent)
- Local OpenAI-compatible HTTP server handler
"""

import os
import tempfile
import unittest
from typing import List, Optional

from vaayu import (
    Vaayu,
    StreamEvent,
    FilesystemTool,
    ShellTool,
    SqliteTool,
    infer_schema_from_callable,
)
from vaayu.server import VaayuHTTPHandler

class DummyTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [1, 2, 3]
    def decode(self, tokens, skip_special_tokens=False):
        return "mock decoded response"

class DummyEngine:
    def __init__(self, variant="base"):
        self.variant = variant
        self.tokenizer = DummyTokenizer()
        self.im_end_id = 1000
        self.tool_call_end_id = 1004

    def generate(self, prompt: str, max_new_tokens: int = 300) -> str:
        return "I am a mock response <|im_end|>"

    def stream_generate(self, prompt: str, max_new_tokens: int = 300):
        yield "I am a "
        yield "streamed response "
        yield "<|im_end|>"

    def extract_tool_calls(self, text: str):
        return []

class TestVaayuSDK(unittest.TestCase):

    def setUp(self):
        self.dummy_engine = DummyEngine()
        self.ai = Vaayu(self.dummy_engine)

    def test_schema_inference(self):
        def sample_func(city: str, days: int = 5) -> str:
            """Fetches weather forecast for a specified city."""
            return f"{city}: {days} days sunny"

        schema = infer_schema_from_callable(sample_func)
        self.assertEqual(schema["name"], "sample_func")
        self.assertIn("weather forecast", schema["description"])
        props = schema["inputSchema"]["properties"]
        self.assertIn("city", props)
        self.assertEqual(props["city"]["type"], "string")
        self.assertIn("days", props)
        self.assertEqual(props["days"]["type"], "integer")
        self.assertEqual(props["days"]["default"], 5)
        self.assertEqual(schema["inputSchema"]["required"], ["city"])

    def test_tool_decorator(self):
        @self.ai.tool
        def calculate_tax(amount: float, rate: float = 0.18) -> float:
            """Calculates total tax for an invoice."""
            return amount * rate

        self.assertIn("calculate_tax", self.ai.local_tools)
        res = self.ai.execute_tool("calculate_tax", {"amount": 100.0, "rate": 0.2})
        self.assertEqual(res["status"], "success")
        self.assertAlmostEqual(res["result"], 20.0)

        @self.ai.tool(name="custom_multiply", description="Multiplies two numbers")
        def mul(a: int, b: int) -> int:
            return a * b

        self.assertIn("custom_multiply", self.ai.local_tools)
        res_mul = self.ai.execute_tool("custom_multiply", {"a": 6, "b": 7})
        self.assertEqual(res_mul["status"], "success")
        self.assertEqual(res_mul["result"], 42)

    def test_filesystem_tool_sandbox(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fs = FilesystemTool(root_dir=tmpdir)
            # Write file
            write_res = fs.write_file("sub/test.txt", "Hello Vaayu SLMM")
            self.assertIn("Successfully wrote", write_res)

            # Read file
            content = fs.read_file("sub/test.txt")
            self.assertEqual(content, "Hello Vaayu SLMM")

            # List dir
            items = fs.list_dir(".")
            self.assertIn("sub/", items)

            # Search
            matches = fs.search_files("*.txt", path=".")
            self.assertIn("sub/test.txt", matches)

            # Security sandbox test
            with self.assertRaises(PermissionError):
                fs.read_file("../../outside.txt")

    def test_sqlite_tool(self):
        db = SqliteTool(db_path=":memory:")
        res = db.execute_sql("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT);")
        self.assertEqual(res["status"], "success")

        db.execute_sql("INSERT INTO users (name) VALUES ('Alice');")
        db.execute_sql("INSERT INTO users (name) VALUES ('Bob');")

        select_res = db.execute_sql("SELECT * FROM users ORDER BY id;")
        self.assertEqual(select_res["status"], "success")
        self.assertEqual(select_res["row_count"], 2)
        self.assertEqual(select_res["rows"][0]["name"], "Alice")
        self.assertEqual(select_res["rows"][1]["name"], "Bob")

    def test_shell_tool(self):
        shell = ShellTool(timeout=5.0)
        res = shell.execute_command("echo hello_vaayu")
        self.assertEqual(res["status"], "success")
        self.assertIn("hello_vaayu", res["stdout"])

    def test_enable_default_tools(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.ai.enable_default_tools(workspace=tmpdir)
            tools = self.ai.get_all_tools()
            tool_names = [t["name"] for t in tools]
            self.assertIn("read_file", tool_names)
            self.assertIn("write_file", tool_names)
            self.assertIn("http_fetch", tool_names)
            self.assertIn("execute_sql", tool_names)

    def test_streaming_chat(self):
        events = list(self.ai.stream_chat("Hello world"))
        text_events = [e for e in events if e.type == "text"]
        self.assertTrue(len(text_events) > 0)
        full_text = "".join(e.content for e in text_events)
        self.assertIn("streamed response", full_text)

if __name__ == "__main__":
    unittest.main()
