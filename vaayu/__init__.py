"""
Vaayu: Small Language Machine Model (SLMM)
A high-performance base machine model for in-process application embedding,
native Model Context Protocol (MCP) execution, and autonomous tool calling.
"""

__version__ = "1.1.0"

from .core import VaayuInferenceEngine, ToolCall
from .mcp_client import MCPServerConnection, MCPTool
from .embed import Vaayu, StreamEvent
from .model.config import VaayuConfig, VaayuLargeConfig
from .model.architecture import VaayuForCausalLM
from .tools import (
    infer_schema_from_callable,
    FilesystemTool,
    ShellTool,
    HttpTool,
    SqliteTool,
)
from .server import start_server

__all__ = [
    "Vaayu",
    "StreamEvent",
    "VaayuInferenceEngine",
    "ToolCall",
    "MCPServerConnection",
    "MCPTool",
    "VaayuConfig",
    "VaayuLargeConfig",
    "VaayuForCausalLM",
    "infer_schema_from_callable",
    "FilesystemTool",
    "ShellTool",
    "HttpTool",
    "SqliteTool",
    "start_server",
    "__version__",
]