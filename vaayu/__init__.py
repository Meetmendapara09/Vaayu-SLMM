"""
Vaayu: Small Language Machine Model (SLMM)
A high-performance base machine model for in-process application embedding,
native Model Context Protocol (MCP) execution, and autonomous tool calling.
"""

__version__ = "1.0.5"

from .core import VaayuInferenceEngine, ToolCall
from .mcp_client import MCPServerConnection, MCPTool
from .embed import Vaayu
from .model.config import VaayuConfig, VaayuLargeConfig
from .model.architecture import VaayuForCausalLM

__all__ = [
    "Vaayu",
    "VaayuInferenceEngine",
    "ToolCall",
    "MCPServerConnection",
    "MCPTool",
    "VaayuConfig",
    "VaayuLargeConfig",
    "VaayuForCausalLM",
    "__version__",
]