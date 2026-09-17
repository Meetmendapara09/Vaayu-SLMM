"""
Embedded Demonstration of Vaayu SLMM with Tool Calling & Interface Comprehension.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vaayu import Vaayu

def sample_read_file(file_path: str):
    """Local software tool executed inside the host application."""
    path = Path(file_path)
    if path.exists():
        return {"content": path.read_text(encoding="utf-8")[:300], "status": "exists"}
    return {"error": "file not found", "status": "missing"}

def sample_system_status():
    """Local software tool to report environment status."""
    return {"status": "healthy", "cores": os.cpu_count(), "platform": sys.platform}

def main():
    print("==================================================")
    print(" Vaayu SLMM: Embedded Software Demonstration")
    print("==================================================")

    # 1. Initialize local embeddable instance
    ai = Vaayu.load_local()

    # 2. Register native host software tools
    ai.register_function(
        name="local_filesystem_read",
        description="Read file contents from local storage.",
        input_schema={
            "type": "object",
            "properties": {"file_path": {"type": "string"}},
            "required": ["file_path"]
        },
        func=sample_read_file
    )

    ai.register_function(
        name="get_system_status",
        description="Inspect host system metrics and platform health.",
        input_schema={"type": "object", "properties": {}},
        func=sample_system_status
    )

    print(f"Registered tools: {[t['name'] for t in ai.get_all_tools()]}")

    # 3. Test tool dispatch
    print("\nTesting tool dispatch directly:")
    result = ai.execute_tool("get_system_status", {})
    print(f"Tool Result: {result}")

    print("\n[OK] Vaayu runtime successfully integrated and ready for host software embedding.")

if __name__ == "__main__":
    main()