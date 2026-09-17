# Step 5: Embedding Vaayu into Any Local Software

## 1. Overview

Vaayu SLMM is designed to be an **in-process machine intelligence engine** rather than an external HTTP dependency. With only 245M parameters, it can be loaded directly inside Python applications, CLI utilities, background daemons, or desktop GUIs (Electron / PyQt / Tauri).

---

## 2. In-Process Python Embedding

### Installation
```bash
pip install -e .
```

### Basic Embedding Example
```python
from vaayu import Vaayu

# Load model locally
ai = Vaayu.load_local("checkpoints/vaayu_final.pt")

# Register custom in-process host functions
def search_workspace(query: str):
    return ["file1.py", "file2.py"]

ai.register_function(
    name="search_workspace",
    description="Search current files for keywords.",
    input_schema={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"]
    },
    func=search_workspace
)

# Run autonomous interaction
response = ai.chat("Find all files related to authentication.")
print(response)
```

---

## 3. Connecting to External MCP Servers

To connect to existing local servers (e.g. SQLite, GitHub, Filesystem):

```python
from vaayu import Vaayu

ai = Vaayu.load_local()

# Connect to standard Node.js or Python MCP server
ai.attach_mcp_server(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "D:/Projects/"]
)

# Model now has full tool-calling access to that filesystem!
result = ai.chat("Create a new directory called 'exports' and list contents.")
print(result)
```

---

## 4. Hardware Sizing & Memory Requirements

| Mode | Precision | RAM / VRAM Footprint | Target Devices |
| :--- | :--- | :--- | :--- |
| **FP32** | 32-bit | ~980 MB | Standard Workstations |
| **FP16** | 16-bit | ~490 MB | Laptops, GTX 1650, RTX series |
| **INT8** | 8-bit | ~245 MB | Edge SBCs, Raspberry Pi 5 |
| **INT4** | 4-bit | ~125 MB | Micro-devices & embedded IoT |