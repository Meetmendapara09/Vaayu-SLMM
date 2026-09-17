# Step 8: Standalone Python Package & PyPI Distribution

## 1. Overview

To make **Vaayu** universally consumable across diverse Python applications, IDE extensions, edge runtimes, and local AI agent pipelines, the codebase was structured into a production-grade Python package published on **PyPI (Python Package Index)**:

- **PyPI Package**: [**`https://pypi.org/project/vaayu/`**](https://pypi.org/project/vaayu/)
- **Installation**: `pip install vaayu`
- **Hugging Face Hub**: [**`meetmendapara/Vaayu-Base`**](https://huggingface.co/meetmendapara/Vaayu-Base)
- **GitHub Repository**: [**`Meetmendapara09/Vaayu-SLMM`**](https://github.com/Meetmendapara09/Vaayu-SLMM)

---

## 2. Package Architecture

The `vaayu` package is packaged with PEP 517/621 standards using modern `setuptools` and `build`:

```
vaayu/
├── __init__.py           # Package exports (Vaayu, VaayuConfig, VaayuForCausalLM, __version__)
├── core.py               # Local inference engine with KV-caching and sampling logic
├── embed.py              # In-process software embedding API (Vaayu.from_pretrained, Vaayu.load_local)
├── mcp_client.py         # Native Model Context Protocol (stdio/SSE) client bridge
├── cli.py                # Command-Line Interface (vaayu chat, vaayu info)
├── model/
│   ├── __init__.py
│   ├── architecture.py   # Transformer decoder with RoPE, GQA, SwiGLU, RMSNorm
│   └── config.py         # Hyperparameter configurations for Vaayu-Base & Vaayu-Large
└── tokenizer/
    ├── __init__.py
    ├── tokenizer.json    # Bundled BPE tokenizer with MCP control tokens
    └── tokenizer_config.json
```

---

## 3. High-Level Python API

### 3.1 Loading from Hugging Face Hub Directly

```python
from vaayu import Vaayu

# Automatically streams architecture, tokenizer, and weights from Hugging Face Hub
ai = Vaayu.from_pretrained("meetmendapara/Vaayu-Base")

# Single-turn or multi-turn generation
response = ai.chat("Explain Model Context Protocol (MCP) in two concise sentences.")
print(response)
```

### 3.2 Loading Local Weights

```python
from vaayu import Vaayu

# Load local checkpoint (.pt or .bin)
ai = Vaayu.load_local("checkpoints/vaayu_base/vaayu_final.pt", variant="base")
print(ai.chat("Hello!"))
```

### 3.3 Connecting Native Model Context Protocol (MCP) Servers

```python
from vaayu import Vaayu

ai = Vaayu.from_pretrained("meetmendapara/Vaayu-Base")

# Connect to any standard MCP server over stdio
ai.attach_mcp_server(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "./"]
)

# Autonomous tool discovery, dispatch, and response formatting
result = ai.agent_step("Inspect the current workspace and list all markdown files.")
print(result)
```

---

## 4. Interactive Command Line Interface (CLI)

The package installs the `vaayu` executable entry point for terminal interaction:

```bash
# Chat with Hugging Face model
vaayu chat --repo meetmendapara/Vaayu-Base

# Chat with local checkpoint
vaayu chat --weights checkpoints/vaayu_base/vaayu_final.pt

# Inspect model configuration
vaayu info --repo meetmendapara/Vaayu-Base
```

---

## 5. Build, Verification & Release Pipeline

The distribution lifecycle uses PEP 517 build tooling and Twine:

1. **Build Distribution**:
   ```bash
   python -m build
   ```
   Produces:
   - `dist/vaayu-<version>-py3-none-any.whl` (Pure Python Wheel)
   - `dist/vaayu-<version>.tar.gz` (Source Distribution)

2. **Validation**:
   ```bash
   python -m twine check dist/*
   ```

3. **Secure Publication**:
   ```bash
   python scripts/publish_to_pypi.py --token <PYPI_API_TOKEN>
   ```
