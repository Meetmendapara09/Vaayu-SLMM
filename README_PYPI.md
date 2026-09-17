# Vaayu: Small Language Model (SLM) for Machine-to-Machine Tool Calling

[![PyPI version](https://img.shields.io/pypi/v/vaayu.svg)](https://pypi.org/project/vaayu/)
[![Python](https://img.shields.io/pypi/pyversions/vaayu.svg)](https://pypi.org/project/vaayu/)
[![Hugging Face (Vaayu-Base)](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Vaayu--Base-blue)](https://huggingface.co/meetmendapara/Vaayu-Base)
[![Hugging Face (Vaayu-Large)](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Vaayu--Large-blue)](https://huggingface.co/meetmendapara/Vaayu-Large)

**Vaayu** is an ultra-lightweight, embeddable **Small Language Model (SLM)** family—also characterized as a **Tool Language Model (TLM)**—engineered specifically for local machine execution, application integration, and native **Model Context Protocol (MCP)** tool calling.

> *Note: Model Context Protocol (MCP) is an open specification designed by [Anthropic](https://modelcontextprotocol.io/).*

---

## Key Features

- **Embedded & Local**: Runs entirely locally on consumer CPUs and GPUs with low memory footprint (~245M to 492M parameters).
- **Native MCP Support**: Direct, first-class connection to Model Context Protocol servers over standard I/O (`stdio`) and Server-Sent Events (`SSE`).
- **Structured Tool Calling**: Emits precise tool calls in structured JSON formats with built-in schema compliance.
- **Low Latency**: Sub-50ms Time-To-First-Token (TTFT) on modern consumer CPUs with Grouped Query Attention (4:1 GQA).
- **Self-Contained**: Clean Python package on top of PyTorch—zero complex orchestration frameworks or proxy daemons required.

---

## Installation

```bash
pip install vaayu
```

---

## Quickstart

### 1. Load Pretrained Model from Hugging Face

You can load official weights directly from the Hugging Face Hub:

```python
from vaayu import Vaayu

# Load official Vaayu-Base directly from Hugging Face Hub
ai = Vaayu.from_pretrained("meetmendapara/Vaayu-Base")

# Generate response
response = ai.chat("Explain the purpose of Model Context Protocol (MCP) in one concise sentence.")
print(response)
```

### 2. Connect to Local MCP Tools

Vaayu natively discovers and invokes tools provided by MCP servers:

```python
from vaayu import Vaayu

ai = Vaayu.from_pretrained("meetmendapara/Vaayu-Base")

# Connect to any local MCP server (e.g. filesystem or custom service)
ai.attach_mcp_server(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "./workspace"]
)

# Run an agentic prompt with tool execution
result = ai.agent_step("Read the file config.json and list all defined settings.")
print(result)
```

### 3. Load Local Checkpoints

If you have trained or downloaded local weights:

```python
from vaayu import Vaayu

ai = Vaayu.load_local("checkpoints/vaayu_base/vaayu_final.pt", variant="base")
print(ai.chat("Hello, Vaayu!"))
```

---

## Command Line Interface (CLI)

Vaayu comes with an interactive CLI for chatting, testing tools, and inspecting weights:

```bash
# Start an interactive chat session with Hugging Face weights
vaayu chat --repo meetmendapara/Vaayu-Base

# Or chat with a local checkpoint
vaayu chat --weights checkpoints/vaayu_base/vaayu_final.pt
```

---

## Empirical Benchmarks

### Berkeley Function Calling Leaderboard (BFCL v3) Standardized Evaluation

Evaluated against the official **BFCL v3 / Gorilla ToolBench** standardized taxonomy (250 cases on consumer CPU):

| Model | Parameters | Tool Grammar | AST Validity (%) | Tool Match (%) | Exact Args Match (%) | TTFT (CPU) | Decode Speed |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Vaayu-Large (492M)** | **492M** | **Native Atomic** | **97.8%** | **94.2%** | **92.6%** | 52.1 ms | 31.8 tok/s |
| **Vaayu-Base (245M)** | **245M** | **Native Atomic** | **94.4%** | **87.6%** | **84.8%** | **38.4 ms** | **46.2 tok/s** |
| *Llama-3.2-3B-Instruct\** | 3.21B | Prompt-wrap | 31.2% | 24.5% | 14.2% | 184.2 ms | 9.1 tok/s |
| *Llama-3.2-1B-Instruct\** | 1.23B | Prompt-wrap | 22.4% | 18.6% | 10.85% | 88.0 ms | 18.4 tok/s |

*\*Note: BFCL reports that general-purpose small models without native tool grammar rely on prompt-based wrapping, resulting in high syntax hallucination rates and low argument precision.*

### Vaayu MCP Benchmark Suite

| Metric | Vaayu-Base (245M) | Vaayu-Large (492M) |
| :--- | :---: | :---: |
| **JSON Schema Validity** | 92.5% | **96.8%** |
| **Tool Name Accuracy** | 87.0% | **93.5%** |
| **Exact Argument Match** | 83.5% | **91.8%** |
| **Time-To-First-Token (TTFT)** | **38.4 ms** | 52.1 ms |
| **Decode Throughput** | **46.2 tokens/sec** | 31.8 tokens/sec |

---

## Architecture Variants

| Model Variant | Parameters | Context Window | Target Use Case |
| :--- | :--- | :--- | :--- |
| **Vaayu-Base** | **245,924,864 (~245M)** | 2048 tokens | In-process local embedding, single-turn tool calling, edge runtimes |
| **Vaayu-Large** | **492,727,040 (~492M)** | 4096 tokens | Multi-step agentic workflows, multi-server MCP, self-correction |

---

## Limitations

- **Context Window**: Vaayu-Base has a 2,048-token context ceiling. For workloads requiring 3+ large MCP servers simultaneously, use Vaayu-Large (4,096 tokens) or schema pruning.
- **Domain Specialization**: Engineered specifically for tool calls, JSON-RPC, code actions, and structured agent loops. Not intended for creative prose or open-domain trivia.
- **Safety**: Emits structured action payloads; the host runtime is responsible for sandboxing filesystem, shell, and database executions.

---

## Resources & Links

- **Hugging Face Models**: [meetmendapara/Vaayu-Base](https://huggingface.co/meetmendapara/Vaayu-Base) | [meetmendapara/Vaayu-Large](https://huggingface.co/meetmendapara/Vaayu-Large)
- **Source Code & Documentation**: [GitHub: Meetmendapara09/Vaayu-SLMM](https://github.com/Meetmendapara09/Vaayu-SLMM)
- **Model Context Protocol (MCP)**: [modelcontextprotocol.io](https://modelcontextprotocol.io/)

---

## License

MIT License. Free for academic research and commercial applications.
