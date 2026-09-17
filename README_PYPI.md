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
- **In-Process Tool Decorator (`@ai.tool`)**: Register standard Python functions directly with automatic JSON schema and type hint introspection.
- **Batteries-Included Tools**: Sandboxed Filesystem, Shell execution, SQLite query engine, and HTTP REST tools ready out-of-the-box (`ai.enable_default_tools()`).
- **Real-Time Streaming & Observability**: Stream thoughts, tool calls, tool results, and text tokens via `stream_chat` and `StreamEvent`.
- **OpenAI-Compatible Local REST Server**: Drop-in OpenAI API replacement (`vaayu serve`) compatible with LangChain, LlamaIndex, Ollama UIs, and CrewAI.
- **Structured Pydantic Validation**: Force machine outputs to strictly conform to Pydantic models or JSON schemas via `generate_structured()`.
- **Native MCP Support**: Direct, first-class connection to Model Context Protocol servers over standard I/O (`stdio`).
- **Ultra-Low Latency**: Sub-50ms Time-To-First-Token (TTFT) on consumer CPUs with Grouped Query Attention (4:1 GQA).

---

## Installation

```bash
pip install vaayu
```

---

## Quickstart Guide

### 1. In-Process Tools with Python Decorator (`@ai.tool`)

Register any standard Python function as an executable tool. Type hints and docstrings are automatically parsed into JSON Schema:

```python
from vaayu import Vaayu

ai = Vaayu.from_pretrained("meetmendapara/Vaayu-Base")

@ai.tool
def get_stock_quote(ticker: str, currency: str = "USD") -> str:
    """Fetches real-time market quote for a stock ticker."""
    return f"{ticker}: $185.40 {currency}"

response = ai.chat("What is the current stock quote for AAPL?")
print(response)
```

### 2. Batteries-Included Standard Tools

Equip Vaayu with sandboxed filesystem access, HTTP fetch, and SQLite capabilities in a single line:

```python
from vaayu import Vaayu

ai = Vaayu.from_pretrained("meetmendapara/Vaayu-Base")

# Enables Filesystem, HTTP, and SQLite tools sandboxed to current directory
ai.enable_default_tools(workspace="./data")

response = ai.chat("Search for all *.json files in the workspace and inspect their content.")
print(response)
```

### 3. Real-Time Streaming & Observability

Monitor thoughts, tool invocations, and text generation as they occur:

```python
from vaayu import Vaayu

ai = Vaayu.from_pretrained("meetmendapara/Vaayu-Base")

for event in ai.stream_chat("Fetch the latest git commit logs"):
    if event.type == "thought":
        print(f"[Reasoning] {event.content}", end="", flush=True)
    elif event.type == "tool_call":
        print(f"\n[Invoking {event.tool_name} with {event.arguments}]")
    elif event.type == "tool_result":
        print(f"[Result: {event.content}]")
    elif event.type == "text":
        print(event.content, end="", flush=True)
```

### 4. Pydantic Structured Outputs

Enforce strict JSON schema compliance with Pydantic models:

```python
from pydantic import BaseModel
from vaayu import Vaayu

class UserExtraction(BaseModel):
    name: str
    email: str
    role: str

ai = Vaayu.from_pretrained("meetmendapara/Vaayu-Base")
user = ai.generate_structured(
    "Extract user info: Alex Morgan (alex.morgan@example.com) is a Lead Systems Architect.",
    response_model=UserExtraction
)
print(user.name, user.role)
# Alex Morgan Lead Systems Architect
```

### 5. Native Model Context Protocol (MCP) Tools

Connect directly to external MCP servers running via `stdio`:

```python
from vaayu import Vaayu

ai = Vaayu.from_pretrained("meetmendapara/Vaayu-Base")

# Connect to any local MCP server
ai.attach_mcp_server(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "./workspace"]
)

result = ai.chat("Read the file config.json and list all defined settings.")
print(result)
```

---

## OpenAI-Compatible Local HTTP Server

Serve Vaayu locally as an OpenAI-compatible REST API:

```bash
# Start server on port 8000
vaayu serve --port 8000 --model meetmendapara/Vaayu-Base
```

Use with the official `openai` Python SDK, LangChain, or curl:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="vaayu-base",
    messages=[
        {"role": "user", "content": "Hello! List the files in the directory."}
    ]
)
print(response.choices[0].message.content)
```

---

## Interactive Developer REPL

Test prompts and tools in an interactive terminal session:

```bash
vaayu repl
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
