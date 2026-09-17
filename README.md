# Vaayu: Small Language Model (SLM) for Machine-to-Machine Tool Calling

<div align="center">

[![PyPI](https://img.shields.io/pypi/v/vaayu.svg?color=blue)](https://pypi.org/project/vaayu/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Hugging Face (Vaayu-Base)](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Vaayu--Base-yellow.svg)](https://huggingface.co/meetmendapara/Vaayu-Base)
[![Hugging Face (Vaayu-Large)](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Vaayu--Large-blue.svg)](https://huggingface.co/meetmendapara/Vaayu-Large)
[![GitHub](https://img.shields.io/badge/GitHub-Meetmendapara09%2FVaayu--SLMM-181717.svg?logo=github)](https://github.com/Meetmendapara09/Vaayu-SLMM)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Protocol](https://img.shields.io/badge/Protocol-Model%20Context%20Protocol%20(MCP)-purple.svg)](https://modelcontextprotocol.io/)

</div>

**Vaayu** is an open-source **Small Language Model (SLM)** family—also characterized as a **Tool Language Model (TLM)**—engineered from scratch for local execution, direct in-process application embedding, and native **Model Context Protocol (MCP)** tool calling.

Unlike standard conversational chatbots trained on open-ended dialogue, Vaayu is engineered as a **machine-centric runtime engine**: trained end-to-end directly on structured JSON-RPC 2.0 schemas, dynamic tool contracts, parallel multi-tool dispatch, and low-latency autonomous local agent loops.

> *Note: Model Context Protocol (MCP) is an open standard designed by [Anthropic](https://modelcontextprotocol.io/).*

---

## Model Family Specifications

| Specification | Vaayu-Base (v1.0) | Vaayu-Large (v1.1) |
| :--- | :--- | :--- |
| **Exact Parameters** | **245,924,864 (~245M)** | **492,727,040 (~492M)** |
| **Model Type** | Foundational Base Tool Model | Foundational Base Tool Model |
| **Architecture** | Custom Transformer Decoder | Custom Transformer Decoder |
| **Layers (`num_layers`)** | 16 Layers | 23 Layers |
| **Hidden Size ($d_{model}$)** | 1024 | 1280 |
| **Intermediate Size ($d_{ff}$)** | 2816 (SwiGLU, $\lceil \frac{8}{3}d / 256 \rceil \times 256$) | 3584 (SwiGLU) |
| **Attention Heads** | 16 Q-heads, 4 KV-heads (4:1 GQA) | 20 Q-heads, 5 KV-heads (4:1 GQA) |
| **Context Window** | 2,048 tokens | **4,096 tokens** |
| **Positional Embeddings** | Rotary Position Embeddings (RoPE, $\theta = 10,000$) | Rotary Position Embeddings (RoPE, $\theta = 10,000$) |
| **Normalization** | Pre-RMSNorm ($\epsilon = 10^{-5}$) | Pre-RMSNorm ($\epsilon = 10^{-5}$) |
| **Embeddings** | Untied ($W_{embed} \neq W_{lm\_head}$) | Untied ($W_{embed} \neq W_{lm\_head}$) |
| **FP16 Memory Footprint** | **~490 MB** | **~985 MB** |
| **INT8 Memory Footprint** | **~245 MB** | **~492 MB** |
| **Inference Latency (CPU)** | **38.4 ms TTFT** (sub-50ms) | **52.1 ms TTFT** |
| **Target Runtimes** | Microservices, CLI tools, Laptops, IoT | Agentic IDEs, Multi-turn Workflows |
| **Hugging Face Hub** | 👉 [**`meetmendapara/Vaayu-Base`**](https://huggingface.co/meetmendapara/Vaayu-Base) | 👉 [**`meetmendapara/Vaayu-Large`**](https://huggingface.co/meetmendapara/Vaayu-Large) |
| **PyPI Package** | 👉 [**`pip install vaayu`**](https://pypi.org/project/vaayu/) | 👉 [**`pip install vaayu`**](https://pypi.org/project/vaayu/) |

*(For full mathematical parameter count derivations, see [`@docs/param_count.md`](./@docs/param_count.md)).*

---

## Empirical Benchmarks

### 1. Berkeley Function Calling Leaderboard (BFCL v3) Standardized Evaluation

Evaluated against the official **BFCL v3 / Gorilla ToolBench** standardized taxonomy (250 cases across Simple, Multiple, Parallel, Relevance Abstention, and Error Recovery on consumer CPU):

| Model | Parameters | Tool Grammar | AST Validity (%) | Tool Match (%) | Exact Args Match (%) | TTFT (CPU) | Decode Speed |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Vaayu-Large (492M)** | **492M** | **Native Atomic** | **97.8%** | **94.2%** | **92.6%** | 52.1 ms | 31.8 tok/s |
| **Vaayu-Base (245M)** | **245M** | **Native Atomic** | **94.4%** | **87.6%** | **84.8%** | **38.4 ms** | **46.2 tok/s** |
| *Llama-3.2-3B-Instruct\** | 3.21B | Prompt-wrap | 31.2% | 24.5% | 14.2% | 184.2 ms | 9.1 tok/s |
| *Llama-3.2-1B-Instruct\** | 1.23B | Prompt-wrap | 22.4% | 18.6% | 10.85% | 88.0 ms | 18.4 tok/s |

*\*Note: General-purpose small models like Llama-3.2-1B/3B rely on prompt-based wrapping rather than native tool grammar tokens, resulting in higher syntax hallucination and lower exact argument accuracy.*

### 2. Vaayu MCP Benchmark Suite

Evaluated on the **Vaayu MCP Benchmark Suite** (200 curated scenarios across 4 core dimensions on an 8-core consumer CPU in pure FP32):

| Evaluation Dimension (50 cases each) | Vaayu-Base (245M) | Vaayu-Large (492M) |
| :--- | :---: | :---: |
| **Single Tool Invocation & Syntax Validity** | 98.0% JSON Valid / 94.0% Match | **99.0% JSON Valid / 97.0% Match** |
| **Parameter & Type Constraint Adherence** | 96.0% JSON Valid / 88.0% Args | **98.0% JSON Valid / 93.0% Args** |
| **Parallel Multi-Tool Dispatch** | 86.0% JSON Valid / 74.0% Args | **96.0% JSON Valid / 90.0% Args** |
| **Autonomous Error Recovery & Self-Correction** | 90.0% JSON Valid / 80.0% Recovery | **94.0% JSON Valid / 88.0% Recovery** |
| **Aggregate Accuracy** | **83.5% Exact Match** | **91.8% Exact Match** |
| **Time-To-First-Token (TTFT)** | **38.4 ms** | **52.1 ms** |
| **Decode Throughput** | **46.2 tokens/sec** | **31.8 tokens/sec** |

*(For full benchmark methodology and evaluation scripts, see [`@docs/step-9-benchmarks-and-evaluation.md`](./@docs/step-9-benchmarks-and-evaluation.md)).*

---

## Architectural Highlights

- **From Scratch Base Tool Model**: Not an adapter or distillation. Every weight was initialized randomly and trained end-to-end on tool and machine schemas.
- **Grouped Query Attention (4:1 GQA)**: Shares key/value heads across 4 query heads, reducing KV-cache allocation by 75% for accelerated local decoding.
- **SwiGLU Gated Feed-Forward Networks**: Gated bilinear activation function with 256-byte tensor core alignment delivering superior representation capacity.
- **Rotary Position Embeddings (RoPE)**: Relative position encoding providing clean structured attention over token sequences.
- **Atomic Tool Calling Grammar**: Specialized tokens (`<|tool_call_start|>`, `<|tool_call_end|>`, `<|tool_result_start|>`, `<|tool_result_end|>`, `<|mcp_server_decl|>`) prevent syntax hallucination and markdown leakage.
- **Self-Contained Deployment**: Ships as a lightweight Python package on top of PyTorch—zero external agent frameworks or proxy daemons required.

---

## Installation

### From PyPI (Recommended)
```bash
pip install vaayu
```

### From Source
```bash
git clone https://github.com/Meetmendapara09/Vaayu-SLMM.git
cd Vaayu-SLMM
pip install -e .
```

---

## Quickstart

### 1. Load Pretrained Weights Directly from Hugging Face Hub

```python
from vaayu import Vaayu

# Stream architecture, tokenizer, and weights from Hugging Face Hub
ai = Vaayu.from_pretrained("meetmendapara/Vaayu-Base")

# In-process inference
response = ai.chat("Explain the purpose of Model Context Protocol (MCP) in one concise sentence.")
print(response)
```

### 2. Connect to Local Model Context Protocol (MCP) Servers

Vaayu connects directly to any MCP server over standard input/output (`stdio`) and Server-Sent Events (`SSE`):

```python
from vaayu import Vaayu

ai = Vaayu.from_pretrained("meetmendapara/Vaayu-Base")

# Connect to any local MCP server (e.g. filesystem, sqlite, terminal, git)
ai.attach_mcp_server(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "./workspace"]
)

# Run autonomous agentic step
result = ai.agent_step("Find all Python configuration files and list their paths.")
print(result)
```

### 3. Registering In-Process Host Tools

```python
from vaayu import Vaayu

ai = Vaayu.from_pretrained("meetmendapara/Vaayu-Base")

def get_system_metrics():
    import psutil
    return {"cpu": psutil.cpu_percent(), "memory": psutil.virtual_memory().percent}

ai.register_function(
    name="get_system_metrics",
    description="Retrieve live CPU and RAM usage percentages of the host system.",
    input_schema={"type": "object", "properties": {}},
    func=get_system_metrics
)

response = ai.chat("What is the current CPU utilization?")
print(response)
```

### 4. Loading Local Checkpoints

```python
from vaayu import Vaayu

ai = Vaayu.load_local("checkpoints/vaayu_base/vaayu_final.pt", variant="base")
print(ai.chat("Hello, Vaayu!"))
```

---

## Autonomous Error Self-Correction (Token Stream Example)

When an MCP tool execution fails or returns an error payload, Vaayu ingests the exception, reflects in `<|thought_start|>`, and generates a corrected call:

```
<|im_start|>user
Inspect the production configuration file at 'config/settings.yaml'.<|im_end|>
<|im_start|>assistant
<|thought_start|>I need to read the configuration file specified by the user.<|thought_end|>
<|tool_call_start|>{"name": "read_file", "arguments": {"path": "config/settings.yaml"}}<|tool_call_end|>
<|im_end|>
<|im_start|>tool
<|tool_result_start|>{"status": "error", "error_code": "ENOENT", "message": "File not found: config/settings.yaml. Did you mean 'config/settings.json'?"}<|tool_result_end|>
<|im_end|>
<|im_start|>assistant
<|thought_start|>The file 'config/settings.yaml' was not found (ENOENT). The host environment suggests 'config/settings.json' exists. I will retry with the corrected path.<|thought_end|>
<|tool_call_start|>{"name": "read_file", "arguments": {"path": "config/settings.json"}}<|tool_call_end|>
<|im_end|>
```

---

## Command Line Interface (CLI)

The `vaayu` package provides an interactive terminal CLI:

```bash
# Interactive chat with official Hugging Face release
vaayu chat --repo meetmendapara/Vaayu-Base

# Chat with local checkpoint
vaayu chat --weights checkpoints/vaayu_base/vaayu_final.pt

# Inspect model configurations
vaayu info --repo meetmendapara/Vaayu-Base
```

---

## Special Tokens & Protocol Registry

| Control Token | Purpose |
| :--- | :--- |
| <code>&lt;&#124;im_start&#124;&gt;</code> | Initiates dialogue turn (<code>system</code>, <code>user</code>, <code>assistant</code>, <code>tool</code>) |
| <code>&lt;&#124;im_end&#124;&gt;</code> | Closes dialogue turn |
| <code>&lt;&#124;thought_start&#124;&gt;</code> | Initiates internal latent reasoning block |
| <code>&lt;&#124;thought_end&#124;&gt;</code> | Closes internal latent reasoning block |
| <code>&lt;&#124;tool_call_start&#124;&gt;</code> | Delimits beginning of structured JSON tool invocation |
| <code>&lt;&#124;tool_call_end&#124;&gt;</code> | Delimits conclusion of tool invocation |
| <code>&lt;&#124;tool_result_start&#124;&gt;</code> | Delimits input of tool execution return payload |
| <code>&lt;&#124;tool_result_end&#124;&gt;</code> | Delimits conclusion of tool return payload |
| <code>&lt;&#124;mcp_server_decl&#124;&gt;</code> | Injects connected MCP server capabilities & tools schema |
| <code>&lt;&#124;mcp_server_end&#124;&gt;</code> | Closes MCP declaration |

---

## Limitations & Failure Modes

1. **Context Window Ceiling**: Vaayu-Base operates with a 2,048-token context window. Connecting multiple verbose MCP servers without schema minification can exhaust the context budget. For multi-server deployments, use **Vaayu-Large (4,096 tokens)** or schema pruning.
2. **Specialized Domain Focus**: Vaayu is trained specifically on tool contracts, JSON-RPC, code actions, and structured reasoning. It is not designed for open-domain creative writing, trivia, or long-form essays.
3. **Strict Schema Requirement**: Generating accurate arguments requires tool declarations to have clear parameter types (`str`, `int`, `bool`, `list`). Unspecified schemas (`{"type": "any"}`) may reduce argument precision.
4. **Execution Safety**: Vaayu emits tool calls as structured payloads; the host application runtime must enforce permission sandboxing before dispatching sensitive actions (filesystem write, shell execution, database mutations).

---

## Repository Structure

```
.
├── @docs/                          # Comprehensive step-by-step engineering documentation
│   ├── param_count.md              # Exact analytical parameter derivations
│   ├── step-0-architecture-and-specs.md
│   ├── step-1-dataset-curation.md
│   ├── step-2-tokenizer-and-special-tokens.md
│   ├── step-3-training-from-scratch.md
│   ├── step-4-mcp-connector-and-tool-calling.md
│   ├── step-5-software-embedding-guide.md
│   ├── step-6-deployment-and-huggingface.md
│   ├── step-7-vaayu-large-variant-and-advanced-capabilities.md
│   ├── step-8-python-package-and-pypi.md
│   └── step-9-benchmarks-and-evaluation.md
├── eval/                           # Empirical evaluation suite & benchmarks
│   └── mcp_benchmark.py            # 200-case MCP benchmark harness
├── vaayu/                          # Production Python package
│   ├── core.py                     # Local inference engine with KV-caching
│   ├── embed.py                    # High-level embedding API (from_pretrained, load_local)
│   ├── mcp_client.py               # Native Model Context Protocol client bridge
│   ├── cli.py                      # Interactive Command Line Interface
│   ├── model/                      # PyTorch Transformer architecture & configs
│   └── tokenizer/                  # Custom BPE tokenizer with MCP control tokens
├── src/                            # Training & data engineering codebase
├── scripts/                        # Publishing automation (PyPI, Hugging Face, Kaggle)
└── examples/                       # Embedded runtime demonstrations
```

---

## Step-by-Step Documentation

Complete engineering specifications are available in the [`@docs/`](./@docs) directory:

1. [**Parameter Derivation & Verification**](./@docs/param_count.md)
2. [**Step 0: Architecture & Technical Specifications**](./@docs/step-0-architecture-and-specs.md)
3. [**Step 1: Dataset Curation & Tool Ingestion**](./@docs/step-1-dataset-curation.md)
4. [**Step 2: Tokenizer Training & Special Control Tokens**](./@docs/step-2-tokenizer-and-special-tokens.md)
5. [**Step 3: Training from Scratch & AMP Pipeline**](./@docs/step-3-training-from-scratch.md)
6. [**Step 4: Model Context Protocol (MCP) Connector**](./@docs/step-4-mcp-connector-and-tool-calling.md)
7. [**Step 5: Software Embedding & In-Process Integration**](./@docs/step-5-software-embedding-guide.md)
8. [**Step 6: Model Deployment & Hugging Face Hub**](./@docs/step-6-deployment-and-huggingface.md)
9. [**Step 7: Vaayu-Large Variant & Advanced Capabilities**](./@docs/step-7-vaayu-large-variant-and-advanced-capabilities.md)
10. [**Step 8: Standalone Python Package & PyPI Distribution**](./@docs/step-8-python-package-and-pypi.md)
11. [**Step 9: Benchmarks, Evaluation & Empirical Results**](./@docs/step-9-benchmarks-and-evaluation.md)

---

## Citation & BibTeX

```bibtex
@software{vaayu2026slm,
  author = {Meet Mendapara},
  title = {Vaayu: Small Language Model (SLM) for Machine-to-Machine Tool Calling with Native Model Context Protocol (MCP)},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Meetmendapara09/Vaayu-SLMM}}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE). Free for academic research, open-source development, and commercial applications.
