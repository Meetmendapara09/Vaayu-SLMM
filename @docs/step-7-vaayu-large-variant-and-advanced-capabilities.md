# Step 7: Vaayu-Large (492M) Variant & Advanced Capabilities

## 1. Overview

**Vaayu-Large** is the flagship 492-Million parameter Small Language Machine Model (SLMM) variant designed to push the boundaries of embedded local machine models while strictly adhering to the 200M–500M parameter limit.

---

## 2. Technical Comparison

| Specification | Vaayu-Base | Vaayu-Large |
| :--- | :--- | :--- |
| **Parameters** | 245,924,864 (~245M) | **492,727,040 (~492M)** |
| **Layers** | 16 | **23** |
| **Hidden Size ($d_{model}$)** | 1024 | **1280** |
| **Intermediate Size ($d_{ff}$)** | 2816 | **3584** (SwiGLU) |
| **Attention Heads** | 16 Q-heads, 4 KV-heads | **20 Q-heads, 5 KV-heads** (4:1 GQA) |
| **Context Length** | 2,048 tokens | **4,096 tokens** |
| **VRAM Footprint (FP16)** | ~490 MB | **~985 MB** (Runs on 4GB-8GB laptops easily) |
| **VRAM Footprint (INT8)** | ~245 MB | **~492 MB** |
| **Kaggle Kernel** | `vaayu-slmm-training` (Complete) | `vaayu-large-slmm-training` (Complete) |
| **Hugging Face Hub** | [`meetmendapara/Vaayu-Base`](https://huggingface.co/meetmendapara/Vaayu-Base) | [**`meetmendapara/Vaayu-Large`**](https://huggingface.co/meetmendapara/Vaayu-Large) |

---

## 3. Advanced Capabilities

### 3.1 Parallel Multi-Tool Calling
In a single inference turn, Vaayu-Large can emit multiple distinct `<|tool_call_start|>` blocks, allowing the host application or MCP client to dispatch requests concurrently:

```
<|im_start|>assistant
<|thought_start|>I need to inspect both git status and directory contents simultaneously.<|thought_end|>
<|tool_call_start|>{"name": "git_branch", "arguments": {}}<|tool_call_end|>
<|tool_call_start|>{"name": "list_files", "arguments": {"directory": "."}}<|tool_call_end|>
<|im_end|>
```

### 3.2 Automated Error Self-Correction Loop
When a tool execution encounters an error or returns a non-zero exit code:
1. The error message is delivered back to Vaayu inside `<|tool_result_start|>`.
2. Vaayu-Large analyzes the error trace inside `<|thought_start|>`.
3. Vaayu-Large generates a corrected tool call (e.g. fixing file paths, modifying SQL queries, changing command flags) without user intervention.

### 3.3 Dynamic Schema & Interface Inference
Vaayu-Large can process complex nested JSON schemas and unknown REST/MCP API declarations zero-shot, extracting required properties, data types, and default values.

---

## 4. Kaggle Online GPU Training

The training kernel for Vaayu-Large is located in `kaggle_run_large/` and is deployed via Kaggle CLI:

```bash
# Push training job to Kaggle
kaggle kernels push -p kaggle_run_large

# Monitor progress
kaggle kernels status mendaparameet/vaayu-large-slmm-training

# Download trained 492M weights
kaggle kernels output mendaparameet/vaayu-large-slmm-training -p checkpoints/
```

---

## 5. Embedding Vaayu-Large Locally

```python
from vaayu import Vaayu

# Initialize Vaayu-Large
ai = Vaayu.load_local(variant="large")

# Attach local MCP server
ai.attach_mcp_server("npx", ["-y", "@modelcontextprotocol/server-filesystem", "./"])

# Chat with autonomous parallel tool execution
response = ai.chat("Read config.yaml and list all files in src/ concurrently.")
print(response)
```