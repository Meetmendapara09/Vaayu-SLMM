# Step 1: Dataset Acquisition via Kaggle CLI & MCP Preparation

## 1. Overview

To create a **Small Language Machine Model (SLMM)** with state-of-the-art tool-calling abilities that can understand local software interfaces, the training data must be focused and dense. General web crawls contain too much noise for a 245M model to attain high zero-shot tool reliability.

We leverage a hybrid approach:
1. **Curated Open Datasets via Kaggle CLI**:
   - `thedevastator/ai-chatbot-conversational-data` (Glaive Function Calling v2)
   - `renato2marinho/vinkius-mcp-registry` (Global Model Context Protocol Registry)
   - `beatsprom/agentic-tool-use-and-function-calling-suite-2026`
2. **Synthetic Canonical MCP Interface Data**:
   - Machine-executable JSON-RPC 2.0 schemas
   - Filesystem, Process Execution, SQLite Database, Git Operations, and System Automation tools
   - Reasoning chains wrapped in `<|thought_start|>` ... `<|thought_end|>` to teach internal interface deliberation before emitting tool calls.

---

## 2. Kaggle CLI Pipeline

The automated script [`src/data/download_datasets.py`](../src/data/download_datasets.py) interacts with the Kaggle API:

```bash
python src/data/download_datasets.py
```

Under the hood, it performs:
```python
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()
api.dataset_download_files("thedevastator/ai-chatbot-conversational-data", path="data/raw/glaive", unzip=True)
```

---

## 3. Structural Prompt Format

Vaayu utilizes a dedicated format that separates system instructions, reasoning, tool invocations, tool results, and final responses using special control tokens:

```
<|im_start|>system
You are Vaayu, an autonomous Small Language Machine Model (SLMM). You have access to local software tools via the Model Context Protocol (MCP):
[
  {
    "name": "local_filesystem_read",
    "description": "Read contents of a file on the local file system.",
    "inputSchema": {
      "type": "object",
      "properties": {
        "file_path": {"type": "string", "description": "Absolute or relative file path."}
      },
      "required": ["file_path"]
    }
  }
]
<|im_end|>
<|im_start|>user
Can you inspect what is inside config.yaml?
<|im_end|>
<|im_start|>assistant
<|thought_start|>The user requires reading config.yaml. I will invoke local_filesystem_read.<|thought_end|>
<|tool_call_start|>{"name": "local_filesystem_read", "arguments": {"file_path": "./config.yaml"}}<|tool_call_end|>
<|im_end|>
<|im_start|>tool
<|tool_result_start|>{"status": "ok", "content": "version: 1.0\ndebug: True\n"}<|tool_result_end|>
<|im_end|>
<|im_start|>assistant
I have read `config.yaml`. It contains `version: 1.0` and `debug: True`.
<|im_end|>
```

---

## 4. Dataset Processing

To process and split into training and validation sets:
```bash
python src/data/prepare_mcp_data.py
```

Outputs:
- `data/processed/train.jsonl` (95% split)
- `data/processed/val.jsonl` (5% split)
