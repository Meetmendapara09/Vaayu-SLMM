# Step 2: Custom Tokenizer & Special Tokens

## 1. Overview

A Small Language Machine Model (SLMM) requires deterministic token boundaries for machine protocol negotiation, tool dispatch, and reasoning delimiters. When standard tokenizers encounter JSON-RPC or tool schemas, they often split structural tags into arbitrary multi-token fragments, degrading parsing accuracy.

Vaayu's custom Byte-Pair Encoding (BPE) tokenizer resolves this by pre-allocating dedicated atomic special tokens for MCP and tool calling.

---

## 2. Special Tokens Registry

| Token | ID | Functional Purpose |
| :--- | :--- | :--- |
| <code>&lt;&#124;pad&#124;&gt;</code> | 0 | Padding token for batch alignment |
| <code>&lt;&#124;bos&#124;&gt;</code> | 1 | Beginning-of-sequence delimiter |
| <code>&lt;&#124;eos&#124;&gt;</code> | 2 | End-of-sequence delimiter |
| <code>&lt;&#124;unk&#124;&gt;</code> | 3 | Unknown character fallback |
| <code>&lt;&#124;im_start&#124;&gt;</code> | 4 | Starts a chat role header (<code>system</code>, <code>user</code>, <code>assistant</code>, <code>tool</code>) |
| <code>&lt;&#124;im_end&#124;&gt;</code> | 5 | Closes a message block |
| <code>&lt;&#124;thought_start&#124;&gt;</code> | 6 | Initiates latent reasoning chain before taking an action |
| <code>&lt;&#124;thought_end&#124;&gt;</code> | 7 | Terminates latent reasoning |
| <code>&lt;&#124;tool_call_start&#124;&gt;</code> | 8 | Marks start of JSON tool call: <code>{"name": "...", "arguments": {...}}</code> |
| <code>&lt;&#124;tool_call_end&#124;&gt;</code> | 9 | Closes JSON tool call payload |
| <code>&lt;&#124;tool_result_start&#124;&gt;</code> | 10 | Marks beginning of tool execution output |
| <code>&lt;&#124;tool_result_end&#124;&gt;</code> | 11 | Marks end of tool execution output |
| <code>&lt;&#124;mcp_server_decl&#124;&gt;</code> | 12 | Injects connected MCP server capabilities & tools list |
| <code>&lt;&#124;mcp_server_end&#124;&gt;</code> | 13 | Closes MCP server capability declaration |

---

## 3. Training the Tokenizer

The script [`src/tokenizer/train_tokenizer.py`](../src/tokenizer/train_tokenizer.py) leverages Hugging Face `tokenizers` with ByteLevel pre-tokenization:

```bash
python src/tokenizer/train_tokenizer.py
```

### Verification
Round-trip encoding and decoding ensures that all special tokens are preserved as single unit IDs without mutation or whitespace leakage.
