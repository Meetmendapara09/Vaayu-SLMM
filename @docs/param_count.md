# Analytical Parameter Count Derivation: Vaayu-Base & Vaayu-Large

This document provides the exact analytical formula and verification code for parameter counts across the **Vaayu** family.

---

## 1. Generalized Parameter Formula

For a decoder-only Transformer with Rotary Position Embeddings (RoPE), Grouped Query Attention (GQA), SwiGLU Feed-Forward Networks, and Pre-RMSNorm:

$$\text{Params}_{\text{embedding}} = V \times d$$

$$\text{Params}_{\text{attn}} = \underbrace{d \times (n_q \times d_h)}_{W_q} + \underbrace{d \times (n_{kv} \times d_h)}_{W_k} + \underbrace{d \times (n_{kv} \times d_h)}_{W_v} + \underbrace{(n_q \times d_h) \times d}_{W_o} = 2 d^2 + 2 d (n_{kv} d_h)$$

$$\text{Params}_{\text{swiglu}} = \underbrace{d \times d_{ff}}_{W_{gate}} + \underbrace{d \times d_{ff}}_{W_{up}} + \underbrace{d_{ff} \times d}_{W_{down}} = 3 d \cdot d_{ff}$$

$$\text{Params}_{\text{norms}} = \underbrace{d}_{\text{attn\_norm}} + \underbrace{d}_{\text{ffn\_norm}} = 2d$$

$$\text{Params}_{\text{layer}} = \text{Params}_{\text{attn}} + \text{Params}_{\text{swiglu}} + \text{Params}_{\text{norms}}$$

$$\text{Params}_{\text{lm\_head}} = \begin{cases} 0 & \text{if tied} \\ d \times V & \text{if untied} \end{cases}$$

$$\text{Params}_{\text{final\_norm}} = d$$

$$\mathbf{\text{Total Parameters}} = \text{Params}_{\text{embedding}} + (L \times \text{Params}_{\text{layer}}) + \text{Params}_{\text{final\_norm}} + \text{Params}_{\text{lm\_head}}$$

Where:
- $V$ = Vocabulary size (`vocab_size`)
- $d$ = Hidden dimension (`hidden_size`)
- $d_{ff}$ = Intermediate dimension (`intermediate_size`)
- $L$ = Number of layers (`num_hidden_layers`)
- $n_q$ = Number of query attention heads (`num_attention_heads`)
- $n_{kv}$ = Number of key-value attention heads (`num_key_value_heads`)
- $d_h = d / n_q$ = Head dimension (`head_dim`)

---

## 2. Vaayu-Base (245M) Exact Breakdown

- $V = 32,000$
- $d = 1,024$
- $d_{ff} = 2,816$
- $L = 16$
- $n_q = 16, n_{kv} = 4, d_h = 64$
- `tie_word_embeddings = False`

| Component | Calculation | Exact Parameters |
| :--- | :--- | :--- |
| **Token Embedding** | $32,000 \times 1,024$ | **32,768,000** |
| **Attention $W_q$** | $1,024 \times (16 \times 64)$ | 1,048,576 |
| **Attention $W_k$** | $1,024 \times (4 \times 64)$ | 262,144 |
| **Attention $W_v$** | $1,024 \times (4 \times 64)$ | 262,144 |
| **Attention $W_o$** | $(16 \times 64) \times 1,024$ | 1,048,576 |
| **SwiGLU $W_{gate}$** | $1,024 \times 2,816$ | 2,883,584 |
| **SwiGLU $W_{up}$** | $1,024 \times 2,816$ | 2,883,584 |
| **SwiGLU $W_{down}$** | $2,816 \times 1,024$ | 2,883,584 |
| **Layer RMSNorms** | $2 \times 1,024$ | 2,048 |
| **Single Layer Total** | $2,621,440 + 8,650,752 + 2,048$ | **11,274,240** |
| **16 Transformer Layers** | $16 \times 11,274,240$ | **180,387,840** |
| **Final RMSNorm** | $1 \times 1,024$ | **1,024** |
| **LM Head (Untied)** | $1,024 \times 32,000$ | **32,768,000** |
| **GRAND TOTAL** | $32,768,000 + 180,387,840 + 1,024 + 32,768,000$ | **245,924,864** |

---

## 3. Vaayu-Large (492M) Exact Breakdown

- $V = 32,000$
- $d = 1,280$
- $d_{ff} = 3,584$
- $L = 23$
- $n_q = 20, n_{kv} = 5, d_h = 64$
- `tie_word_embeddings = False`

| Component | Calculation | Exact Parameters |
| :--- | :--- | :--- |
| **Token Embedding** | $32,000 \times 1,280$ | **40,960,000** |
| **Attention $W_q$** | $1,280 \times (20 \times 64)$ | 1,638,400 |
| **Attention $W_k$** | $1,280 \times (5 \times 64)$ | 409,600 |
| **Attention $W_v$** | $1,280 \times (5 \times 64)$ | 409,600 |
| **Attention $W_o$** | $(20 \times 64) \times 1,280$ | 1,638,400 |
| **SwiGLU $W_{gate}$** | $1,280 \times 3,584$ | 4,587,520 |
| **SwiGLU $W_{up}$** | $1,280 \times 3,584$ | 4,587,520 |
| **SwiGLU $W_{down}$** | $3,584 \times 1,280$ | 4,587,520 |
| **Layer RMSNorms** | $2 \times 1,280$ | 2,560 |
| **Single Layer Total** | $4,096,000 + 13,762,560 + 2,560$ | **17,861,120** |
| **23 Transformer Layers** | $23 \times 17,861,120$ | **410,805,760** |
| **Final RMSNorm** | $1 \times 1,280$ | **1,280** |
| **LM Head (Untied)** | $1,280 \times 32,000$ | **40,960,000** |
| **GRAND TOTAL** | $40,960,000 + 410,805,760 + 1,280 + 40,960,000$ | **492,727,040** |

---

## 4. Verification Script

You can verify these counts in Python with PyTorch directly:

```python
from src.model.config import VaayuConfig, VaayuLargeConfig
from src.model.architecture import VaayuForCausalLM

base_model = VaayuForCausalLM(VaayuConfig())
base_params = sum(p.numel() for p in base_model.parameters())
assert base_params == 245_924_864, f"Base mismatch: {base_params}"

large_model = VaayuForCausalLM(VaayuLargeConfig())
large_params = sum(p.numel() for p in large_model.parameters())
assert large_params == 492_727_040, f"Large mismatch: {large_params}"

print(f"Verified: Base = {base_params:,} | Large = {large_params:,}")
```
