"""
Kaggle Online GPU Training Script for Vaayu SLMM (245M Parameters).
Executed directly inside Kaggle GPU environment via Kaggle CLI.
"""

import math
import os
import sys
import time
import json
from pathlib import Path
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. ARCHITECTURE & CONFIGURATION
# ==========================================

@dataclass
class VaayuConfig:
    vocab_size: int = 32000
    hidden_size: int = 1024
    intermediate_size: int = 2816
    num_hidden_layers: int = 16
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    max_position_embeddings: int = 1024
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = False
    initializer_range: float = 0.02
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0
    model_type: str = "vaayu_slmm"

    def to_dict(self):
        return asdict(self)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        t = torch.arange(max_position_embeddings, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int):
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype, device=x.device),
            self.sin_cached[:seq_len].to(dtype=x.dtype, device=x.device)
        )

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

class VaayuAttention(nn.Module):
    def __init__(self, config: VaayuConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor, rotary_emb: tuple):
        b, s, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = rotary_emb
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(b, s, self.hidden_size)
        return self.o_proj(attn_out)

class VaayuMLP(nn.Module):
    def __init__(self, config: VaayuConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class VaayuBlock(nn.Module):
    def __init__(self, config: VaayuConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = VaayuAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = VaayuMLP(config)

    def forward(self, hidden_states: torch.Tensor, rotary_emb: tuple):
        hidden_states = hidden_states + self.self_attn(self.input_layernorm(hidden_states), rotary_emb)
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states

class VaayuForCausalLM(nn.Module):
    def __init__(self, config: VaayuConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rotary_emb = RotaryEmbedding(
            dim=config.hidden_size // config.num_attention_heads,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta
        )
        self.layers = nn.ModuleList([VaayuBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None):
        b, s = input_ids.shape
        h = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(h, seq_len=s)
        rotary_tuple = (cos, sin)

        for layer in self.layers:
            h = layer(h, rotary_tuple)

        h = self.norm(h)
        logits = self.lm_head(h)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1), ignore_index=-100)

        return {"loss": loss, "logits": logits}

# ==========================================
# 2. DATASET & TOKENIZER SETUP
# ==========================================

class SyntheticMCPDataset(Dataset):
    def __init__(self, num_samples=10000, seq_len=512, vocab_size=32000):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Deterministic token sequences with pattern structure
        torch.manual_seed(idx)
        ids = torch.randint(4, self.vocab_size, (self.seq_len,), dtype=torch.long)
        # Injects tool-call patterns
        ids[0] = 4  # <|im_start|>
        ids[50] = 5 # <|im_end|>
        ids[51] = 4
        ids[80] = 8 # <|tool_call_start|>
        ids[150] = 9 # <|tool_call_end|>
        ids[151] = 5
        labels = ids.clone()
        return {"input_ids": ids, "labels": labels}

# ==========================================
# 3. KAGGLE TRAINING LOOP
# ==========================================

def main():
    print("==================================================")
    print(" Vaayu SLMM: Training from Scratch on Kaggle GPU")
    print("==================================================")
    if not torch.cuda.is_available():
        raise RuntimeError("CRITICAL ERROR: CUDA GPU was not allocated by Kaggle! This model must be trained on GPU. Terminating execution to prevent CPU training.")
    device = "cuda"
    print(f"[GPU CONFIRMED] Device: {device}")
    print(f"[GPU CONFIRMED] GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"[GPU CONFIRMED] Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print(f"[GPU CONFIRMED] CUDA Capability: {torch.cuda.get_device_capability(0)}")

    output_dir = Path("/kaggle/working/vaayu_checkpoint") if Path("/kaggle/working").exists() else Path("./checkpoints")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = VaayuConfig()
    model = VaayuForCausalLM(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Model Parameters: {total_params:,} ({total_params/1e6:.2f}M)")

    dataset = SyntheticMCPDataset(num_samples=12000, seq_len=512, vocab_size=config.vocab_size)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))

    start_time = time.time()
    max_training_seconds = 11.5 * 3600 # 11.5 hour safety guardrail
    epochs = 4

    print("Starting training passes...")
    global_step = 0
    model.train()

    for epoch in range(1, epochs + 1):
        for step, batch in enumerate(loader):
            if (time.time() - start_time) >= max_training_seconds:
                print("11.5 hour time limit reached. Gracefully concluding training.")
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(device == "cuda"), dtype=torch.float16):
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs["loss"]

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            if global_step % 50 == 0:
                print(f"Epoch {epoch}/{epochs} | Step {global_step} | Loss: {loss.item():.4f} | Time: {(time.time() - start_time)/60:.2f}m")

    # Save final model state
    final_path = output_dir / "vaayu_final.pt"
    torch.save(model.state_dict(), final_path)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=2)

    print(f"\n[OK] Model successfully trained from scratch and saved to {final_path}")

if __name__ == "__main__":
    main()
