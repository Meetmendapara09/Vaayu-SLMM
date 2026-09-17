"""
Model configuration for Vaayu SLMM family:
- Vaayu-Base: ~245M Parameters (16 layers, 1024 hidden, 2816 intermediate, 2k context)
- Vaayu-Large: ~492M Parameters (23 layers, 1280 hidden, 3584 intermediate, 4k context)
"""

from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Optional

@dataclass
class VaayuConfig:
    vocab_size: int = 32000
    hidden_size: int = 1024
    intermediate_size: int = 2816
    num_hidden_layers: int = 16
    num_attention_heads: int = 16
    num_key_value_heads: int = 4        # 4:1 GQA
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = False   # ~245M params
    initializer_range: float = 0.02
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0
    model_type: str = "vaayu_slmm"
    variant: str = "base"
    architectures: tuple = ("VaayuForCausalLM",)

    def to_dict(self):
        return asdict(self)

    def save_pretrained(self, save_directory: str):
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        with open(save_path / "config.json", "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_pretrained(cls, save_directory: str):
        config_path = Path(save_directory) / "config.json"
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

@dataclass
class VaayuLargeConfig(VaayuConfig):
    """Vaayu-Large: 492M Parameter SLMM for advanced multi-tool calling & interface reasoning."""
    hidden_size: int = 1280
    intermediate_size: int = 3584
    num_hidden_layers: int = 23
    num_attention_heads: int = 20
    num_key_value_heads: int = 5        # 4:1 GQA (20 / 5 = 4)
    max_position_embeddings: int = 4096 # Extended 4k context window
    variant: str = "large"