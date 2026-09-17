"""
Core local inference engine for Vaayu SLMM & Vaayu-Large.
Optimized for low-latency local execution, KV caching, parallel tool calls, and error recovery.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Generator, Tuple

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast

from .model.config import VaayuConfig, VaayuLargeConfig
from .model.architecture import VaayuForCausalLM

@dataclass
class ToolCall:
    name: str
    arguments: Dict[str, Any]
    raw: str

class VaayuInferenceEngine:
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        tokenizer_dir: Optional[str] = None,
        device: Optional[str] = None,
        variant: str = "base",
        dtype: torch.dtype = torch.float32
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.variant = variant

        # Fallback to internal package tokenizer if none provided
        pkg_tok_dir = Path(__file__).resolve().parent / "tokenizer"
        root_tok_dir = Path(__file__).resolve().parent.parent / "src" / "tokenizer" / "vaayu_tokenizer"
        
        if tokenizer_dir and Path(tokenizer_dir).exists():
            tok_path = Path(tokenizer_dir)
        elif pkg_tok_dir.exists() and (pkg_tok_dir / "tokenizer.json").exists():
            tok_path = pkg_tok_dir
        else:
            tok_path = root_tok_dir
        
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tok_path))
        
        # Load Config according to variant
        if variant == "large":
            self.config = VaayuLargeConfig(vocab_size=len(self.tokenizer))
        else:
            self.config = VaayuConfig(vocab_size=len(self.tokenizer))

        self.model = VaayuForCausalLM(self.config)
        
        if checkpoint_path and Path(checkpoint_path).exists():
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            self.model.load_state_dict(state_dict, strict=False)

        self.model.to(self.device).to(self.dtype)
        self.model.eval()

        self.im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.tool_call_end_id = self.tokenizer.convert_tokens_to_ids("<|tool_call_end|>")

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.6,
        top_k: int = 40,
        stop_at_tool_call: bool = False
    ) -> str:
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        generated = input_ids.clone()
        past_key_values = None

        for _ in range(max_new_tokens):
            if past_key_values is None:
                outputs = self.model(generated, use_cache=True)
            else:
                outputs = self.model(generated[:, -1:], past_key_values=past_key_values, use_cache=True)

            past_key_values = outputs["past_key_values"]
            logits = outputs["logits"][:, -1, :]

            if temperature > 0:
                logits = logits / temperature
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=-1)
            token_id = next_token.item()

            if token_id == self.im_end_id:
                break
            if stop_at_tool_call and token_id == self.tool_call_end_id:
                break

        full_output = self.tokenizer.decode(generated[0].tolist(), skip_special_tokens=False)
        return full_output[len(prompt):]

    @torch.no_grad()
    def stream_generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.6,
        top_k: int = 40,
        stop_at_tool_call: bool = False
    ) -> Generator[str, None, None]:
        """Streams generated tokens one by one as they are decoded."""
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        generated = input_ids.clone()
        past_key_values = None

        for _ in range(max_new_tokens):
            if past_key_values is None:
                outputs = self.model(generated, use_cache=True)
            else:
                outputs = self.model(generated[:, -1:], past_key_values=past_key_values, use_cache=True)

            past_key_values = outputs["past_key_values"]
            logits = outputs["logits"][:, -1, :]

            if temperature > 0:
                logits = logits / temperature
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            token_id = next_token.item()
            generated = torch.cat([generated, next_token], dim=-1)

            if token_id == self.im_end_id:
                break

            token_text = self.tokenizer.decode([token_id], skip_special_tokens=False)
            yield token_text

            if stop_at_tool_call and token_id == self.tool_call_end_id:
                break

    def extract_tool_calls(self, text: str) -> List[ToolCall]:
        """Extracts all JSON tool calls enclosed within <|tool_call_start|> and <|tool_call_end|>."""
        pattern = r"<\|tool_call_start\|>(.*?)<\|tool_call_end\|>"
        matches = re.findall(pattern, text, re.DOTALL)
        calls = []
        for raw in matches:
            raw_clean = raw.strip()
            try:
                data = json.loads(raw_clean)
                calls.append(ToolCall(
                    name=data.get("name", ""),
                    arguments=data.get("arguments", {}),
                    raw=raw_clean
                ))
            except json.JSONDecodeError:
                name_match = re.search(r'"name":\s*"([^"]+)"', raw_clean)
                if name_match:
                    calls.append(ToolCall(
                        name=name_match.group(1),
                        arguments={},
                        raw=raw_clean
                    ))
        return calls