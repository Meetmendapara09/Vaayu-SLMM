"""
Custom Byte-Pair Encoding (BPE) Tokenizer training for Vaayu SLMM.
Includes dedicated structural tokens for MCP protocol negotiation and tool execution.
"""

import json
import os
import sys
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.processors import ByteLevel as ByteLevelProcessor
from transformers import PreTrainedTokenizerFast

PROCESSED_DATA = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "train.jsonl"
TOKENIZER_DIR = Path(__file__).resolve().parent.parent.parent / "src" / "tokenizer" / "vaayu_tokenizer"

SPECIAL_TOKENS = [
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|unk|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|thought_start|>",
    "<|thought_end|>",
    "<|tool_call_start|>",
    "<|tool_call_end|>",
    "<|tool_result_start|>",
    "<|tool_result_end|>",
    "<|mcp_server_decl|>",
    "<|mcp_server_end|>"
]

def batch_iterator(file_path: Path, batch_size: int = 1000):
    batch = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                batch.append(item.get("text", ""))
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
    if batch:
        yield batch

def train_vaayu_tokenizer(vocab_size: int = 32000):
    print("==================================================")
    print(" Vaayu SLMM: Training Custom BPE Tokenizer")
    print("==================================================")
    
    if not PROCESSED_DATA.exists():
        raise FileNotFoundError(f"Training data not found at {PROCESSED_DATA}. Run prepare_mcp_data.py first.")
        
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize ByteLevel BPE tokenizer
    raw_tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
    raw_tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    raw_tokenizer.decoder = ByteLevelDecoder()
    raw_tokenizer.post_processor = ByteLevelProcessor(trim_offsets=True)
    
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
        initial_alphabet=ByteLevel.alphabet()
    )
    
    print(f"Training tokenizer on {PROCESSED_DATA}...")
    raw_tokenizer.train_from_iterator(batch_iterator(PROCESSED_DATA), trainer=trainer)
    
    # Save raw tokenizer.json
    raw_json_path = TOKENIZER_DIR / "tokenizer.json"
    raw_tokenizer.save(str(raw_json_path))
    print(f"[OK] Saved raw tokenizer to {raw_json_path}")
    
    # Wrap in Hugging Face PreTrainedTokenizerFast for seamless ecosystem compatibility
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw_tokenizer,
        bos_token="<|bos|>",
        eos_token="<|eos|>",
        unk_token="<|unk|>",
        pad_token="<|pad|>",
        additional_special_tokens=SPECIAL_TOKENS[4:]
    )
    
    fast_tokenizer.save_pretrained(str(TOKENIZER_DIR))
    print(f"[OK] Saved full HuggingFace tokenizer package to {TOKENIZER_DIR}")
    
    # Validation test
    test_str = (
        "<|im_start|>user\nCan you inspect config.yaml?<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<|thought_start|>Checking file.<|thought_end|>\n"
        "<|tool_call_start|>{\"name\": \"local_filesystem_read\"}<|tool_call_end|><|im_end|>"
    )
    encoded = fast_tokenizer(test_str)
    decoded = fast_tokenizer.decode(encoded["input_ids"])
    print("\n--- Validation Test ---")
    print(f"Input tokens count: {len(encoded['input_ids'])}")
    print(f"Decoded matches input: {test_str == decoded}")
    assert test_str == decoded, "Tokenizer round-trip failed!"
    print("[OK] Tokenizer successfully verified.")

if __name__ == "__main__":
    train_vaayu_tokenizer()
