"""
Training engine for Vaayu SLMM (245M Parameters).
Trains from scratch with AdamW, Cosine LR scheduling, AMP, and strict 12-hour budget guardrails.
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast

from src.model.config import VaayuConfig
from src.model.architecture import VaayuForCausalLM
from src.training.dataset import VaayuTextDataset

def parse_args():
    parser = argparse.ArgumentParser(description="Train Vaayu SLMM from scratch")
    parser.add_argument("--train_file", type=str, default="data/processed/train.jsonl")
    parser.add_argument("--val_file", type=str, default="data/processed/val.jsonl")
    parser.add_argument("--tokenizer_dir", type=str, default="src/tokenizer/vaayu_tokenizer")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--max_hours", type=float, default=11.5, help="Strict training ceiling in hours")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=250)
    return parser.parse_args()

def get_lr(step: int, warmup_steps: int, total_steps: int, base_lr: float, min_lr: float = 1e-5) -> float:
    """Cosine learning rate schedule with linear warmup."""
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    if step > total_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (base_lr - min_lr)

def evaluate(model: VaayuForCausalLM, val_loader: DataLoader, device: str) -> float:
    model.eval()
    total_loss = 0.0
    steps = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs["loss"]
            if loss is not None and not torch.isnan(loss):
                total_loss += loss.item()
                steps += 1
            if steps >= 50:  # Evaluate on up to 50 batches for fast feedback
                break
    model.train()
    return total_loss / max(1, steps)

def train():
    args = parse_args()
    print("==================================================")
    print(" Vaayu SLMM: Scratch Training Engine")
    print("==================================================")
    print(f"Target Device: {args.device.upper()}")
    print(f"Max Sequence Length: {args.max_seq_len}")
    print(f"Max Training Hours: {args.max_hours} hours (< 12-hour limit)")
    
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Tokenizer
    print(f"Loading tokenizer from {args.tokenizer_dir}...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_dir)
    vocab_size = len(tokenizer)
    print(f"[OK] Tokenizer loaded with vocab size: {vocab_size}")

    # 2. Initialize Model from Scratch
    config = VaayuConfig(vocab_size=vocab_size, max_position_embeddings=args.max_seq_len)
    model = VaayuForCausalLM(config)
    model.to(args.device)
    
    param_count = model.get_num_params()
    print(f"[OK] Initialized Vaayu Base Model: {param_count:,} parameters ({param_count/1e6:.2f}M)")

    # 3. Prepare Datasets & Loaders
    print("Loading datasets...")
    train_dataset = VaayuTextDataset(args.train_file, tokenizer, max_length=args.max_seq_len)
    val_dataset = VaayuTextDataset(args.val_file, tokenizer, max_length=args.max_seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    total_training_steps = (len(train_loader) // args.gradient_accumulation_steps) * args.epochs
    warmup_steps = int(total_training_steps * 0.05)
    print(f"Total Steps: {total_training_steps} | Warmup Steps: {warmup_steps}")

    # 4. Optimizer & Mixed Precision Scaler
    decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": args.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=args.lr, betas=(0.9, 0.95), eps=1e-8)
    use_amp = args.device == "cuda"
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # 5. Training Loop with 12-Hour Budget Watchdog
    start_time = time.time()
    max_seconds = args.max_hours * 3600
    global_step = 0
    best_val_loss = float("inf")
    model.train()

    print("\nBeginning training from scratch...")
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        optimizer.zero_grad()
        
        for step, batch in enumerate(train_loader):
            elapsed = time.time() - start_time
            if elapsed >= max_seconds:
                print(f"\n[ALERT] Reached training time ceiling ({args.max_hours} hours). Gracefully terminating training.")
                break

            current_lr = get_lr(global_step, warmup_steps, total_training_steps, args.lr)
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

            input_ids = batch["input_ids"].to(args.device)
            labels = batch["labels"].to(args.device)

            with torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.float16):
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs["loss"] / args.gradient_accumulation_steps

            scaler.scale(loss).backward()
            epoch_loss += loss.item() * args.gradient_accumulation_steps

            if (step + 1) % args.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 20 == 0:
                    step_loss = loss.item() * args.gradient_accumulation_steps
                    print(f"Epoch {epoch}/{args.epochs} | Step {global_step}/{total_training_steps} | Loss: {step_loss:.4f} | LR: {current_lr:.2e} | Elapsed: {elapsed/60:.1f}m")

                if global_step % args.eval_steps == 0:
                    val_loss = evaluate(model, val_loader, args.device)
                    print(f"\n--- Evaluation at Step {global_step} --- Val Loss: {val_loss:.4f} ---")
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_path = checkpoint_dir / "vaayu_best.pt"
                        torch.save({
                            "epoch": epoch,
                            "step": global_step,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "config": config.to_dict(),
                            "val_loss": val_loss,
                        }, best_path)
                        print(f"[OK] New best checkpoint saved to {best_path}\n")

                if global_step % args.save_steps == 0:
                    ckpt_path = checkpoint_dir / f"vaayu_step_{global_step}.pt"
                    torch.save(model.state_dict(), ckpt_path)

        if time.time() - start_time >= max_seconds:
            break

    # Save final model & weights
    final_weights_path = checkpoint_dir / "vaayu_final.pt"
    torch.save(model.state_dict(), final_weights_path)
    config.save_pretrained(str(checkpoint_dir))
    tokenizer.save_pretrained(str(checkpoint_dir))
    print(f"\n[OK] Training completed successfully. Final weights saved to {final_weights_path}")

if __name__ == "__main__":
    train()
