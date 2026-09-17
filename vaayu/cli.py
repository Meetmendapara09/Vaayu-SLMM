"""
Command-line interface for Vaayu SLMM.
Usage:
    vaayu info
    vaayu chat "Inspect git repository status"
    vaayu run --model meetmendapara/Vaayu-Base
"""

import argparse
import sys
from vaayu import Vaayu, __version__

def main():
    parser = argparse.ArgumentParser(
        prog="vaayu",
        description="Vaayu SLMM: Small Language Machine Model CLI"
    )
    parser.add_argument("--version", action="version", version=f"vaayu {__version__}")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # info
    subparsers.add_parser("info", help="Display Vaayu SLMM specifications and environment details")
    
    # chat
    chat_parser = subparsers.add_parser("chat", help="Execute an autonomous tool-calling prompt")
    chat_parser.add_argument("prompt", type=str, help="Prompt or task instruction")
    chat_parser.add_argument("--model", type=str, default="meetmendapara/Vaayu-Base", help="Model checkpoint or Hugging Face ID")
    chat_parser.add_argument("--variant", type=str, default="base", choices=["base", "large"])
    
    args = parser.parse_args()
    
    if args.command == "info":
        print("==================================================")
        print(f" Vaayu SLMM Engine (v{__version__})")
        print("==================================================")
        print("Architecture: Custom Transformer Decoder")
        print("Variants:")
        print("  - Vaayu-Base: 245M Parameters (RoPE, 4:1 GQA, SwiGLU, 2k ctx)")
        print("  - Vaayu-Large: 492M Parameters (RoPE, 4:1 GQA, SwiGLU, 4k ctx)")
        print("Capabilities:")
        print("  - In-Process Local Execution")
        print("  - Native Model Context Protocol (MCP) Bridge")
        print("  - Parallel Tool Dispatch & Error Self-Correction")
        print("Public Hub: https://huggingface.co/meetmendapara/Vaayu-Base")
    elif args.command == "chat":
        print(f"Loading {args.model} ({args.variant})...")
        ai = Vaayu.from_pretrained(args.model, variant=args.variant)
        print(f"\nPrompt: {args.prompt}\n")
        response = ai.chat(args.prompt)
        print(f"\nResponse:\n{response}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()