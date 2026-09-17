"""
Command-line interface for Vaayu SLMM.
Usage:
    vaayu info
    vaayu chat "Inspect git repository status" --stream
    vaayu repl
    vaayu serve --port 8000
    vaayu benchmark --limit 50
"""

import os
import sys
import argparse
from typing import Optional

# Enable ANSI escape sequences on Windows console
if sys.platform == "win32":
    os.system("")

from vaayu import Vaayu, __version__

def run_repl(ai: Vaayu):
    """Interactive developer console with streaming events and syntax highlighting."""
    print("==================================================")
    print(f" Vaayu Interactive REPL ({ai.engine.variant.upper()})")
    print(" Type 'exit', 'quit', or Ctrl+C to terminate.")
    print(" Type 'tools' to inspect currently registered tools.")
    print("==================================================")

    while True:
        try:
            prompt = input("\033[1;34mvaayu>\033[0m ").strip()
            if not prompt:
                continue
            if prompt.lower() in ("exit", "quit"):
                print("Bye!")
                break
            if prompt.lower() == "tools":
                tools = ai.get_all_tools()
                print(f"\033[33mRegistered Tools ({len(tools)}):\033[0m")
                for t in tools:
                    print(f"  - \033[1m{t.get('name')}\033[0m: {t.get('description')}")
                continue

            print()
            for event in ai.stream_chat(prompt):
                if event.type == "thought":
                    print(f"\033[2;36m[Thought] {event.content}\033[0m", end="", flush=True)
                elif event.type == "tool_call":
                    print(f"\n\033[1;33m[Tool Call] {event.tool_name}\033[0m({event.arguments})", flush=True)
                elif event.type == "tool_result":
                    print(f"\033[35m[Tool Result] {event.content}\033[0m", flush=True)
                elif event.type == "text":
                    print(f"\033[32m{event.content}\033[0m", end="", flush=True)
            print("\n")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting REPL.")
            break

def main():
    parser = argparse.ArgumentParser(
        prog="vaayu",
        description="Vaayu SLMM: Machine-Centric Small Language Model CLI & Runtime"
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
    chat_parser.add_argument("--stream", action="store_true", help="Stream tokens and events in real time")
    chat_parser.add_argument("--tools", action="store_true", help="Enable default filesystem/http/sqlite tools")

    # repl
    repl_parser = subparsers.add_parser("repl", aliases=["interactive"], help="Launch interactive developer REPL session")
    repl_parser.add_argument("--model", type=str, default="meetmendapara/Vaayu-Base", help="Model checkpoint or Hugging Face ID")
    repl_parser.add_argument("--variant", type=str, default="base", choices=["base", "large"])
    repl_parser.add_argument("--tools", action="store_true", default=True, help="Enable default filesystem/http/sqlite tools")

    # serve
    serve_parser = subparsers.add_parser("serve", help="Start OpenAI-compatible HTTP server (/v1/chat/completions)")
    serve_parser.add_argument("--model", type=str, default="meetmendapara/Vaayu-Base", help="Model checkpoint or Hugging Face ID")
    serve_parser.add_argument("--variant", type=str, default="base", choices=["base", "large"])
    serve_parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface (default: 0.0.0.0)")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")

    # benchmark
    bench_parser = subparsers.add_parser("benchmark", help="Run BFCL v3 benchmark evaluation suite")
    bench_parser.add_argument("--limit", type=int, default=50, help="Number of benchmark test cases to evaluate")
    bench_parser.add_argument("--variant", type=str, default="base", choices=["base", "large"])

    args = parser.parse_args()

    if args.command == "info":
        print("==================================================")
        print(f" Vaayu SLMM Engine (v{__version__})")
        print("==================================================")
        print("Architecture: Custom Transformer Decoder")
        print("Variants:")
        print("  - Vaayu-Base:  245M Parameters (RoPE, 4:1 GQA, SwiGLU, 2k ctx)")
        print("  - Vaayu-Large: 492M Parameters (RoPE, 4:1 GQA, SwiGLU, 4k ctx)")
        print("Capabilities:")
        print("  - In-Process Python Tool Decorator (@ai.tool)")
        print("  - Batteries-Included Tools (Filesystem, HTTP, SQLite, Shell)")
        print("  - Real-Time Streaming & Observability (StreamEvent)")
        print("  - OpenAI-Compatible REST Server (vaayu serve)")
        print("  - Native Model Context Protocol (MCP) Bridge")
        print("  - Parallel Tool Dispatch & Error Self-Correction")
        print("Public Hub: https://huggingface.co/meetmendapara/Vaayu-Base")

    elif args.command == "chat":
        print(f"Loading {args.model} ({args.variant})...")
        ai = Vaayu.from_pretrained(args.model, variant=args.variant)
        if args.tools:
            ai.enable_default_tools()
        print(f"\nPrompt: {args.prompt}\n")

        if args.stream:
            for event in ai.stream_chat(args.prompt):
                if event.type == "thought":
                    print(f"\033[2;36m{event.content}\033[0m", end="", flush=True)
                elif event.type == "tool_call":
                    print(f"\n\033[33m[Call: {event.tool_name}({event.arguments})]\033[0m\n", flush=True)
                elif event.type == "tool_result":
                    print(f"\033[35m[Result: {event.content}]\033[0m\n", flush=True)
                elif event.type == "text":
                    print(event.content, end="", flush=True)
            print()
        else:
            response = ai.chat(args.prompt)
            print(f"\nResponse:\n{response}")

    elif args.command in ("repl", "interactive"):
        print(f"Loading {args.model} ({args.variant})...")
        ai = Vaayu.from_pretrained(args.model, variant=args.variant)
        if args.tools:
            ai.enable_default_tools()
        run_repl(ai)

    elif args.command == "serve":
        from .server import start_server
        start_server(model=args.model, host=args.host, port=args.port, variant=args.variant)

    elif args.command == "benchmark":
        from eval.bfcl_standard_benchmark import run_bfcl_evaluation
        print(f"Running BFCL v3 benchmark on {args.variant} (limit: {args.limit})...")
        run_bfcl_evaluation(limit=args.limit, variant=args.variant)

    else:
        parser.print_help()

if __name__ == "__main__":
    main()