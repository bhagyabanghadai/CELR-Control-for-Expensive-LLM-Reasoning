#!/usr/bin/env python3
"""
CELR CLI Wrapper — unified entry point.

Routes to subcommands:
  celr init    — configuration wizard
  celr run     — execute a headless task (delegates to celr.cli)
  celr chat    — launch interactive chat
"""
import sys
import argparse
import subprocess
import os
from celr.core.config import CELRConfig


def run_init():
    """Run the configuration wizard."""
    print("🧙‍♂️ CELR Configuration Wizard")
    print("----------------------------")
    
    # 1. Check for Ollama
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Ollama detected locally.")
            print("   Available models:")
            for line in result.stdout.splitlines()[1:]:
                if line.strip():
                    print(f"     - {line.split()[0]}")
        else:
            print("⚠️  Ollama found but returned error. Is it running?")
    except FileNotFoundError:
        print("❌ Ollama not found in PATH. Install from https://ollama.com/ for local models.")

    # 2. API Keys
    print("\n🔑 API Configuration")
    openai_key = input("Enter OpenAI API Key (leave blank to skip): ").strip()
    anthropic_key = input("Enter Anthropic API Key (leave blank to skip): ").strip()
    
    env_content = []
    if openai_key:
        env_content.append(f"CELR_OPENAI_API_KEY={openai_key}")
    if anthropic_key:
        env_content.append(f"CELR_ANTHROPIC_API_KEY={anthropic_key}")
        
    # Write .env
    if env_content:
        with open(".env", "w") as f:
            f.write("\n".join(env_content) + "\n")
        print("\n✅ Saved to .env")
    else:
        print("\nℹ️  No API keys provided. CELR will use local Ollama models by default.")


def run_task(args):
    """Run a headless task by calling celr.cli.main() directly."""
    # Build sys.argv for cli.main() as if called from command line
    cli_args = [args.prompt]
    if args.budget is not None:
        cli_args.extend(["--budget", str(args.budget)])
    if args.ui:
        cli_args.append("--ui")
    if args.verbose:
        cli_args.append("--verbose")
    if args.small_model:
        cli_args.extend(["--small-model", args.small_model])
    if args.large_model:
        cli_args.extend(["--large-model", args.large_model])
    if args.reliability_mode:
        cli_args.extend(["--reliability-mode", args.reliability_mode])

    # Pass mid model via environment variable if specified
    if args.mid_model:
        os.environ["CELR_MID_MODEL"] = args.mid_model

    # Direct call instead of subprocess — shares environment, faster startup
    original_argv = sys.argv
    try:
        sys.argv = ["celr"] + cli_args
        from celr.cli import main as cli_main
        cli_main()
    finally:
        sys.argv = original_argv


def run_chat():
    """Launch the interactive chat TUI."""
    try:
        from celr.interface.chat import main as chat_main
        chat_main()
    except ImportError:
        print("❌ Error: Could not import chat interface. Is 'celr.interface.chat' available?")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="CELR: Control for Expensive LLM Reasoning")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # celr init
    subparsers.add_parser("init", help="Run configuration wizard")

    # celr run "task"
    run_parser = subparsers.add_parser("run", help="Execute a task")
    run_parser.add_argument("prompt", help="The task prompt")
    run_parser.add_argument("--budget", type=float, default=None, help="Max budget in USD")
    run_parser.add_argument("--ui", action="store_true", help="Launch Cerebro war room dashboard")
    run_parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    run_parser.add_argument("--small-model", type=str, default=None, help="Override small/reasoning model")
    run_parser.add_argument("--mid-model", type=str, default=None, help="Override mid-tier model")
    run_parser.add_argument("--large-model", type=str, default=None, help="Override large/expensive model")
    run_parser.add_argument("--reliability-mode", type=str, default=None,
                            choices=["balanced", "strict", "research"],
                            help="Reliability mode: balanced (default), strict, research")

    # celr chat
    subparsers.add_parser("chat", help="Launch interactive chat")

    args = parser.parse_args()

    if args.command == "init":
        run_init()
    elif args.command == "run":
        run_task(args)
    elif args.command == "chat":
        run_chat()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
