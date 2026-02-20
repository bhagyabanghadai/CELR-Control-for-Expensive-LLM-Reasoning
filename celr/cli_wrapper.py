#!/usr/bin/env python3
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

def run_task(prompt: str, budget: float, ui: bool, verbose: bool, small: str, mid: str, large: str):
    """Run a headless task via celr.cli module."""
    # We use subprocess to run the module so it picks up the environment correctly
    cmd = [sys.executable, "-m", "celr.cli", prompt]
    if budget:
        cmd.extend(["--budget", str(budget)])
    if ui:
        cmd.append("--ui")
    if verbose:
        cmd.append("--verbose")
    if small:
        cmd.extend(["--small-model", small])
    if large:
        cmd.extend(["--large-model", large])
    # Note: celr.cli doesn't have --mid-model flag yet, but we can pass it via env if needed,
    # or just rely on config/env vars for that.
    
    # Pass mid model via environment variable if specified
    env = os.environ.copy()
    if mid:
        env["CELR_MID_MODEL"] = mid

    subprocess.run(cmd, env=env)

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

    # celr chat
    subparsers.add_parser("chat", help="Launch interactive chat")

    args = parser.parse_args()

    if args.command == "init":
        run_init()
    elif args.command == "run":
        run_task(
            args.prompt, 
            args.budget, 
            args.ui, 
            args.verbose, 
            args.small_model, 
            args.mid_model, 
            args.large_model
        )
    elif args.command == "chat":
        run_chat()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
