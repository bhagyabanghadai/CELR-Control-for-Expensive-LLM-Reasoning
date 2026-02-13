"""
CELR Interactive Chat — Talk to AI in your terminal.

A user-friendly chat interface where anyone can:
  1. Pick their AI model (OpenAI, Anthropic, Groq, or Local Ollama)
  2. Set a budget
  3. Chat back and forth with the AI
  4. See cost tracking in real-time

Usage:
    python celr_chat.py

No coding knowledge needed. Just run and talk.
"""

import os
import sys
import time
import logging

from dotenv import load_dotenv

# Suppress noisy logs for clean chat experience
logging.basicConfig(level=logging.WARNING)
logging.getLogger("celr").setLevel(logging.WARNING)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

load_dotenv()

# ─── Pretty Printing ──────────────────────────────────────────────

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.markdown import Markdown
    from rich.prompt import Prompt, FloatPrompt, IntPrompt
    from rich.text import Text
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


class ChatUI:
    """Simple terminal UI that works with or without Rich."""

    def __init__(self):
        if HAS_RICH:
            self.console = Console()
        else:
            self.console = None

    def clear(self):
        os.system("cls" if os.name == "nt" else "clear")

    def banner(self):
        if HAS_RICH:
            self.console.print(Panel(
                "[bold cyan]CELR[/bold cyan] — [white]Control for Expensive LLM Reasoning[/white]\n"
                "[dim]Your personal AI assistant in the terminal[/dim]",
                border_style="cyan",
                padding=(1, 2),
            ))
        else:
            print("=" * 55)
            print("  CELR — Control for Expensive LLM Reasoning")
            print("  Your personal AI assistant in the terminal")
            print("=" * 55)
        print()

    def section(self, title):
        if HAS_RICH:
            self.console.print(f"\n[bold yellow]{title}[/bold yellow]")
        else:
            print(f"\n{title}")

    def info(self, text):
        if HAS_RICH:
            self.console.print(f"[dim]{text}[/dim]")
        else:
            print(text)

    def success(self, text):
        if HAS_RICH:
            self.console.print(f"[bold green]{text}[/bold green]")
        else:
            print(text)

    def error(self, text):
        if HAS_RICH:
            self.console.print(f"[bold red]{text}[/bold red]")
        else:
            print(f"ERROR: {text}")

    def warning(self, text):
        if HAS_RICH:
            self.console.print(f"[yellow]{text}[/yellow]")
        else:
            print(f"WARNING: {text}")

    def ai_response(self, text):
        if HAS_RICH:
            try:
                self.console.print(Markdown(text))
            except Exception:
                self.console.print(text)
        else:
            print(text)

    def cost_bar(self, spent, budget):
        pct = min(spent / budget * 100, 100) if budget > 0 else 0
        remaining = budget - spent
        if HAS_RICH:
            color = "green" if pct < 50 else "yellow" if pct < 80 else "red"
            self.console.print(
                f"  [{color}]${spent:.4f} spent[/{color}] "
                f"[dim]| ${remaining:.4f} remaining | {pct:.0f}% used[/dim]"
            )
        else:
            print(f"  ${spent:.4f} spent | ${remaining:.4f} remaining | {pct:.0f}% used")


ui = ChatUI()


# ─── Model Selection ──────────────────────────────────────────────

MODELS = {
    "1": {
        "name": "GPT-4o Mini",
        "model": "gpt-4o-mini",
        "provider": "openai",
        "key_env": "OPENAI_API_KEY",
        "cost": "~$0.15 / 1M input tokens (very cheap)",
        "description": "Fast, cheap, great for most tasks",
    },
    "2": {
        "name": "GPT-4o",
        "model": "gpt-4o",
        "provider": "openai",
        "key_env": "OPENAI_API_KEY",
        "cost": "~$5.00 / 1M input tokens (moderate)",
        "description": "Smartest OpenAI model. Better for complex tasks",
    },
    "3": {
        "name": "Claude 3.5 Sonnet",
        "model": "claude-3-5-sonnet-20241022",
        "provider": "anthropic",
        "key_env": "ANTHROPIC_API_KEY",
        "cost": "~$3.00 / 1M input tokens (moderate)",
        "description": "Excellent for writing, analysis, and coding",
    },
    "4": {
        "name": "Groq (Llama 3 70B)",
        "model": "groq/llama-3.3-70b-versatile",
        "provider": "groq",
        "key_env": "GROQ_API_KEY",
        "cost": "~$0.59 / 1M input tokens (cheap + fast)",
        "description": "Ultra-fast inference. Great speed-to-cost ratio",
    },
    "5": {
        "name": "Ollama (Local)",
        "model": "ollama/llama3",
        "provider": "ollama",
        "key_env": None,
        "cost": "FREE (runs on your computer)",
        "description": "100% private, no internet needed. Requires Ollama installed",
    },
}


def detect_available_models():
    """Check which API keys are set."""
    available = {}
    for key, model in MODELS.items():
        if model["key_env"] is None:
            # Local model — always available (user may need Ollama running)
            available[key] = model
        elif os.getenv(model["key_env"]):
            available[key] = model
    return available


def show_model_menu():
    """Display model selection and return chosen model."""
    available = detect_available_models()
    all_models = MODELS

    ui.section("Choose your AI Model:")
    print()

    if HAS_RICH:
        table = Table(show_header=True, header_style="bold cyan", show_lines=True)
        table.add_column("#", style="bold", width=3)
        table.add_column("Model", style="cyan", min_width=20)
        table.add_column("Cost", min_width=25)
        table.add_column("Status", width=12)
        table.add_column("Description", min_width=30)

        for key, model in all_models.items():
            is_available = key in available
            status = "[green]Ready[/green]" if is_available else "[red]No API Key[/red]"
            if model["key_env"] is None:
                status = "[yellow]Local[/yellow]"
            name_style = "bold" if is_available else "dim"
            table.add_row(
                key,
                f"[{name_style}]{model['name']}[/{name_style}]",
                model["cost"],
                status,
                model["description"],
            )
        ui.console.print(table)
    else:
        for key, model in all_models.items():
            is_available = key in available
            status = "[Ready]" if is_available else "[No API Key]"
            if model["key_env"] is None:
                status = "[Local]"
            print(f"  {key}. {model['name']:20s} {status:15s} {model['cost']}")

    print()

    if not available:
        ui.error("No AI models available!")
        print()
        print("  To use CELR, you need at least one of:")
        print("    - OpenAI API key    → set OPENAI_API_KEY in .env")
        print("    - Anthropic API key → set ANTHROPIC_API_KEY in .env")
        print("    - Groq API key      → set GROQ_API_KEY in .env")
        print("    - Ollama running    → install from https://ollama.com")
        print()
        print("  Copy .env.example to .env and add your key(s).")
        return None

    while True:
        choice = input("  Enter number (or press Enter for auto-select): ").strip()

        if choice == "":
            # Auto-select first available
            first = list(available.keys())[0]
            model = available[first]
            ui.success(f"  Auto-selected: {model['name']}")
            return model

        if choice in available:
            model = available[choice]
            ui.success(f"  Selected: {model['name']}")
            return model

        if choice in all_models and choice not in available:
            needed_key = all_models[choice]["key_env"]
            ui.error(f"  {all_models[choice]['name']} requires {needed_key} in your .env file.")
            continue

        ui.error(f"  Invalid choice '{choice}'. Please enter a number 1-{len(all_models)}.")


def get_budget():
    """Ask user for budget."""
    ui.section("Set your budget:")
    print()
    print("  How much are you willing to spend this session?")
    print("  (Typical conversation costs $0.01 - $0.10)")
    print()

    while True:
        raw = input("  Budget in USD (default $0.50): $").strip()
        if raw == "":
            ui.info("  Using default budget: $0.50")
            return 0.50
        try:
            budget = float(raw)
            if budget <= 0:
                ui.error("  Budget must be positive.")
                continue
            if budget > 10:
                ui.warning(f"  That's ${budget:.2f} — are you sure? (yes/no): ")
                confirm = input("  ").strip().lower()
                if confirm not in ("yes", "y"):
                    continue
            return budget
        except ValueError:
            ui.error("  Please enter a number (e.g., 0.50)")


# ─── Chat Engine ──────────────────────────────────────────────────

def create_chat_engine(model_info, budget):
    """Build the CELR pipeline for chat."""
    from celr.core.config import CELRConfig
    from celr.core.types import TaskContext, ModelConfig
    from celr.core.cost_tracker import CostTracker
    from celr.core.llm import LiteLLMProvider, LLMUsage

    # Create a model config for the chosen model
    model_config = ModelConfig(
        name=model_info["model"],
        provider=model_info["provider"],
        cost_per_million_input_tokens=0.15,  # Default, actual cost from LiteLLM
        cost_per_million_output_tokens=0.60,
    )

    provider = LiteLLMProvider(model_config)
    context = TaskContext(original_request="interactive_chat", budget_limit_usd=budget)
    tracker = CostTracker(context)

    return provider, context, tracker


def chat_loop(provider, context, tracker, model_name):
    """Main conversation loop."""
    conversation_history = []
    message_count = 0

    print()
    ui.success("  Chat is ready! Start typing your questions.")
    print()
    ui.info("  Commands:")
    ui.info("    /help     — Show available commands")
    ui.info("    /cost     — Show current spending")
    ui.info("    /clear    — Clear conversation history")
    ui.info("    /exit     — End the session")
    print()

    system_prompt = (
        "You are a helpful AI assistant powered by CELR. "
        "Be concise, clear, and helpful. "
        "When asked to write code, provide clean, well-commented code. "
        "When asked questions, give direct, accurate answers."
    )

    while True:
        # Show prompt
        try:
            user_input = input("  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.startswith("/"):
            cmd = user_input.lower().split()[0]

            if cmd in ("/exit", "/quit", "/bye"):
                break

            elif cmd == "/help":
                print()
                ui.info("  Available commands:")
                ui.info("    /help     — This help message")
                ui.info("    /cost     — Show spending summary")
                ui.info("    /clear    — Start fresh conversation")
                ui.info("    /model    — Show current model info")
                ui.info("    /exit     — End session")
                print()
                continue

            elif cmd == "/cost":
                print()
                ui.cost_bar(context.current_spread_usd, context.budget_limit_usd)
                ui.info(f"  Messages sent: {message_count}")
                print()
                continue

            elif cmd == "/clear":
                conversation_history.clear()
                message_count = 0
                ui.success("  Conversation cleared.")
                print()
                continue

            elif cmd == "/model":
                ui.info(f"  Model: {model_name}")
                ui.info(f"  Messages: {message_count}")
                print()
                continue

            else:
                ui.warning(f"  Unknown command: {cmd}. Type /help for options.")
                continue

        # Check budget
        remaining = context.budget_limit_usd - context.current_spread_usd
        if remaining <= 0:
            print()
            ui.error("  Budget exhausted! Session ended.")
            ui.info(f"  Total spent: ${context.current_spread_usd:.4f}")
            break

        # Add user message to history
        conversation_history.append({"role": "user", "content": user_input})

        # Build the full prompt with conversation context
        full_prompt = ""
        for msg in conversation_history[-10:]:  # Keep last 10 messages for context
            role = "User" if msg["role"] == "user" else "Assistant"
            full_prompt += f"{role}: {msg['content']}\n\n"

        # Call LLM
        try:
            start = time.time()

            if HAS_RICH:
                with ui.console.status("[bold cyan]  Thinking...", spinner="dots"):
                    response, usage = provider.generate(
                        prompt=full_prompt,
                        system_prompt=system_prompt,
                    )
            else:
                print("  Thinking...")
                response, usage = provider.generate(
                    prompt=full_prompt,
                    system_prompt=system_prompt,
                )

            elapsed = time.time() - start

            # Track cost
            cost = provider.calculate_cost(usage)
            tracker.add_cost(cost)
            message_count += 1

            # Add AI response to history
            conversation_history.append({"role": "assistant", "content": response})

            # Display response
            print()
            if HAS_RICH:
                ui.console.print(Panel(
                    Markdown(response),
                    title=f"[cyan]{model_name}[/cyan]",
                    subtitle=f"[dim]${cost:.4f} | {elapsed:.1f}s | {usage.total_tokens} tokens[/dim]",
                    border_style="blue",
                    padding=(0, 1),
                ))
            else:
                print(f"  AI: {response}")
                print(f"  [{cost:.4f} USD | {elapsed:.1f}s | {usage.total_tokens} tokens]")
            print()

        except Exception as e:
            error_msg = str(e)
            print()

            if "api_key" in error_msg.lower() or "auth" in error_msg.lower():
                ui.error("  Authentication failed. Check your API key in .env")
            elif "rate" in error_msg.lower() or "limit" in error_msg.lower():
                ui.warning("  Rate limited. Wait a moment and try again.")
            elif "connection" in error_msg.lower() or "timeout" in error_msg.lower():
                if "ollama" in model_name.lower():
                    ui.error("  Cannot connect to Ollama. Is it running? (ollama serve)")
                else:
                    ui.error("  Connection failed. Check your internet.")
            else:
                ui.error(f"  Error: {error_msg}")
            print()

    return message_count


# ─── Session Summary ──────────────────────────────────────────────

def show_summary(context, message_count, model_name):
    """End-of-session summary."""
    print()
    if HAS_RICH:
        table = Table(title="Session Summary", show_lines=True, border_style="cyan")
        table.add_column("Metric", style="bold")
        table.add_column("Value", style="cyan")
        table.add_row("Model", model_name)
        table.add_row("Messages", str(message_count))
        table.add_row("Total Cost", f"${context.current_spread_usd:.4f}")
        table.add_row("Budget Remaining", f"${context.budget_limit_usd - context.current_spread_usd:.4f}")
        if message_count > 0:
            avg = context.current_spread_usd / message_count
            table.add_row("Avg Cost/Message", f"${avg:.4f}")
        ui.console.print(table)
    else:
        print("  ── Session Summary ──")
        print(f"  Model:            {model_name}")
        print(f"  Messages:         {message_count}")
        print(f"  Total Cost:       ${context.current_spread_usd:.4f}")
        print(f"  Budget Remaining: ${context.budget_limit_usd - context.current_spread_usd:.4f}")

    print()
    ui.info("  Thanks for using CELR! Goodbye.")
    print()


# ─── Main ─────────────────────────────────────────────────────────

def main():
    ui.clear()
    ui.banner()

    # 1. Pick model
    model_info = show_model_menu()
    if model_info is None:
        sys.exit(1)

    # 2. Set budget
    budget = get_budget()

    # 3. Build engine
    print()
    ui.info(f"  Setting up {model_info['name']}...")
    try:
        provider, context, tracker = create_chat_engine(model_info, budget)
    except Exception as e:
        ui.error(f"  Failed to initialize: {e}")
        sys.exit(1)

    ui.clear()
    ui.banner()
    ui.info(f"  Model: {model_info['name']}  |  Budget: ${budget:.2f}")

    # 4. Chat loop
    message_count = chat_loop(provider, context, tracker, model_info["name"])

    # 5. Summary
    show_summary(context, message_count, model_info["name"])


if __name__ == "__main__":
    main()
