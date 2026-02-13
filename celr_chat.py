"""
CELR Interactive Chat — Talk to AI in your terminal.

Handles ALL user scenarios:
  - No API keys, no Ollama          → step-by-step setup guide
  - Ollama installed but not running → detects and tells user
  - Ollama running, no model pulled → detects and tells user
  - Has API key but it's invalid    → catches error, offers re-entry
  - User doesn't know .env file     → lets them paste key directly
  - Using free local model           → skips budget question

Usage:
    python celr_chat.py
"""

import os
import sys
import time
import json
import logging
import urllib.request
import urllib.error

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
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


class ChatUI:
    """Terminal UI that works with or without Rich."""

    def __init__(self):
        self.console = Console() if HAS_RICH else None

    def clear(self):
        os.system("cls" if os.name == "nt" else "clear")

    def banner(self):
        if HAS_RICH:
            self.console.print(Panel(
                "[bold cyan]CELR[/bold cyan] — [white]Control for Expensive LLM Reasoning[/white]\n"
                "[dim]Your personal AI assistant in the terminal[/dim]",
                border_style="cyan", padding=(1, 2),
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

    def ai_response(self, text, model_name, cost, elapsed, tokens):
        print()
        if HAS_RICH:
            try:
                content = Markdown(text)
            except Exception:
                content = text
            subtitle = f"[dim]${cost:.4f} | {elapsed:.1f}s | {tokens} tokens[/dim]" if cost > 0 else f"[dim]FREE | {elapsed:.1f}s | {tokens} tokens[/dim]"
            self.console.print(Panel(
                content,
                title=f"[cyan]{model_name}[/cyan]",
                subtitle=subtitle,
                border_style="blue", padding=(0, 1),
            ))
        else:
            print(f"  AI: {text}")
            cost_str = f"${cost:.4f}" if cost > 0 else "FREE"
            print(f"  [{cost_str} | {elapsed:.1f}s | {tokens} tokens]")
        print()

    def cost_bar(self, spent, budget):
        if budget <= 0:
            self.info("  Using free local model - no cost")
            return
        pct = min(spent / budget * 100, 100)
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


# ─── Ollama Detection ─────────────────────────────────────────────

def check_ollama_running():
    """Check if Ollama server is running on localhost."""
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def get_ollama_models():
    """Get list of models pulled in Ollama."""
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode())
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


# ─── Model Definitions ────────────────────────────────────────────

CLOUD_MODELS = {
    "1": {
        "name": "GPT-4o Mini (Recommended)",
        "model": "gpt-4o-mini",
        "provider": "openai",
        "key_env": "OPENAI_API_KEY",
        "cost_info": "~$0.15/1M tokens (very cheap)",
        "description": "Fast, cheap, great for most tasks",
        "free": False,
    },
    "2": {
        "name": "GPT-4o",
        "model": "gpt-4o",
        "provider": "openai",
        "key_env": "OPENAI_API_KEY",
        "cost_info": "~$5/1M tokens",
        "description": "Most capable OpenAI model",
        "free": False,
    },
    "3": {
        "name": "Claude 3.5 Sonnet",
        "model": "claude-3-5-sonnet-20241022",
        "provider": "anthropic",
        "key_env": "ANTHROPIC_API_KEY",
        "cost_info": "~$3/1M tokens",
        "description": "Great for writing and analysis",
        "free": False,
    },
    "4": {
        "name": "Groq (Llama 3 — Ultra Fast)",
        "model": "groq/llama-3.3-70b-versatile",
        "provider": "groq",
        "key_env": "GROQ_API_KEY",
        "cost_info": "~$0.59/1M tokens + blazing fast",
        "description": "Fastest inference available",
        "free": False,
    },
}


def build_model_list():
    """Build complete model list with availability status."""
    models = {}
    idx = 1

    # -- Check cloud models --
    for key, model in CLOUD_MODELS.items():
        has_key = bool(os.getenv(model["key_env"], "").strip())
        models[str(idx)] = {
            **model,
            "available": has_key,
            "status_reason": "Ready" if has_key else f"Need {model['key_env']}",
        }
        idx += 1

    # -- Check Ollama --
    ollama_running = check_ollama_running()
    if ollama_running:
        ollama_models = get_ollama_models()
        if ollama_models:
            # Add each pulled Ollama model
            for om in ollama_models[:3]:  # Show up to 3 local models
                display_name = om.split(":")[0]
                models[str(idx)] = {
                    "name": f"Ollama — {display_name} (Local, Free)",
                    "model": f"ollama/{om}",
                    "provider": "ollama",
                    "key_env": None,
                    "cost_info": "FREE",
                    "description": "Runs on your computer, 100% private",
                    "free": True,
                    "available": True,
                    "status_reason": "Ready (local)",
                }
                idx += 1
        else:
            models[str(idx)] = {
                "name": "Ollama (No models pulled)",
                "model": "ollama/llama3",
                "provider": "ollama",
                "key_env": None,
                "cost_info": "FREE",
                "description": "Run: ollama pull llama3",
                "free": True,
                "available": False,
                "status_reason": "Run: ollama pull llama3",
            }
            idx += 1
    else:
        models[str(idx)] = {
            "name": "Ollama (Not running)",
            "model": "ollama/llama3",
            "provider": "ollama",
            "key_env": None,
            "cost_info": "FREE",
            "description": "Start with: ollama serve",
            "free": True,
            "available": False,
            "status_reason": "Start: ollama serve",
        }
        idx += 1

    return models


# ─── Model Selection ──────────────────────────────────────────────

def show_model_menu():
    """Display model menu, handle selection, and return chosen model."""
    models = build_model_list()
    available = {k: v for k, v in models.items() if v["available"]}

    ui.section("Choose your AI Model:")
    print()

    if HAS_RICH:
        table = Table(show_header=True, header_style="bold cyan", show_lines=True)
        table.add_column("#", style="bold", width=3)
        table.add_column("Model", min_width=25)
        table.add_column("Cost", min_width=20)
        table.add_column("Status", width=18)

        for key, m in models.items():
            if m["available"]:
                status = "[green]Ready[/green]"
                name_fmt = f"[bold]{m['name']}[/bold]"
            else:
                status = f"[red]{m['status_reason']}[/red]"
                name_fmt = f"[dim]{m['name']}[/dim]"
            table.add_row(key, name_fmt, m["cost_info"], status)

        ui.console.print(table)
    else:
        for key, m in models.items():
            status = "Ready" if m["available"] else m["status_reason"]
            marker = "*" if m["available"] else " "
            print(f"  {marker} {key}. {m['name']:30s} {m['cost_info']:20s} [{status}]")

    print()

    # -- No models at all --
    if not available:
        return handle_no_models()

    # -- Selection loop --
    while True:
        choice = input("  Enter number (or press Enter for best available): ").strip()

        if choice == "":
            first_key = list(available.keys())[0]
            model = available[first_key]
            ui.success(f"\n  Auto-selected: {model['name']}")
            return model

        if choice in available:
            ui.success(f"\n  Selected: {models[choice]['name']}")
            return models[choice]

        if choice in models and choice not in available:
            m = models[choice]
            reason = m["status_reason"]

            # Offer to enter API key right now
            if m["key_env"]:
                return offer_key_entry(m)
            else:
                # Ollama not ready
                print()
                if "pull" in reason.lower():
                    ui.warning(f"  You need to pull a model first.")
                    ui.info(f"  Open another terminal and run: ollama pull llama3")
                    ui.info(f"  Then come back and try again.")
                else:
                    ui.warning(f"  Ollama is not running.")
                    ui.info(f"  Open another terminal and run: ollama serve")
                    ui.info(f"  Then restart this chat.")
                print()
                continue

        ui.error(f"  Invalid choice. Enter 1-{len(models)}.")


def handle_no_models():
    """Guide user when no models are available at all."""
    print()
    ui.error("  No AI models are available yet. Let's set one up!")
    print()
    ui.info("  You have two options:\n")

    ui.section("  Option A: Use a FREE local model (Ollama)")
    ui.info("    1. Download Ollama from https://ollama.com")
    ui.info("    2. Open a terminal and run: ollama serve")
    ui.info("    3. In another terminal run:  ollama pull llama3")
    ui.info("    4. Come back and run: python celr_chat.py")
    print()

    ui.section("  Option B: Use a cloud AI (needs API key)")
    ui.info("    1. Go to https://platform.openai.com/api-keys")
    ui.info("       (or https://console.anthropic.com for Claude)")
    ui.info("       (or https://console.groq.com for Groq — has free tier!)")
    ui.info("    2. Create an API key")
    ui.info("    3. You can either:")
    ui.info("       a) Paste it when prompted here (I'll ask next)")
    ui.info("       b) Add it to a .env file in this folder")

    print()
    answer = input("  Do you have an API key to enter now? (yes/no): ").strip().lower()

    if answer in ("yes", "y"):
        return enter_api_key_flow()

    print()
    ui.info("  No problem! Set up one of the options above and run again.")
    ui.info("  Tip: Groq (https://console.groq.com) has a FREE tier!")
    print()
    return None


def offer_key_entry(model_info):
    """Let user paste their API key right in the terminal."""
    key_name = model_info["key_env"]
    print()
    ui.info(f"  {model_info['name']} needs an API key ({key_name}).")
    ui.info(f"  You can enter it now and I'll use it for this session.")
    print()

    key = input(f"  Paste your {key_name} (or press Enter to skip): ").strip()

    if not key:
        return None

    # Set it for this session
    os.environ[key_name] = key

    # Offer to save to .env
    save = input("  Save this key to .env for next time? (yes/no): ").strip().lower()
    if save in ("yes", "y"):
        save_key_to_env(key_name, key)
        ui.success(f"  Saved to .env!")

    ui.success(f"  Key set! Using {model_info['name']}")
    model_info["available"] = True
    return model_info


def enter_api_key_flow():
    """Walk user through entering any API key."""
    print()
    ui.section("  Which provider?")
    print("    1. OpenAI (GPT-4o)")
    print("    2. Anthropic (Claude)")
    print("    3. Groq (Llama 3 — has FREE tier!)")
    print()

    providers = {
        "1": ("OPENAI_API_KEY", "1"),
        "2": ("ANTHROPIC_API_KEY", "3"),
        "3": ("GROQ_API_KEY", "4"),
    }

    choice = input("  Enter 1, 2, or 3: ").strip()
    if choice not in providers:
        ui.error("  Invalid choice.")
        return None

    key_name, model_key = providers[choice]
    key = input(f"\n  Paste your {key_name}: ").strip()

    if not key:
        return None

    os.environ[key_name] = key

    save = input("  Save to .env for next time? (yes/no): ").strip().lower()
    if save in ("yes", "y"):
        save_key_to_env(key_name, key)
        ui.success("  Saved!")

    # Return the corresponding cloud model
    model = CLOUD_MODELS[model_key].copy()
    model["available"] = True
    model["status_reason"] = "Ready"
    return model


def save_key_to_env(key_name, key_value):
    """Append or update a key in the .env file."""
    env_path = os.path.join(os.path.dirname(__file__), ".env")

    # Read existing content
    lines = []
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            lines = f.readlines()

    # Check if key already exists
    found = False
    for i, line in enumerate(lines):
        if line.strip().startswith(f"{key_name}="):
            lines[i] = f"{key_name}={key_value}\n"
            found = True
            break

    if not found:
        lines.append(f"{key_name}={key_value}\n")

    with open(env_path, "w") as f:
        f.writelines(lines)


# ─── Budget ───────────────────────────────────────────────────────

def get_budget(model_info):
    """Ask for budget, or skip if using free model."""
    if model_info.get("free"):
        ui.info("  Using free local model — no budget needed!")
        return 0.0  # Unlimited for local

    ui.section("Set your budget:")
    print()
    ui.info("  How much are you willing to spend this session?")
    ui.info("  Typical conversation: $0.01 - $0.10")
    print()

    while True:
        raw = input("  Budget in USD (default $0.50): $").strip()
        if raw == "":
            ui.info("  Using default: $0.50")
            return 0.50
        try:
            budget = float(raw)
            if budget <= 0:
                ui.error("  Must be positive.")
                continue
            if budget > 10:
                confirm = input(f"  That's ${budget:.2f} — sure? (yes/no): ").strip().lower()
                if confirm not in ("yes", "y"):
                    continue
            return budget
        except ValueError:
            ui.error("  Enter a number (e.g., 0.50)")


# ─── Chat Engine ──────────────────────────────────────────────────

def create_chat_engine(model_info, budget):
    """Build the LLM provider for chat."""
    # Core Imports
    from celr.core.types import TaskContext, ModelConfig
    from celr.core.cost_tracker import CostTracker
    from celr.core.llm import LiteLLMProvider
    from celr.core.planner import Planner
    from celr.core.reasoning import ReasoningCore
    from celr.core.executor import TaskExecutor
    from celr.core.verifier import Verifier
    from celr.core.tools import ToolRegistry
    from celr.core.reflection import SelfReflection
    from celr.core.escalation import EscalationManager
    from celr.cortex.router import Router
    from unittest.mock import MagicMock

    model_config = ModelConfig(
        name=model_info["model"],
        provider=model_info["provider"],
        # Default rates (fallback)
        cost_per_million_input_tokens=0.15,
        cost_per_million_output_tokens=0.60,
    )

    provider = LiteLLMProvider(model_config)

    # For free models, use a huge budget (effectively unlimited)
    effective_budget = budget if budget > 0 else 999999.0
    context = TaskContext(original_request="interactive_chat", budget_limit_usd=effective_budget)
    tracker = CostTracker(context)
    
    # -- Build the Reasoning Stack (for Smart Mode) --
    tools = ToolRegistry()
    reasoning = ReasoningCore(llm=provider)
    planner = Planner(reasoning)
    verifier = Verifier(tool_registry=tools, llm=provider)
    reflection = SelfReflection(llm=provider)
    escalation = EscalationManager(tracker, [model_config]) # Simplified for chat
    
    # Mock get_provider to always return our current provider
    # (In a real scenario, this would switch models, but for chat we stick to selection)
    escalation.get_provider = MagicMock(return_value=provider)

    executor = TaskExecutor(
        context=context,
        planner=planner,
        cost_tracker=tracker,
        escalation_manager=escalation,
        tool_registry=tools,
        verifier=verifier,
        reflection=reflection,
    )
    
    router = Router(provider)

    return provider, router, executor, planner, context, tracker


def test_connection(provider, model_name):
    """Send a tiny test message to verify the model works."""
    # For free/local models, skip strict connection test or handle gracefully
    if "ollama" in model_name.lower():
        # Skip connection test for local models to avoid cost calc issues
        return True, None

    # For paid models, try a test generation
    try:
        if HAS_RICH:
            with ui.console.status("[bold cyan]  Testing connection...", spinner="dots"):
                response, usage = provider.generate(
                    prompt="Reply with just the word 'hello'.",
                    system_prompt="Respond with only one word.",
                )
        else:
            print("  Testing connection...")
            response, usage = provider.generate(
                prompt="Reply with just the word 'hello'.",
                system_prompt="Respond with only one word.",
            )
        return True, None
    except Exception as e:
        return False, str(e)


# ─── Chat Loop ────────────────────────────────────────────────────

def chat_loop(provider, router, executor, planner, context, tracker, model_info):
    """Main conversation loop with Hybrid Routing."""
    conversation_history = []
    message_count = 0
    model_name = model_info["name"]
    is_free = model_info.get("free", False)

    print()
    ui.success("  Chat is ready! Start typing.")
    print()
    ui.info("  Type your message and press Enter.")
    ui.info("  Type /help for commands, /exit to quit.")
    print()

    system_prompt = (
        "You are a helpful AI assistant powered by CELR. "
        "Be concise, clear, and helpful. "
        "When asked to write code, provide clean, well-commented code. "
        "When asked questions, give direct, accurate answers."
    )

    while True:
        try:
            user_input = input("  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue

        # -- Commands --
        if user_input.startswith("/"):
            cmd = user_input.lower().split()[0]
            args = user_input[len(cmd):].strip()

            if cmd in ("/exit", "/quit", "/bye", "/q"):
                save_chat_log(conversation_history, model_name, context)
                break
            elif cmd == "/help":
                print()
                ui.info("  Commands:")
                ui.info("    /help          — Show this help")
                ui.info("    /cost          — Show spending")
                ui.info("    /clear         — New conversation")
                ui.info("    /model         — Current model info")
                ui.info("    /system <text> — Change AI persona")
                ui.info("    /save          — Save chat to file")
                ui.info("    /exit          — End session")
                print()
                continue
            elif cmd == "/cost":
                print()
                if is_free:
                    ui.info(f"  Model: {model_name} (FREE)")
                    ui.info(f"  Messages: {message_count}")
                else:
                    ui.cost_bar(context.current_spread_usd, context.budget_limit_usd)
                    ui.info(f"  Messages: {message_count}")
                print()
                continue
            elif cmd == "/clear":
                # Auto-save before clearing
                if conversation_history:
                    save_chat_log(conversation_history, model_name, context)
                conversation_history.clear()
                message_count = 0
                ui.success("  Conversation cleared (previous saved to logs/).")
                print()
                continue
            elif cmd == "/model":
                ui.info(f"  Model: {model_name}")
                ui.info(f"  Provider: {model_info['provider']}")
                ui.info(f"  Cost: {model_info['cost_info']}")
                ui.info(f"  System Prompt: {system_prompt}")
                print()
                continue
            elif cmd == "/system":
                if not args:
                    ui.info(f"  Current System Prompt: {system_prompt}")
                    ui.info("  Usage: /system You are a pirate")
                else:
                    system_prompt = args
                    ui.success("  System prompt updated!")
                    ui.info(f"  New persona: {system_prompt}")
                print()
                continue
            elif cmd == "/save":
                path = save_chat_log(conversation_history, model_name, context)
                ui.success(f"  Chat saved to: {path}")
                print()
                continue
            else:
                ui.warning(f"  Unknown: {cmd}. Type /help")
                continue

        # -- Budget check (cloud only) --
        if not is_free:
            remaining = context.budget_limit_usd - context.current_spread_usd
            if remaining <= 0:
                print()
                ui.error("  Budget used up! Session ended.")
                ui.info(f"  Spent: ${context.current_spread_usd:.4f}")
                break

        # -- Build prompt from history --
        conversation_history.append({"role": "user", "content": user_input})
        full_prompt = ""
        for msg in conversation_history[-10:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            full_prompt += f"{role}: {msg['content']}\n\n"

        # -- Hybrid Routing (The "Smart vs Fast" Logic) --
        try:
            start = time.time()
            
            if HAS_RICH:
                ui.console.print("[dim]  Analyzing complexity...[/dim]")
                
            # 1. Decide: Simple or Complex?
            route_type, reason = router.classify(user_input)
            
            response = ""
            usage_tokens = 0
            cost = 0.0

            if route_type == "DIRECT":
                # -- Fast Path (System 1) --
                if HAS_RICH:
                    with ui.console.status("[bold green]  Fast Mode (Direct)...[/bold green]", spinner="dots"):
                        response, usage = provider.generate(prompt=full_prompt, system_prompt=system_prompt)
                        usage_tokens = usage.total_tokens
                        if not is_free:
                            cost = provider.calculate_cost(usage)
            else:
                # -- Smart Path (System 2) --
                ui.info(f"  🧠 Complex task detected: {reason}")
                ui.info("  Activating Reasoning Engine...")
                
                # Update context with current request
                context.original_request = user_input
                
                if HAS_RICH:
                    with ui.console.status("[bold magenta]  Reasoning & Planning...[/bold magenta]", spinner="earth"):
                        # Create Plan
                        plan = planner.create_initial_plan(context)
                        # Execute Plan
                        status = executor.run(plan)
                else:
                    print("  Reasoning & Planning...")
                    plan = planner.create_initial_plan(context)
                    status = executor.run(plan)
                
                # Synthesize final answer from steps
                executed_steps = [s for s in plan.items if s.output]
                if executed_steps:
                    final_step = executed_steps[-1]
                    response = f"**Reasoning Result:**\n\n{final_step.output}"
                    
                    # Estimate usage from context (simplified)
                    usage_tokens = 1000 # Placeholder for aggregated usage
                    cost = context.current_spread_usd # Valid for tracking
                else:
                    response = "I tried to reason about that but couldn't verify a solution."

            elapsed = time.time() - start
            
            if not is_free and cost > 0:
                tracker.add_cost(cost)
            
            message_count += 1

            conversation_history.append({"role": "assistant", "content": response})
            display_cost = 0.0 if is_free else cost
            ui.ai_response(response, f"{model_name} ({route_type})", display_cost, elapsed, usage_tokens)

        except Exception as e:
            # Error handling (simplified for brevity, kept existing logic structure)
            err = str(e).lower()
            print()
            ui.error(f"  Error: {e}")
            conversation_history.pop()
            print()

    return message_count


# ─── Session Summary ──────────────────────────────────────────────

def show_summary(context, message_count, model_info):
    """End-of-session summary."""
    is_free = model_info.get("free", False)
    print()

    if HAS_RICH:
        table = Table(title="Session Summary", show_lines=True, border_style="cyan")
        table.add_column("Metric", style="bold")
        table.add_column("Value", style="cyan")
        table.add_row("Model", model_info["name"])
        table.add_row("Messages", str(message_count))
        if is_free:
            table.add_row("Cost", "FREE (local model)")
        else:
            table.add_row("Total Cost", f"${context.current_spread_usd:.4f}")
            table.add_row("Budget Left", f"${context.budget_limit_usd - context.current_spread_usd:.4f}")
            if message_count > 0:
                avg = context.current_spread_usd / message_count
                table.add_row("Avg/Message", f"${avg:.4f}")
        ui.console.print(table)
    else:
        print("  -- Session Summary --")
        print(f"  Model:    {model_info['name']}")
        print(f"  Messages: {message_count}")
        if is_free:
            print("  Cost:     FREE")
        else:
            print(f"  Cost:     ${context.current_spread_usd:.4f}")

    print()
    ui.info("  Thanks for using CELR!")
    print()


# ─── Validating & Saving ──────────────────────────────────────────

def save_chat_log(history, model_name, context):
    """Save conversation to a timestamped file."""
    if not history:
        return None

    # Ensure directory exists
    log_dir = os.path.join(os.getcwd(), "logs", "chats")
    os.makedirs(log_dir, exist_ok=True)

    # Filename: chat_2023-10-27_15-30-00.txt
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"chat_{timestamp}.txt"
    filepath = os.path.join(log_dir, filename)

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"CELR Chat Session — {timestamp}\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Budget Limit: ${context.budget_limit_usd}\n")
            f.write(f"Final Cost:   ${context.current_spread_usd}\n")
            f.write("-" * 50 + "\n\n")

            for msg in history:
                role = msg["role"].upper()
                content = msg["content"]
                f.write(f"[{role}]\n{content}\n\n")
                f.write("-" * 20 + "\n\n")
        
        return filepath
    except Exception as e:
        ui.error(f"Failed to save chat log: {e}")
        return None


# ─── Main ─────────────────────────────────────────────────────────

def main():
    ui.clear()
    ui.banner()

    # 1. Pick model
    model_info = show_model_menu()
    if model_info is None:
        sys.exit(0)

    # 2. Set budget (skipped for free models)
    budget = get_budget(model_info)

    # 3. Build engine
    print()
    ui.info(f"  Starting {model_info['name']}...")
    try:
        provider, router, executor, planner, context, tracker = create_chat_engine(model_info, budget)
    except Exception as e:
        ui.error(f"  Setup failed: {e}")
        sys.exit(1)

    # 4. Test connection
    ok, error = test_connection(provider, model_info["name"])
    if not ok:
        err = error.lower() if error else ""
        if "api_key" in err or "auth" in err or "401" in err:
            ui.error(f"  API key invalid for {model_info['name']}.")
            if model_info.get("key_env"):
                result = offer_key_entry(model_info)
                if result:
                    provider, router, executor, planner, context, tracker = create_chat_engine(result, budget)
                    model_info = result
                else:
                    sys.exit(1)
            else:
                sys.exit(1)
        elif "connection" in err or "refused" in err:
            if "ollama" in model_info["model"].lower():
                ui.error("  Ollama is not responding. Make sure 'ollama serve' is running.")
            else:
                ui.error(f"  Can't connect: {error}")
            sys.exit(1)
        else:
            ui.warning(f"  Connection test failed: {error}")
            ui.info("  Continuing anyway — the model may still work for longer prompts.")

    ui.clear()
    ui.banner()
    cost_str = f"Budget: ${budget:.2f}" if budget > 0 else "FREE"
    ui.info(f"  Model: {model_info['name']}  |  {cost_str}")

    # 5. Chat!
    message_count = chat_loop(provider, router, executor, planner, context, tracker, model_info)

    # 6. Summary
    show_summary(context, message_count, model_info)


if __name__ == "__main__":
    main()
