"""
CELR CLI — Command-line interface for CELR agent.

Usage:
    celr "Write a Python function to calculate fibonacci"
    celr "Build a REST API" --budget 1.0 --verbose
    celr "Solve this math problem" --small-model ollama/phi3
"""

import argparse
import logging
import sys

from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from celr.core.config import CELRConfig
from celr.core.cost_tracker import CostTracker
from celr.core.escalation import EscalationManager
from celr.core.executor import TaskExecutor
from celr.core.llm import LiteLLMProvider
from celr.core.logger import TrajectoryLogger
from celr.core.planner import Planner
from celr.core.reasoning import ReasoningCore
from celr.core.reflection import SelfReflection
from celr.core.tools import ToolRegistry
from celr.core.types import TaskContext
from celr.core.verifier import Verifier
from celr.training.self_reward import SelfRewardScorer  # Phase 10: Online TRM

logger = logging.getLogger(__name__)
console = Console()


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="CELR — Control for Expensive LLM Reasoning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    celr "What is 2+2?"
    celr "Build a REST API" --budget 1.0 --verbose
    celr "Summarize this text" --small-model ollama/phi3
        """,
    )
    parser.add_argument("task", help="The task to perform")
    parser.add_argument("--budget", type=float, default=None, help="Budget limit in USD")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument("--small-model", type=str, default=None, help="Override small model name")
    parser.add_argument("--large-model", type=str, default=None, help="Override large model name")
    parser.add_argument("--ui", action="store_true", help="Launch Cerebro war room dashboard")
    args = parser.parse_args()

    # 1. Load Config (env vars → .env file → CLI args override)
    config_overrides = {}
    if args.budget is not None:
        config_overrides["budget_limit"] = args.budget
    if args.verbose:
        config_overrides["verbose"] = True
    if args.small_model:
        config_overrides["small_model"] = args.small_model
    if args.large_model:
        config_overrides["large_model"] = args.large_model

    config = CELRConfig(**config_overrides)
    config.setup_logging()

    # Phase 9: Launch Cerebro dashboard in background if --ui
    if getattr(args, "ui", False):
        import subprocess, sys as _sys
        dashboard_path = Path(__file__).parent / "interface" / "dashboard.py"
        subprocess.Popen(
            [_sys.executable, "-m", "streamlit", "run", str(dashboard_path),
             "--server.headless", "true"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        console.print("[bold magenta]🧠 Cerebro launched → http://localhost:8501[/bold magenta]")

    # 2. Rich header
    console.print(Panel(
        f"[bold cyan]CELR[/bold cyan] — Control for Expensive LLM Reasoning\n"
        f"[dim]Budget: ${config.budget_limit:.2f} | Models: {config.small_model} → {config.large_model}[/dim]",
        border_style="cyan",
    ))

    # 3. Build components
    context = TaskContext(
        original_request=args.task,
        budget_limit_usd=config.budget_limit,
    )

    cost_tracker = CostTracker(context)
    trajectory_logger = TrajectoryLogger(log_dir=config.log_dir)

    # Reasoning core uses the small model for planning
    reasoning_llm = LiteLLMProvider(config.get_model_tiers()[0])
    reasoning_core = ReasoningCore(reasoning_llm)
    planner = Planner(reasoning_core)

    # Escalation manager with all tiers
    escalation = EscalationManager(
        cost_tracker=cost_tracker,
        model_tiers=config.get_model_tiers(),
    )

    # Verifier uses the cheapest available model
    verifier_llm = LiteLLMProvider(config.get_model_tiers()[0])
    tool_registry = ToolRegistry()
    verifier = Verifier(tool_registry=tool_registry, llm=verifier_llm)

    # Reflection uses the same cheap model
    reflection = SelfReflection(llm=verifier_llm)

    # Phase 10: Online Self-Reward Scorer (Samsung TRM)
    # Uses the small model for fast, recursive self-grading
    self_reward_scorer = SelfRewardScorer(llm=verifier_llm)

    # Build the executor with ALL components wired
    executor = TaskExecutor(
        context=context,
        planner=planner,
        cost_tracker=cost_tracker,
        escalation_manager=escalation,
        tool_registry=tool_registry,
        verifier=verifier,
        reflection=reflection,
        trajectory_logger=trajectory_logger,
        self_reward_scorer=self_reward_scorer,
    )

    # 4. Execute
    console.print(f"\n[bold]Task:[/bold] {args.task}\n")

    plan = None
    try:
        with console.status("[bold green]Planning...", spinner="dots"):
            plan = planner.create_initial_plan(context)

        console.print(f"[green]✓[/green] Plan created with [bold]{len(plan.items)}[/bold] steps\n")

        # Show plan
        plan_table = Table(title="Execution Plan", show_lines=True)
        plan_table.add_column("#", style="dim", width=3)
        plan_table.add_column("Description", style="cyan")
        plan_table.add_column("Type", style="magenta")
        plan_table.add_column("Difficulty", style="yellow")
        for i, step in enumerate(plan.items, 1):
            plan_table.add_row(
                str(i),
                step.description[:60],
                step.step_type.value,
                f"{step.estimated_difficulty:.1f}",
            )
        console.print(plan_table)
        console.print()

        # Execute
        with console.status("[bold green]Executing...", spinner="dots"):
            final_status = executor.run(plan)

    except KeyboardInterrupt:
        final_status = "INTERRUPTED"
        console.print("\n[yellow]Interrupted by user.[/yellow]")
    except Exception as e:
        final_status = "ERROR"
        console.print(f"\n[red]Error: {e}[/red]")
        logger.exception("Execution failed")

    # 5. Results
    status_colors = {
        "SUCCESS": "green",
        "FAILED": "red",
        "STUCK": "yellow",
        "ERROR": "red",
        "INTERRUPTED": "yellow",
    }
    color = status_colors.get(final_status, "white")

    result_table = Table(title="Results")
    result_table.add_column("Metric", style="bold")
    result_table.add_column("Value")
    result_table.add_row("Status", f"[{color}]{final_status}[/{color}]")
    result_table.add_row("Total Cost", f"${context.current_spread_usd:.6f}")
    result_table.add_row("Budget Remaining", f"${context.budget_remaining:.6f}")
    result_table.add_row("Steps", str(len(plan.items)) if plan else "N/A")
    console.print(result_table)

    # 6. Save trajectory
    if plan:
        try:
            trajectory_logger.save_trajectory(context, plan, final_status)
            console.print(f"\n[dim]Trajectory saved to {config.log_dir}/[/dim]")
        except Exception as e:
            logger.warning(f"Failed to save trajectory: {e}")

    # Exit code
    sys.exit(0 if final_status == "SUCCESS" else 1)


if __name__ == "__main__":
    main()
