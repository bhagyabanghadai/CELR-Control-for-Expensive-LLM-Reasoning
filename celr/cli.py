import argparse
import sys
import os
import dotenv
from rich.console import Console
from rich.panel import Panel

from celr.core.types import TaskContext, ModelConfig
from celr.core.cost_tracker import CostTracker
from celr.core.llm import LiteLLMProvider
from celr.core.reasoning import ReasoningCore
from celr.core.planner import Planner
from celr.core.executor import TaskExecutor
from celr.core.logger import TrajectoryLogger

# Load env vars
dotenv.load_dotenv()

console = Console()

def main():
    parser = argparse.ArgumentParser(description="CELR: Control for Expensive LLM Reasoning")
    parser.add_argument("task", type=str, help="The task you want the agent to solve.")
    parser.add_argument("--budget", type=float, default=0.50, help="Max budget in USD (default: $0.50)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--small-model", type=str, default="ollama/llama3", help="Name of small model")
    
    args = parser.parse_args()
    
    console.print(Panel(f"[bold green]🧠 CELR is thinking...[/bold green]\nTarget: {args.task}\nBudget: ${args.budget:.2f}"))
    
    # 1. Initialize Context & Budget
    ctx = TaskContext(
        original_request=args.task,
        budget_limit_usd=args.budget
    )
    tracker = CostTracker(ctx)
    logger = TrajectoryLogger()
    
    # 2. Setup "Small Brain" (Reasoning Core)
    # In production, we'd load this from .env or args
    small_model_config = ModelConfig(
        name=args.small_model,
        provider="ollama" if "ollama" in args.small_model else "openai",
        cost_per_million_input_tokens=0.0, # Assuming local/free for now
        cost_per_million_output_tokens=0.0
    )
    llm_provider = LiteLLMProvider(small_model_config)
    reasoning_core = ReasoningCore(llm_provider)
    
    # 3. Plan
    planner = Planner(reasoning_core)
    console.print("[yellow]Phase 1: Recursive Planning & Decomposition...[/yellow]")
    try:
        plan = planner.create_initial_plan(ctx)
        console.print(f"[bold]Plan Generated:[/bold] {len(plan.items)} steps.")
        for step in plan.items:
            console.print(f" - [{step.id}] {step.description} (Diff: {step.estimated_difficulty})")
    except Exception as e:
        console.print(f"[bold red]Planning Failed:[/bold red] {e}")
        sys.exit(1)

    # 4. Execute
    console.print("\n[yellow]Phase 2: Execution & Escalation Loop...[/yellow]")
    executor = TaskExecutor(ctx, planner, tracker)
    final_status = executor.run(plan)
    
    # 5. Report
    console.print(f"\n[bold]Final Status:[/bold] {final_status}")
    console.print(f"[bold]Total Cost:[/bold] ${ctx.current_spread_usd:.4f} / ${ctx.budget_limit_usd:.2f}")
    
    # 6. Save Trajectory
    logger.save_trajectory(ctx, plan, final_status)
    print("\nTrajectory log saved for future training.")

if __name__ == "__main__":
    main()
