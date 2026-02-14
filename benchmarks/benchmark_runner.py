"""
CELR Benchmark Runner — Compare CELR vs Direct LLM Calls.

Runs standardized tasks through:
  1. CELR pipeline (plan → route → execute → verify)
  2. Direct LLM call (single LLM call)

Measures: accuracy, cost, latency, escalation count.

Supports two task suites:
  - standard: 12 general tasks (easy/medium/hard)
  - gpt4:     20 GPT-4-level tasks (MMLU, HumanEval, GSM8K, ARC, MATH)

Usage:
    python -m benchmarks.benchmark_runner --model ollama/llama3.2
    python -m benchmarks.benchmark_runner --model ollama/llama3.2 --suite gpt4
    python -m benchmarks.benchmark_runner --model ollama/llama3.2 --suite gpt4 --budget 0.30
"""

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional
from unittest.mock import patch

from celr.core.config import CELRConfig
from celr.core.types import TaskContext, TaskStatus, ModelConfig
from celr.core.cost_tracker import CostTracker
from celr.core.reasoning import ReasoningCore
from celr.core.planner import Planner
from celr.core.escalation import EscalationManager
from celr.core.tools import ToolRegistry
from celr.core.verifier import Verifier
from celr.core.reflection import SelfReflection
from celr.core.executor import TaskExecutor
from celr.core.llm import LiteLLMProvider, LLMUsage

from benchmarks.benchmark_tasks import BENCHMARK_TASKS
from benchmarks.benchmark_gpt4_tasks import GPT4_BENCHMARK_TASKS, GPT4_REFERENCE_SCORES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark")


@dataclass
class TaskResult:
    """Result of running a single benchmark task."""
    task_id: str
    method: str  # "celr" or "direct"
    prompt: str
    output: str = ""
    success: bool = False
    accuracy: bool = False  # Whether expected keywords found
    cost_usd: float = 0.0
    latency_s: float = 0.0
    escalation_count: int = 0
    steps_executed: int = 0
    retries: int = 0
    error: str = ""


@dataclass
class BenchmarkReport:
    """Full benchmark report comparing CELR vs Direct."""
    model: str
    budget: float
    timestamp: str = ""
    celr_results: List[TaskResult] = field(default_factory=list)
    direct_results: List[TaskResult] = field(default_factory=list)

    def summary(self, tasks=None) -> str:
        """Generate a comparison table with optional GPT-4 reference scores."""
        tasks = tasks or BENCHMARK_TASKS
        total_tasks = len(tasks)

        lines = [
            f"\n{'='*80}",
            "CELR BENCHMARK REPORT",
            f"{'='*80}",
            f"Model: {self.model}  |  Budget: ${self.budget:.2f}",
            f"Tasks: {total_tasks}  |  Time: {self.timestamp}",
            f"\n{'─'*80}",
            f"{'Method':<10} {'Accuracy':<12} {'Avg Cost':<14} {'Avg Latency':<14} {'Escalations':<12}",
            f"{'─'*80}",
        ]

        for label, results in [("CELR", self.celr_results), ("Direct", self.direct_results)]:
            if not results:
                continue
            acc = sum(1 for r in results if r.accuracy) / len(results) * 100
            avg_cost = sum(r.cost_usd for r in results) / len(results)
            avg_lat = sum(r.latency_s for r in results) / len(results)
            esc = sum(r.escalation_count for r in results)
            lines.append(f"{label:<10} {acc:>6.1f}%     ${avg_cost:>10.6f}   {avg_lat:>8.2f}s       {esc}")

        lines.append(f"{'─'*80}")

        # CELR boost metric
        if self.celr_results and self.direct_results:
            celr_acc = sum(1 for r in self.celr_results if r.accuracy) / len(self.celr_results) * 100
            direct_acc = sum(1 for r in self.direct_results if r.accuracy) / len(self.direct_results) * 100
            boost = celr_acc - direct_acc
            lines.append(f"\n🚀 CELR Boost: {'+' if boost >= 0 else ''}{boost:.1f}% (CELR {celr_acc:.1f}% vs Direct {direct_acc:.1f}%)")

        # Per-category breakdown with GPT-4 reference scores
        categories = {}
        for task in tasks:
            cat = task.get("category", "general")
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(task["id"])

        if len(categories) > 1:
            lines.append(f"\n{'='*80}")
            lines.append("PER-CATEGORY ACCURACY:")
            lines.append(f"{'─'*80}")
            lines.append(f"{'Category':<16} {'Direct':<12} {'CELR':<12} {'CELR Boost':<12} {'GPT-4 Ref':<12}")
            lines.append(f"{'─'*80}")

            for cat, task_ids in categories.items():
                celr_cat = [r for r in self.celr_results if r.task_id in task_ids]
                direct_cat = [r for r in self.direct_results if r.task_id in task_ids]

                celr_pct = sum(1 for r in celr_cat if r.accuracy) / len(celr_cat) * 100 if celr_cat else 0
                direct_pct = sum(1 for r in direct_cat if r.accuracy) / len(direct_cat) * 100 if direct_cat else 0
                boost = celr_pct - direct_pct
                gpt4_ref = GPT4_REFERENCE_SCORES.get(cat, 0.0)
                gpt4_str = f"{gpt4_ref:.1f}%" if gpt4_ref > 0 else "N/A"

                lines.append(f"{cat.upper():<16} {direct_pct:>5.0f}%       {celr_pct:>5.0f}%       {'+' if boost >= 0 else ''}{boost:>5.0f}%       {gpt4_str:>6}")

        lines.append(f"{'─'*80}")

        # Cost savings
        if self.celr_results and self.direct_results:
            celr_total = sum(r.cost_usd for r in self.celr_results)
            direct_total = sum(r.cost_usd for r in self.direct_results)
            if direct_total > 0:
                saving_pct = (1 - celr_total / direct_total) * 100
                lines.append(f"\nTotal CELR cost:   ${celr_total:.6f}")
                lines.append(f"Total Direct cost: ${direct_total:.6f}")
                if saving_pct > 0:
                    lines.append(f"💰 Cost savings:   {saving_pct:.1f}%")
                else:
                    lines.append(f"⚠️  CELR overhead:  {abs(saving_pct):.1f}% more expensive")
            else:
                lines.append(f"\n💰 Local model — $0 cost (Ollama)")

        lines.append(f"\n{'='*80}\n")

        # Per-task detail
        lines.append("DETAILED RESULTS:")
        lines.append(f"{'─'*80}")
        for task in tasks:
            celr_r = next((r for r in self.celr_results if r.task_id == task["id"]), None)
            direct_r = next((r for r in self.direct_results if r.task_id == task["id"]), None)

            cat = task.get('category', task.get('difficulty', 'general'))
            lines.append(f"\n📋 {task['id']} ({cat})")
            lines.append(f"   Prompt: {task['prompt'][:70]}...")
            if celr_r:
                lines.append(f"   CELR:   {'✅' if celr_r.accuracy else '❌'} ${celr_r.cost_usd:.6f} {celr_r.latency_s:.2f}s (steps={celr_r.steps_executed}, retries={celr_r.retries})")
            if direct_r:
                lines.append(f"   Direct: {'✅' if direct_r.accuracy else '❌'} ${direct_r.cost_usd:.6f} {direct_r.latency_s:.2f}s")

        return "\n".join(lines)


def check_accuracy(output: str, expected_contains: List[str]) -> bool:
    """Check if output contains expected keywords."""
    output_lower = output.lower()
    return all(kw.lower() in output_lower for kw in expected_contains)


def create_config(model_name: str) -> ModelConfig:
    """Create a default ModelConfig for the benchmark."""
    is_ollama = model_name.startswith("ollama") or model_name.startswith("local")
    provider = "ollama" if is_ollama else "openai"
    
    # Use 0 cost for local, standard rates for others (fallback)
    cost_in = 0.0 if is_ollama else 0.15 # gpt-4o-mini approx
    cost_out = 0.0 if is_ollama else 0.60
    
    return ModelConfig(
        name=model_name,
        provider=provider,
        cost_per_million_input_tokens=cost_in,
        cost_per_million_output_tokens=cost_out,
    )


def run_direct(task: dict, model: str) -> TaskResult:
    """Run task with a DIRECT LLM call — no CELR overhead."""
    result = TaskResult(task_id=task["id"], method="direct", prompt=task["prompt"])

    try:
        config = create_config(model)
        provider = LiteLLMProvider(config=config)
        start = time.time()
        response, usage = provider.generate(task["prompt"])
        result.latency_s = time.time() - start
        result.output = response
        result.cost_usd = provider.calculate_cost(usage)
        result.success = True
        result.accuracy = check_accuracy(response, task["expected_contains"])
    except Exception as e:
        result.error = str(e)
        logger.error(f"Direct call failed for {task['id']}: {e}")

    return result


def run_celr(task: dict, model: str, budget: float) -> TaskResult:
    """Run task through the CELR pipeline."""
    result = TaskResult(task_id=task["id"], method="celr", prompt=task["prompt"])

    try:
        config = CELRConfig(
            budget_limit=min(budget, task.get("max_budget", budget)),
            small_model=model,
            mid_model=model,
            large_model=model,
        )

        context = TaskContext(
            original_request=task["prompt"],
            budget_limit_usd=config.budget_limit,
        )

        llm_config = create_config(model)
        provider = LiteLLMProvider(config=llm_config)
        tracker = CostTracker(context)
        escalation = EscalationManager(tracker, config.get_model_tiers())
        tools = ToolRegistry()
        reasoning = ReasoningCore(llm=provider)
        planner = Planner(reasoning)
        verifier = Verifier(tool_registry=tools, llm=provider)
        reflection = SelfReflection(llm=provider)

        with patch.object(escalation, "get_provider", return_value=provider):
            executor = TaskExecutor(
                context=context,
                planner=planner,
                cost_tracker=tracker,
                escalation_manager=escalation,
                tool_registry=tools,
                verifier=verifier,
                reflection=reflection,
            )

            start = time.time()
            plan = planner.create_initial_plan(context)
            status = executor.run(plan)
            result.latency_s = time.time() - start

            # Collect metrics
            result.cost_usd = context.current_spread_usd
            result.steps_executed = len(plan.items)
            result.retries = sum(s.retry_count for s in plan.items)
            result.success = status == "SUCCESS"

            # Combine all step outputs for accuracy check
            combined_output = "\n".join(
                s.output or "" for s in plan.items if s.output
            )
            result.output = combined_output
            result.accuracy = check_accuracy(combined_output, task["expected_contains"])

    except Exception as e:
        result.error = str(e)
        logger.error(f"CELR pipeline failed for {task['id']}: {e}")

    return result


def run_benchmark(model: str = "gpt-4o-mini", budget: float = 0.50, tasks=None, suite: str = "standard"):
    """Run the full benchmark suite."""
    if tasks is None:
        tasks = GPT4_BENCHMARK_TASKS if suite == "gpt4" else BENCHMARK_TASKS

    report = BenchmarkReport(
        model=model,
        budget=budget,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )

    suite_label = "GPT-4 Level" if suite == "gpt4" else "Standard"
    logger.info(f"Starting {suite_label} benchmark: {len(tasks)} tasks, model={model}, budget=${budget:.2f}")

    for i, task in enumerate(tasks, 1):
        cat = task.get('category', task.get('difficulty', 'unknown'))
        logger.info(f"\n[{i}/{len(tasks)}] {task['id']} ({cat})")

        # 1. Direct LLM call
        logger.info(f"  Running DIRECT...")
        direct_result = run_direct(task, model)
        report.direct_results.append(direct_result)
        logger.info(f"  Direct: {'✅' if direct_result.accuracy else '❌'} ${direct_result.cost_usd:.6f} ({direct_result.latency_s:.1f}s)")

        # 2. CELR pipeline
        logger.info(f"  Running CELR...")
        celr_result = run_celr(task, model, budget)
        report.celr_results.append(celr_result)
        logger.info(f"  CELR:   {'✅' if celr_result.accuracy else '❌'} ${celr_result.cost_usd:.6f} ({celr_result.latency_s:.1f}s)")

    # Print summary
    print(report.summary(tasks=tasks))

    # Save results
    os.makedirs("benchmarks/results", exist_ok=True)
    results_file = f"benchmarks/results/benchmark_{suite}_{int(time.time())}.json"
    with open(results_file, "w") as f:
        json.dump(asdict(report), f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

    return report


def main():
    parser = argparse.ArgumentParser(description="CELR Benchmark Runner")
    parser.add_argument("--model", default="ollama/llama3.2", help="LLM model to test")
    parser.add_argument("--budget", type=float, default=0.50, help="Budget per task")
    parser.add_argument("--suite", choices=["standard", "gpt4"], default="standard", help="Task suite: standard (12 tasks) or gpt4 (20 GPT-4-level tasks)")
    parser.add_argument("--category", help="Filter by category (e.g. mmlu, humaneval, gsm8k, arc, math)")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], help="Filter by difficulty (standard suite only)")
    parser.add_argument("--dry-run", action="store_true", help="Print tasks without running")
    args = parser.parse_args()

    if args.suite == "gpt4":
        tasks = GPT4_BENCHMARK_TASKS
        if args.category:
            tasks = [t for t in tasks if t["category"] == args.category]
    else:
        tasks = BENCHMARK_TASKS
        if args.difficulty:
            tasks = [t for t in tasks if t["difficulty"] == args.difficulty]

    if args.dry_run:
        suite_label = "GPT-4 Level" if args.suite == "gpt4" else "Standard"
        print(f"\n{'='*60}")
        print(f"CELR {suite_label} BENCHMARK TASKS ({len(tasks)} tasks)")
        print(f"{'='*60}")
        for t in tasks:
            cat = t.get('category', t.get('difficulty', 'general'))
            print(f"\n  [{cat:>10}] {t['id']}")
            print(f"             {t['prompt'][:60]}...")
        if args.suite == "gpt4":
            print(f"\nGPT-4 Reference Scores:")
            for cat, score in GPT4_REFERENCE_SCORES.items():
                print(f"  {cat.upper():<12} {score:.1f}%")
        return

    run_benchmark(model=args.model, budget=args.budget, tasks=tasks, suite=args.suite)


if __name__ == "__main__":
    main()
