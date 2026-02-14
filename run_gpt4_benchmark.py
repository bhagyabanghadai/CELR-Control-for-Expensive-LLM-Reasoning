"""
Quick GPT-4 benchmark runner with per-task timeout.
Usage: python run_gpt4_benchmark.py
"""
import sys, os, json, time, signal, threading
sys.path.insert(0, os.path.dirname(__file__))
os.environ["LITELLM_LOG"] = "ERROR"

from dataclasses import asdict
from benchmarks.benchmark_runner import run_direct, run_celr, check_accuracy, BenchmarkReport, TaskResult
from benchmarks.benchmark_gpt4_tasks import GPT4_BENCHMARK_TASKS, GPT4_REFERENCE_SCORES

MODEL = "ollama/llama3.2"
BUDGET = 0.30
TASK_TIMEOUT = 120  # 2 minutes max per task


def run_with_timeout(fn, timeout, *args, **kwargs):
    """Run a function with a timeout. Returns the result or a failed TaskResult."""
    result = [None]
    error = [None]

    def wrapper():
        try:
            result[0] = fn(*args, **kwargs)
        except Exception as e:
            error[0] = str(e)

    thread = threading.Thread(target=wrapper)
    thread.daemon = True
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        # Timeout — return a failed result
        return None, "TIMEOUT"
    if error[0]:
        return None, error[0]
    return result[0], None


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run GPT-4 Benchmark with Timeout")
    parser.add_argument("--category", help="Filter by category (e.g. math, humaneval)")
    parser.add_argument("--task-id", help="Filter by specific task ID")
    args = parser.parse_args()

    tasks = GPT4_BENCHMARK_TASKS
    if args.category:
        tasks = [t for t in tasks if t["category"] == args.category]
    if args.task_id:
        tasks = [t for t in tasks if t["id"] == args.task_id]

    if not tasks:
        print("No tasks found matching criteria.")
        return
    report = BenchmarkReport(
        model=MODEL, budget=BUDGET,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )

    print(f"\n{'='*60}")
    print(f"  CELR GPT-4 BENCHMARK")
    print(f"  Model: {MODEL}  |  Budget: ${BUDGET:.2f}")
    print(f"  Tasks: {len(tasks)}  |  Timeout: {TASK_TIMEOUT}s/task")
    print(f"{'='*60}\n")

    for i, task in enumerate(tasks, 1):
        cat = task.get("category", "unknown")
        tid = task["id"]
        print(f"[{i:2d}/{len(tasks)}] {tid:<24} ({cat})")

        # === Direct LLM ===
        sys.stdout.write(f"         Direct... ")
        sys.stdout.flush()
        start = time.time()
        dr, err = run_with_timeout(run_direct, TASK_TIMEOUT, task, MODEL)
        if dr is None:
            dr = TaskResult(task_id=tid, method="direct", prompt=task["prompt"], error=err or "timeout")
        report.direct_results.append(dr)
        sym = "✅" if dr.accuracy else "❌"
        print(f"{sym}  {dr.latency_s:.1f}s")

        # === CELR Pipeline ===
        sys.stdout.write(f"         CELR...   ")
        sys.stdout.flush()
        cr, err = run_with_timeout(run_celr, TASK_TIMEOUT, task, MODEL, BUDGET)
        if cr is None:
            cr = TaskResult(task_id=tid, method="celr", prompt=task["prompt"], error=err or "timeout")
        report.celr_results.append(cr)
        sym = "✅" if cr.accuracy else "❌"
        print(f"{sym}  {cr.latency_s:.1f}s")
        print()

    # ── Summary ──
    print(report.summary(tasks=tasks))

    # ── Save ──
    os.makedirs("benchmarks/results", exist_ok=True)
    fname = f"benchmarks/results/benchmark_gpt4_{int(time.time())}.json"
    with open(fname, "w") as f:
        json.dump(asdict(report), f, indent=2, default=str)
    print(f"\n📁 Results saved to {fname}")


if __name__ == "__main__":
    main()
