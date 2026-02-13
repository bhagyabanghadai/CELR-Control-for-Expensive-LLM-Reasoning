"""
CELR Live Simulation — Full Feature Test
==========================================
Tests EVERY major feature in a real scenario flow using MockLLM.

Features tested:
  [1] Configuration & Setup
  [2] Task Decomposition (Planner)
  [3] Cost Tracking & Budget Enforcement
  [4] Tool Execution (Python REPL)
  [5] Verification (Heuristic + LLM-as-Judge)
  [6] Reflection & Smart Retry
  [7] Escalation Manager (Model Routing)
  [8] Adaptive Cortex (State Extraction + Policy)
  [9] Full Executor Pipeline (End-to-End)
  [10] Training Pipeline (Scorer + Data Generator)
  [11] Self-Reward Scorer
  [12] Demo Script Compatibility

Usage:
    python live_simulation.py
"""

import json
import os
import sys
import time
import tempfile
import shutil
import traceback
from unittest.mock import patch, MagicMock

# Fix Windows console encoding for emoji output
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ─── Test Infrastructure ───────────────────────────────────────────

PASS = 0
FAIL = 0
TESTS = []


def test(name):
    """Decorator to register a test."""
    def decorator(func):
        TESTS.append((name, func))
        return func
    return decorator


def run_all():
    global PASS, FAIL
    print(f"\n{'='*70}")
    print("  CELR LIVE SIMULATION — Full Feature Test")
    print(f"{'='*70}\n")

    for name, func in TESTS:
        try:
            func()
            PASS += 1
            print(f"  ✅ {name}")
        except Exception as e:
            FAIL += 1
            print(f"  ❌ {name}")
            print(f"     Error: {e}")
            traceback.print_exc()

    total = PASS + FAIL
    print(f"\n{'─'*70}")
    print(f"  Results: {PASS}/{total} passed, {FAIL} failed")
    if FAIL == 0:
        print("  🎉 ALL FEATURES WORKING!")
    else:
        print(f"  ⚠️  {FAIL} feature(s) need attention")
    print(f"{'='*70}\n")
    return FAIL == 0


# ─── Mock LLM Provider ────────────────────────────────────────────

from celr.core.llm import BaseLLMProvider, LLMUsage


class SimulationLLM(BaseLLMProvider):
    """Smart mock that returns context-aware responses."""

    def __init__(self):
        self.call_count = 0
        self.total_tokens = 0

    def generate(self, prompt, system_prompt=None, tools=None):
        self.call_count += 1
        tokens = 100

        # Planning / Decomposition
        if "Original Goal:" in prompt and "Context:" in prompt:
            response = json.dumps({
                "original_goal": "Test simulation task",
                "items": [
                    {"id": "s1", "description": "Calculate 2+2 using Python", "estimated_difficulty": 0.2, "dependencies": []},
                    {"id": "s2", "description": "Verify the calculation is correct", "estimated_difficulty": 0.3, "dependencies": ["s1"]},
                    {"id": "s3", "description": "Write a summary of the result", "estimated_difficulty": 0.1, "dependencies": ["s2"]},
                ]
            })
            tokens = 200

        # Difficulty Estimation
        elif "Estimate the difficulty" in prompt:
            response = json.dumps({"difficulty_score": 0.3, "reasoning": "Simple computation"})
            tokens = 50

        # Self-reward scoring (must be before generic code/reasoning checks)
        elif "Scoring Instructions" in prompt or ("Accuracy" in prompt and "Completeness" in prompt and "Efficiency" in prompt):
            response = "Accuracy: 4\nCompleteness: 5\nEfficiency: 4\nTotal: 13"
            tokens = 30

        # Verification
        elif "VERIFICATION" in prompt.upper() or "verdict" in prompt.lower():
            response = "VERDICT: YES\nCONFIDENCE: 0.92\nREASON: The output is correct."
            tokens = 40

        # Reflection / Failure Analysis
        elif "FAILURE ANALYSIS" in prompt:
            response = "The step failed because of a syntax error. Fix: correct the indentation."
            tokens = 60

        # Code generation
        elif "Calculate" in prompt or "compute" in prompt.lower():
            response = "```python\nresult = 2 + 2\nprint(f'Result: {result}')\n```"
            tokens = 50

        # Default
        else:
            response = "The result is 4. The calculation 2+2=4 is correct."
            tokens = 30

        self.total_tokens += tokens
        usage = LLMUsage(prompt_tokens=tokens // 2, completion_tokens=tokens // 2, total_tokens=tokens)
        return response, usage

    def calculate_cost(self, usage):
        return (usage.prompt_tokens * 0.15 + usage.completion_tokens * 0.60) / 1_000_000


# ─── Feature Tests ─────────────────────────────────────────────────

@test("[1] Configuration & Setup")
def test_config():
    from celr.core.config import CELRConfig
    config = CELRConfig(budget_limit=1.00, verbose=True, small_model="gpt-4o-mini")
    assert config.budget_limit == 1.00
    assert config.small_model == "gpt-4o-mini"
    
    tiers = config.get_model_tiers()
    assert len(tiers) >= 3, f"Expected 3+ tiers, got {len(tiers)}"
    
    # Test provider inference
    assert config._infer_provider("gpt-4o") == "openai"
    assert config._infer_provider("claude-3") == "anthropic"
    assert config._infer_provider("ollama/llama3") == "ollama"
    assert config._infer_provider("groq/mixtral") == "groq"
    assert config._infer_provider("deepseek/coder") == "deepseek"
    assert config._infer_provider("o3-mini") == "openai"


@test("[2] Task Decomposition (Planner)")
def test_planner():
    from celr.core.reasoning import ReasoningCore
    from celr.core.planner import Planner
    from celr.core.types import TaskContext

    llm = SimulationLLM()
    reasoning = ReasoningCore(llm=llm)
    planner = Planner(reasoning)
    context = TaskContext(original_request="Calculate 2+2", budget_limit_usd=1.0)

    plan = planner.create_initial_plan(context)
    assert plan is not None
    assert len(plan.items) == 3, f"Expected 3 steps, got {len(plan.items)}"
    assert plan.items[0].id == "s1"
    assert plan.items[1].dependencies == ["s1"]
    assert plan.items[2].dependencies == ["s2"]

    # Test DAG-based ready step detection
    ready = planner.get_ready_steps(plan)
    ready_ids = [s.id for s in ready]
    assert "s1" in ready_ids, "Step s1 should be ready (no dependencies)"
    assert "s2" not in ready_ids, "Step s2 should NOT be ready (depends on s1)"


@test("[3] Cost Tracking & Budget Enforcement")
def test_cost_tracking():
    from celr.core.types import TaskContext
    from celr.core.cost_tracker import CostTracker
    from celr.core.exceptions import BudgetExhaustedError

    context = TaskContext(original_request="Test", budget_limit_usd=0.10)
    tracker = CostTracker(context)

    # Add some cost
    tracker.add_cost(0.03)
    assert abs(context.current_spread_usd - 0.03) < 1e-6

    # Check affordability
    assert tracker.can_afford(0.05) is True
    assert tracker.can_afford(0.10) is False

    # Add more cost
    tracker.add_cost(0.05)
    assert abs(context.current_spread_usd - 0.08) < 1e-6
    assert tracker.can_afford(0.05) is False  # Only $0.02 left


@test("[4] Tool Execution (Python REPL)")
def test_tools():
    from celr.core.tools import ToolRegistry
    from celr.core.exceptions import ToolExecutionError

    registry = ToolRegistry()
    tools = registry.list_tools()
    assert "python_repl" in tools
    assert "shell_exec" in tools

    # Execute safe code
    result = registry.execute("python_repl", code="print(2 + 2)")
    assert "4" in result

    # Execute math
    result = registry.execute("python_repl", code="print(sum(range(10)))")
    assert "45" in result

    # Block unsafe code
    try:
        registry.execute("python_repl", code="import os")
        assert False, "Should have raised ToolExecutionError"
    except ToolExecutionError:
        pass  # Expected

    # Block file access
    try:
        registry.execute("python_repl", code="open('hack.txt', 'w')")
        assert False, "Should have raised ToolExecutionError"
    except ToolExecutionError:
        pass  # Expected


@test("[5] Verification (Heuristic + LLM-as-Judge)")
def test_verification():
    from celr.core.verifier import Verifier
    from celr.core.tools import ToolRegistry
    from celr.core.types import Step, StepType, TaskContext

    llm = SimulationLLM()
    verifier = Verifier(tool_registry=ToolRegistry(), llm=llm)
    context = TaskContext(original_request="Test", budget_limit_usd=1.0)

    # Execution step — clean output passes
    step1 = Step(id="s1", description="Run code", step_type=StepType.EXECUTION, output="Result: 4")
    assert verifier.verify(step1, context) is True

    # Execution step — Python error fails
    step2 = Step(id="s2", description="Run code", step_type=StepType.EXECUTION, output="NameError: name 'x' is not defined")
    assert verifier.verify(step2, context) is False

    # Empty output fails
    step3 = Step(id="s3", description="Run code", step_type=StepType.EXECUTION, output="")
    assert verifier.verify(step3, context) is False

    # Reasoning step — LLM says YES
    step4 = Step(id="s4", description="Explain result", step_type=StepType.REASONING, output="The answer is 4.")
    assert verifier.verify(step4, context) is True
    assert "0.92" in step4.verification_notes  # Confidence score extracted

    # Verify cost tracking
    assert verifier.total_verification_cost > 0


@test("[6] Reflection & Smart Retry")
def test_reflection():
    from celr.core.reflection import SelfReflection
    from celr.core.types import Step, TaskContext

    llm = SimulationLLM()
    reflection = SelfReflection(llm=llm, max_retries=3)
    context = TaskContext(original_request="Test", budget_limit_usd=1.0)

    # Analyze failure
    step = Step(id="s1", description="Parse data", verification_notes="JSON parse error")
    analysis = reflection.analyze_failure(step, context)
    assert len(analysis) > 0
    assert reflection.total_reflection_cost > 0

    # Smart retry: under max
    assert reflection.should_retry(step, attempt_count=0) is True
    assert reflection.should_retry(step, attempt_count=2) is True

    # Smart retry: at max
    assert reflection.should_retry(step, attempt_count=3) is False

    # Never retry on budget exhaustion
    budget_step = Step(id="s2", description="X", verification_notes="budget exhausted")
    assert reflection.should_retry(budget_step, attempt_count=0) is False

    # Always retry on rate limits
    rate_step = Step(id="s3", description="X", verification_notes="rate limit exceeded")
    assert reflection.should_retry(rate_step, attempt_count=0) is True


@test("[7] Escalation Manager (Model Routing)")
def test_escalation():
    from celr.core.types import TaskContext, ModelConfig, Step
    from celr.core.cost_tracker import CostTracker
    from celr.core.escalation import EscalationManager

    context = TaskContext(original_request="Test", budget_limit_usd=1.0)
    tracker = CostTracker(context)
    tiers = [
        ModelConfig(name="small", provider="mock", cost_per_million_input_tokens=0.1, cost_per_million_output_tokens=0.2),
        ModelConfig(name="mid", provider="mock", cost_per_million_input_tokens=1.0, cost_per_million_output_tokens=2.0),
        ModelConfig(name="large", provider="mock", cost_per_million_input_tokens=10.0, cost_per_million_output_tokens=30.0),
    ]

    escalation = EscalationManager(cost_tracker=tracker, model_tiers=tiers)

    # Easy step → small model
    easy_step = Step(id="s1", description="Simple", estimated_difficulty=0.1)
    model = escalation.select_model(easy_step)
    assert "small" in model.lower(), f"Easy step routed to {model}, expected small"

    # Hard step → larger model
    hard_step = Step(id="s2", description="Complex", estimated_difficulty=0.9)
    model = escalation.select_model(hard_step)
    assert "small" not in model.lower(), f"Hard step should NOT route to small, got {model}"


@test("[8] Adaptive Cortex (State + Policy + Gate)")
def test_cortex():
    from celr.core.types import TaskContext, Plan, Step
    from celr.cortex.state import StateExtractor
    from celr.cortex.policy import MetaPolicy, CortexAction
    from celr.cortex.gatekeeper import PromotionGate
    import numpy as np

    context = TaskContext(original_request="Write Python code", budget_limit_usd=1.0)
    context.current_spread_usd = 0.30  # 30% spent

    plan = Plan(
        original_goal="Test",
        items=[
            Step(id="s1", description="Step 1", estimated_difficulty=0.5),
            Step(id="s2", description="Step 2", estimated_difficulty=0.8),
        ]
    )

    # State Extraction
    extractor = StateExtractor()
    state = extractor.extract(context, plan, current_step_idx=0, retry_count=1)

    assert state.shape == (8,), f"Expected 8-dim state, got {state.shape}"
    assert 0.0 <= state[0] <= 1.0, f"Budget ratio out of range: {state[0]}"
    assert state[7] == 1.0, f"Should detect 'code' task, got {state[7]}"

    budget_ratio = state[0]
    expected_budget = (1.0 - 0.30) / 1.0
    assert abs(budget_ratio - expected_budget) < 0.01, f"Budget ratio wrong: {budget_ratio} vs {expected_budget}"

    # Policy Decision
    policy = MetaPolicy()
    action = policy.get_action(state)
    assert isinstance(action, CortexAction)
    assert action in list(CortexAction)

    # Promotion Gate
    baseline = {"success_rate": 0.80, "avg_cost": 0.15, "avg_retries": 1.0}
    gate = PromotionGate(baseline_stats=baseline)

    # Good candidate → promote
    good = {"success_rate": 0.85, "avg_cost": 0.10, "avg_retries": 0.5, "unsafe_count": 0}
    assert gate.evaluate(good) is True

    # Bad candidate (unsafe) → reject
    bad = {"success_rate": 0.90, "avg_cost": 0.05, "avg_retries": 0.2, "unsafe_count": 1}
    assert gate.evaluate(bad) is False

    # Worse cost → reject
    expensive = {"success_rate": 0.85, "avg_cost": 0.20, "avg_retries": 0.5, "unsafe_count": 0}
    assert gate.evaluate(expensive) is False


@test("[9] Full Executor Pipeline (End-to-End)")
def test_executor_e2e():
    from celr.core.config import CELRConfig
    from celr.core.types import TaskContext
    from celr.core.cost_tracker import CostTracker
    from celr.core.reasoning import ReasoningCore
    from celr.core.planner import Planner
    from celr.core.escalation import EscalationManager
    from celr.core.tools import ToolRegistry
    from celr.core.verifier import Verifier
    from celr.core.reflection import SelfReflection
    from celr.core.executor import TaskExecutor

    llm = SimulationLLM()
    config = CELRConfig(budget_limit=1.00)
    context = TaskContext(original_request="Calculate 2+2", budget_limit_usd=1.0)

    tracker = CostTracker(context)
    escalation = EscalationManager(tracker, config.get_model_tiers())
    tools = ToolRegistry()
    reasoning = ReasoningCore(llm=llm)
    planner = Planner(reasoning)
    verifier = Verifier(tool_registry=tools, llm=llm)
    reflection = SelfReflection(llm=llm)

    with patch.object(escalation, "get_provider", return_value=llm):
        executor = TaskExecutor(
            context=context,
            planner=planner,
            cost_tracker=tracker,
            escalation_manager=escalation,
            tool_registry=tools,
            verifier=verifier,
            reflection=reflection,
        )

        plan = planner.create_initial_plan(context)
        status = executor.run(plan)

    assert status == "SUCCESS", f"Expected SUCCESS, got {status}"
    assert context.current_spread_usd > 0, "Cost should have been tracked"
    assert len(context.execution_history) > 0, "Execution history should not be empty"

    # Check all steps completed
    completed = sum(1 for s in plan.items if s.status.value == "COMPLETED")
    assert completed == len(plan.items), f"Only {completed}/{len(plan.items)} steps completed"


@test("[10] Training Pipeline (Scorer + Data Generator)")
def test_training_pipeline():
    from celr.core.config import CELRConfig
    from celr.training.scorer import TrajectoryScorer
    from celr.training.data_generator import PreferencePairGenerator
    from celr.training.pipeline import TrainingPipeline

    # Score trajectories
    scorer = TrajectoryScorer()

    good_traj = {
        "task_id": "t1", "original_request": "Compute 2+2",
        "final_status": "SUCCESS", "budget_limit_usd": 1.0,
        "total_cost_usd": 0.01,
        "plan": [{"id": "s1", "status": "COMPLETED", "retry_count": 0}],
    }
    bad_traj = {
        "task_id": "t2", "original_request": "Complex task",
        "final_status": "FAILED", "budget_limit_usd": 1.0,
        "total_cost_usd": 0.90,
        "plan": [{"id": "s1", "status": "FAILED", "retry_count": 5}],
    }

    good_score = scorer.score(good_traj)
    bad_score = scorer.score(bad_traj)
    assert good_score.composite_score > bad_score.composite_score, \
        f"Good ({good_score.composite_score:.2f}) should outscore bad ({bad_score.composite_score:.2f})"

    # Ranking — returns List[TrajectoryScore], sorted by composite_score
    ranked = scorer.rank([good_traj, bad_traj])
    assert ranked[0].trajectory_id == "t1", "Good trajectory should rank first"

    # Data generation — requires List[TrajectoryScore]
    generator = PreferencePairGenerator()
    scored = [good_score, bad_score]
    pairs = generator.create_dpo_pairs(scored)
    assert len(pairs) >= 1, "Should generate at least 1 DPO pair"

    # Full pipeline with temp dir
    tmp_log = tempfile.mkdtemp()
    tmp_out = tempfile.mkdtemp()
    log_file = os.path.join(tmp_log, "test.jsonl")
    with open(log_file, "w") as f:
        f.write(json.dumps(good_traj) + "\n")
        f.write(json.dumps(bad_traj) + "\n")

    try:
        config = CELRConfig(log_dir=tmp_log)
        pipeline = TrainingPipeline(config=config, output_dir=tmp_out)
        report = pipeline.run()
        assert report.trajectories_loaded == 2
        assert report.trajectories_scored == 2
    finally:
        shutil.rmtree(tmp_log)
        shutil.rmtree(tmp_out)


@test("[11] Self-Reward Scorer (LLM-as-Judge)")
def test_self_reward():
    from celr.training.self_reward import SelfRewardScorer

    llm = SimulationLLM()
    scorer = SelfRewardScorer(llm=llm)

    # Score a single response
    score = scorer.score_response("What is 2+2?", "The answer is 4.")
    assert 0.0 <= score <= 1.0, f"Score out of range: {score}"

    # Score a trajectory
    traj_score = scorer.score_trajectory({
        "original_request": "Calculate something",
        "final_status": "SUCCESS",
        "plan": [{"status": "COMPLETED", "description": "Did the thing"}]
    })
    assert 0.0 <= traj_score <= 1.0

    # Cost tracking
    assert scorer.total_scoring_cost > 0


@test("[12] Package Exports & Version")
def test_package():
    import celr
    assert hasattr(celr, "__version__")
    assert celr.__version__ == "0.1.0"
    assert hasattr(celr, "CELRConfig")
    assert hasattr(celr, "TaskExecutor")
    assert hasattr(celr, "BaseLLMProvider")
    assert hasattr(celr, "BudgetExhaustedError")


@test("[13] Custom Exceptions Hierarchy")
def test_exceptions():
    from celr.core.exceptions import (
        CELRError, BudgetExhaustedError, PlanningError,
        EscalationError, LLMProviderError, ToolExecutionError,
        VerificationError
    )

    # All inherit from CELRError
    assert issubclass(BudgetExhaustedError, CELRError)
    assert issubclass(PlanningError, CELRError)
    assert issubclass(EscalationError, CELRError)
    assert issubclass(LLMProviderError, CELRError)
    assert issubclass(ToolExecutionError, CELRError)
    assert issubclass(VerificationError, CELRError)

    # Can be caught as CELRError
    try:
        raise BudgetExhaustedError(budget_limit=1.0, current_spend=1.5)
    except CELRError as e:
        assert "Budget exhausted" in str(e)


@test("[14] Demo Script Runs Successfully")
def test_demo():
    """Run the demo.py script as a subprocess to verify it works."""
    import subprocess
    result = subprocess.run(
        [sys.executable, "demo.py"],
        cwd=r"f:\LLM CELR",
        capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, f"Demo failed with:\n{result.stderr}"
    assert "CELR LIVE DEMO" in result.stdout or "SUCCESS" in result.stdout or "DEMO" in result.stdout


# ─── Run ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
