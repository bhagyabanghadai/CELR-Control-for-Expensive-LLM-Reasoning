"""
CELR Live Demo — Dry run without API keys.

This script demonstrates the CELR system running a task using a Mock LLM.
It shows:
  - Task decomposition (Plan)
  - Cost tracking
  - Tool execution (mocked)
  - Reflection and retries
  - Structured logging

Usage:
    python demo.py
"""

import logging
import os
import sys
import time
import json
from unittest.mock import MagicMock, patch

from celr.core.config import CELRConfig
from celr.core.executor import TaskExecutor
from celr.core.types import TaskContext, TaskStatus
from celr.core.llm import BaseLLMProvider, LLMUsage
from celr.core.cost_tracker import CostTracker
from celr.core.reasoning import ReasoningCore
from celr.core.planner import Planner
from celr.core.escalation import EscalationManager
from celr.core.tools import ToolRegistry
from celr.core.verifier import Verifier
from celr.core.reflection import SelfReflection

# Configure logging to show the action
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
# Silence internal noisy loggers if needed
logging.getLogger("celr.core.tools").setLevel(logging.INFO)

logger = logging.getLogger("demo")

class DemoLLM(BaseLLMProvider):
    """Mock LLM that returns pre-scripted responses for the demo."""
    
    def __init__(self):
        self.call_count = 0

    def generate(self, prompt: str, **kwargs):
        self.call_count += 1
        time.sleep(0.3)  # Simulate latency
        
        # 1. Decomposition (Planning)
        if "Original Goal:" in prompt and "Context:" in prompt:
            return (
                json.dumps({
                    "original_goal": "Write a Python script that prints Hello World",
                    "items": [
                        {
                            "id": "step-1",
                            "description": "Generate Python code to print Hello World",
                            "estimated_difficulty": 0.1,
                            "dependencies": []
                        },
                        {
                            "id": "step-2",
                            "description": "Verify the code is correct",
                            "estimated_difficulty": 0.2,
                            "dependencies": ["step-1"]
                        }
                    ]
                }),
                LLMUsage(prompt_tokens=100, completion_tokens=150)
            )
        
        # 2. Difficulty Estimation
        elif "Estimate the difficulty" in prompt:
            return (
                json.dumps({"difficulty_score": 0.2, "reasoning": "Simple task"}),
                LLMUsage(prompt_tokens=50, completion_tokens=20)
            )
            
        # 3. Verification
        elif "verdict" in prompt.lower() or "Verification Task:" in prompt:
             return (
                'VERDICT: YES\nExplanation: The code is valid Python and matches the intent.',
                LLMUsage(prompt_tokens=50, completion_tokens=20)
            )
            
        # 4. Code Generation (Tool Step)
        elif "Generate Python code" in prompt:
            return (
                "```python\nprint('Hello, World from CELR Demo!')\n```",
                LLMUsage(prompt_tokens=60, completion_tokens=30)
            )
            
        # Default fallback
        else:
            return (
                "I am a mock LLM. I received your request.",
                LLMUsage(prompt_tokens=10, completion_tokens=10)
            )

    def calculate_cost(self, usage: LLMUsage) -> float:
        # Mock cost: $1.00 per million tokens (simplified)
        return 1e-6 * (usage.prompt_tokens + usage.completion_tokens)


def run_demo():
    print(f"\n{'='*60}")
    print("[START] CELR LIVE DEMO (DRY RUN)")
    print(f"{'='*60}\n")
    
    # 1. Configuration
    config = CELRConfig(
        budget_limit=0.50, 
        verbose=True,
        small_model="mock/small",
    )
    
    # 2. Shared Components
    demo_llm = DemoLLM()
    
    request = "Write a Python script that prints Hello World"
    context = TaskContext(
        task_id="demo-run-001",
        original_request=request,
        budget_limit_usd=config.budget_limit
    )
    
    # 3. Dependency Injection
    tracker = CostTracker(context)
    escalation = EscalationManager(tracker, config.get_model_tiers())
    
    tools = ToolRegistry()
    
    # Reasoning & Planner use the LLM directly
    reasoning = ReasoningCore(llm=demo_llm)
    planner = Planner(reasoning)
    
    # Verifier & Reflection use the LLM directly
    verifier = Verifier(tool_registry=tools, llm=demo_llm)
    reflection = SelfReflection(llm=demo_llm)
    
    # 4. Patch Escalation Manager
    # The Executor asks EscalationManager for a provider for each step.
    # We patch it to always return our demo_llm.
    with patch.object(escalation, 'get_provider', return_value=demo_llm):
        
        # 5. Initialize Executor
        executor = TaskExecutor(
            context=context,
            planner=planner,
            cost_tracker=tracker,
            escalation_manager=escalation,
            tool_registry=tools,
            verifier=verifier,
            reflection=reflection
        )
        
        print(f"[TASK]   {request}")
        print(f"[BUDGET] ${config.budget_limit:.2f}")
        print(f"[MODEL]  Mock Provider (Free)")
        print("-" * 60)
        
        # 6. Run Execution
        start_time = time.time()
        try:
            # First, create the plan (usually done by CLI, but we do it manually here)
            plan = planner.create_initial_plan(context)
            context.log("Plan created successfully")
            
            # Then execute the plan
            status = executor.run(plan)
            
        except Exception as e:
            logger.exception(f"Demo crashed: {e}")
            status = "CRASHED"
            
        duration = time.time() - start_time
        print("-" * 60)
        
        # 7. Results
        if status == "SUCCESS":
            print("\n[SUCCESS] DEMO COMPLETED")
            print(f"[TIME]    {duration:.2f}s")
            print(f"[COST]    ${context.current_spread_usd:.6f}")
            print(f"[STEPS]   {len(plan.items)}")
            
            # Save trajectory for training pipeline
            os.makedirs("logs/Traj", exist_ok=True)
            traj_file = f"logs/Traj/demo_{int(time.time())}.json"
            traj_data = {
                "task_id": context.task_id,
                "original_request": context.original_request,
                "final_status": status,
                "total_cost": context.current_spread_usd,
                "budget_limit_usd": context.budget_limit_usd,
                "plan": [
                    {
                        "id": step.id,
                        "description": step.description,
                        "status": step.status.value,
                        "difficulty": step.estimated_difficulty,
                        "agent": step.assigned_agent,
                        "output": step.output,
                        "retry_count": step.retry_count,
                    }
                    for step in plan.items
                ],
                "execution_log": context.execution_history,
            }
            with open(traj_file, "w") as f:
                json.dump(traj_data, f, indent=2)
            print(f"[LOG]     Saved trajectory to {traj_file}")
            
        else:
            print(f"\n[FAILED] DEMO INCOMPLETE: status={status}")
            print("\n[DEBUG] Execution History:")
            for line in context.execution_history:
                print(line)

if __name__ == "__main__":
    run_demo()
