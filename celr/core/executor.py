"""
Task Executor — the main execution loop for CELR.

This is the heart of the system. It:
  1. Gets runnable steps from the plan (respecting dependency order)
  2. Routes each step to the right model via EscalationManager
  3. Dispatches to LLM or ToolRegistry based on StepType
  4. Verifies results with Verifier
  5. On failure: runs SelfReflection → retries with better approach
  6. Tracks cost for every call

Key improvements (Overhaul Phase O-3):
  - Verifier, Reflection, and ToolRegistry are fully wired
  - Real LLM dispatch replaces mock execution
  - Verify-then-reflect retry loop with max_retries
  - Uses custom exceptions (BudgetExhaustedError)
  - Returns structured result with cost summary
"""

import logging
import time
from typing import Optional

from celr.core.cost_tracker import CostTracker
from celr.core.escalation import EscalationManager
from celr.core.exceptions import BudgetExhaustedError, ToolExecutionError
from celr.core.llm import BaseLLMProvider, LLMUsage
from celr.core.planner import Planner
from celr.core.reflection import SelfReflection
from celr.core.tools import ToolRegistry
from celr.core.types import Plan, Step, StepType, TaskContext, TaskStatus
from celr.core.verifier import Verifier

logger = logging.getLogger(__name__)


class TaskExecutor:
    """
    Main execution engine.
    
    Orchestrates: Plan → Route → Execute → Verify → (Reflect → Retry) → Log
    """

    def __init__(
        self,
        context: TaskContext,
        planner: Planner,
        cost_tracker: CostTracker,
        escalation_manager: EscalationManager,
        tool_registry: ToolRegistry,
        verifier: Verifier,
        reflection: SelfReflection,
    ):
        self.context = context
        self.planner = planner
        self.tracker = cost_tracker
        self.escalation = escalation_manager
        self.tools = tool_registry
        self.verifier = verifier
        self.reflection = reflection

    def run(self, plan: Plan) -> str:
        """
        Main execution loop.
        
        1. Get runnable steps (dependencies met).
        2. Execute each step (route → dispatch → verify → reflect).
        3. Repeat until done, failed, or stuck.
        
        Returns:
            "SUCCESS", "FAILED", or "STUCK"
        """
        self.context.log("Starting execution loop...")
        logger.info(f"Executing plan with {len(plan.items)} steps")

        while True:
            runnable_steps = self.planner.get_ready_steps(plan)

            # Check for completion
            if not runnable_steps:
                if all(s.status in [TaskStatus.COMPLETED, TaskStatus.SKIPPED] for s in plan.items):
                    self.context.log("All steps completed successfully.")
                    logger.info("Plan completed successfully")
                    return "SUCCESS"

                failed_steps = [s for s in plan.items if s.status == TaskStatus.FAILED]
                if failed_steps:
                    self.context.log(f"Plan failed. {len(failed_steps)} steps failed.")
                    logger.error(f"Plan failed: {len(failed_steps)} steps failed")
                    return "FAILED"

                # Pending steps but none runnable = cycle or bug
                self.context.log("Stuck! No runnable steps but plan is not complete.")
                logger.error("Execution stuck: no runnable steps")
                return "STUCK"

            for step in runnable_steps:
                try:
                    self._execute_with_retries(step)
                except BudgetExhaustedError as e:
                    self.context.log(f"Budget exhausted: {e}")
                    logger.error(f"Budget exhausted: {e}")
                    step.status = TaskStatus.FAILED
                    return "FAILED"

            # Throttle to avoid tight loops
            time.sleep(0.1)

    def _execute_with_retries(self, step: Step) -> None:
        """
        Execute a step with the Verify → Reflect → Retry loop.
        
        For each attempt:
          1. Dispatch (LLM call or tool execution)
          2. Verify the result
          3. If verification fails: analyze failure → update step → retry
        """
        self.context.log(f"Executing step: {step.description} ({step.id})")
        step.status = TaskStatus.RUNNING

        for attempt in range(step.max_retries + 1):
            step.retry_count = attempt

            if attempt > 0:
                self.context.log(f"Retry {attempt}/{step.max_retries} for step {step.id}")
                logger.info(f"Retry {attempt}/{step.max_retries} for step {step.id}")

            try:
                # 1. Budget check
                estimated_cost = 0.01
                if not self.tracker.can_afford(estimated_cost):
                    raise BudgetExhaustedError(
                        budget_limit=self.context.budget_limit_usd,
                        current_spend=self.context.current_spread_usd,
                        step_description=step.description,
                    )

                # 2. Dispatch (route + execute)
                result = self._dispatch(step)
                step.output = result

                # 3. Verify
                is_valid = self.verifier.verify(step, self.context)

                if is_valid:
                    step.status = TaskStatus.COMPLETED
                    self.context.log(
                        f"Step {step.id} completed (attempt {attempt + 1}). "
                        f"output_len={len(result)}"
                    )
                    logger.info(f"Step {step.id} verified OK on attempt {attempt + 1}")
                    return
                else:
                    # 4. Verification failed → Reflect
                    self.context.log(
                        f"Step {step.id} verification failed: {step.verification_notes}"
                    )

                    if self.reflection.should_retry(step, attempt):
                        # Get reflection analysis and enhance the step description
                        fix = self.reflection.analyze_failure(step, self.context)
                        step.description += f"\n[Retry {attempt + 1} fix]: {fix}"
                        logger.info(f"Reflection suggested fix for step {step.id}")
                    else:
                        logger.warning(f"Step {step.id} failed verification, no more retries")
                        break

            except BudgetExhaustedError:
                raise  # Propagate budget errors up
            except ToolExecutionError as e:
                self.context.log(f"Tool error on step {step.id}: {e}")
                logger.error(f"Tool execution failed: {e}")
                step.output = str(e)
                if not self.reflection.should_retry(step, attempt):
                    break
            except Exception as e:
                self.context.log(f"Step {step.id} failed: {e}")
                logger.exception(f"Unexpected error on step {step.id}")
                step.output = str(e)
                if not self.reflection.should_retry(step, attempt):
                    break

        # All retries exhausted
        step.status = TaskStatus.FAILED
        self.context.log(f"Step {step.id} FAILED after {step.retry_count + 1} attempts.")
        logger.error(f"Step {step.id} failed after {step.retry_count + 1} attempts")

    def _dispatch(self, step: Step) -> str:
        """
        Route and execute a step based on its type.
        
        - EXECUTION → ToolRegistry
        - REASONING → LLM call via EscalationManager
        - VERIFICATION → LLM call (verifier handles this separately)
        """
        if step.step_type == StepType.EXECUTION:
            return self._dispatch_tool(step)
        else:
            return self._dispatch_llm(step)

    def _dispatch_tool(self, step: Step) -> str:
        """Execute a tool-based step."""
        # Extract tool name and code from description
        # Convention: step description starts with "TOOL:tool_name:" for tool steps
        if step.description.startswith("TOOL:"):
            parts = step.description.split(":", 2)
            tool_name = parts[1] if len(parts) > 1 else "python_repl"
            code = parts[2] if len(parts) > 2 else step.description
        else:
            # Default: ask LLM for code, then execute
            tool_name = "python_repl"
            code = self._get_code_from_llm(step)

        self.context.log(f"Dispatching to tool: {tool_name}")
        result = self.tools.execute(tool_name, code=code)
        return result

    def _dispatch_llm(self, step: Step) -> str:
        """Execute a reasoning/verification step via LLM."""
        # Get the right provider based on difficulty + budget
        provider = self.escalation.get_provider(step)

        prompt = (
            f"Task: {step.description}\n\n"
            f"Context (recent history):\n{self.context.execution_history[-5:]}\n\n"
            f"Provide a clear, actionable response."
        )

        text, usage = provider.generate(prompt)

        # Track cost
        cost = provider.calculate_cost(usage)
        self.tracker.add_cost(cost)
        step.cost_usd = cost

        self.context.log(
            f"LLM call: {step.assigned_agent}, "
            f"tokens={usage.total_tokens}, cost=${cost:.6f}"
        )

        return text

    def _get_code_from_llm(self, step: Step) -> str:
        """Ask LLM to generate executable code for a tool step."""
        provider = self.escalation.get_provider(step)
        prompt = (
            f"Generate Python code to accomplish this task:\n{step.description}\n\n"
            f"Return ONLY the code, no explanation."
        )
        code, usage = provider.generate(prompt)
        cost = provider.calculate_cost(usage)
        self.tracker.add_cost(cost)
        return code
