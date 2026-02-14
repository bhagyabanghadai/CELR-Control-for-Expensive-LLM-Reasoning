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
from celr.cortex import StateExtractor, MetaPolicy
from celr.cortex.policy import CortexAction # Explicit import

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
        
        # Phase 8: Adaptive Cortex
        self.state_extractor = StateExtractor()
        self.policy = MetaPolicy()
        self.current_plan: Optional[Plan] = None

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
        self.current_plan = plan # For Cortex state extraction

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
                # --- Phase 8: Adaptive Cortex Decision ---
                # 1. Extract State
                current_plan_idx = 0 
                if self.current_plan:
                    try:
                        current_plan_idx = [s.id for s in self.current_plan.items].index(step.id)
                    except ValueError:
                        pass
                
                state = self.state_extractor.extract(
                    self.context, 
                    self.current_plan, 
                    current_step_idx=current_plan_idx, 
                    retry_count=attempt
                )
                
                # 2. Get Action from Policy
                action = self.policy.get_action(state)
                logger.info(f"Cortex Action for step {step.id}: {action}")

                force_expensive = False
                
                if action == CortexAction.ABORT:
                    self.context.log(f"🛑 Cortex aborted step {step.id} (Safety/Cost risk)")
                    logger.warning("Cortex triggered ABORT")
                    step.status = TaskStatus.FAILED
                    break # Stop retries
                
                elif action == CortexAction.ESCALATE:
                    self.context.log("⚡ Cortex triggered ESCALATION (forcing stronger model)")
                    force_expensive = True
                
                elif action == CortexAction.STOP:
                    # Rare case: Early success prediction? 
                    # For now, treat as "skip execution if we think we assume success", 
                    # but mostly this action is for higher level loops.
                    pass

                # 3. Budget check (Legacy safety net)
                estimated_cost = 0.01
                if not self.tracker.can_afford(estimated_cost):
                    raise BudgetExhaustedError(
                        budget_limit=self.context.budget_limit_usd,
                        current_spend=self.context.current_spread_usd,
                        step_description=step.description,
                    )

                # 4. Dispatch (route + execute)
                # Pass force_expensive flag if Cortex requested escalation
                if step.step_type == StepType.EXECUTION:
                     # Tools don't use LLM provider directly in dispatch_tool yet, 
                     # but _get_code_from_llm does.
                     result = self._dispatch_tool(step, force_expensive)
                else:
                     result = self._dispatch_llm(step, force_expensive)
                
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

    def _dispatch_tool(self, step: Step, force_expensive: bool = False) -> str:
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
            code = self._get_code_from_llm(step, force_expensive=force_expensive)

        self.context.log(f"Dispatching to tool: {tool_name}")
        result = self.tools.execute(tool_name, code=code)
        
        # OPTIMIZATION for Benchmarks (HumanEval):
        # Always output the code so the grader can check it,
        # even if there is execution output (e.g. test prints).
        result += f"\n\n[Captured Code]\n{code}"
            
        return result

    def _dispatch_llm(self, step: Step, force_expensive: bool = False) -> str:
        """Execute a reasoning/verification step via LLM."""
        # Get the right provider based on difficulty + budget
        provider = self.escalation.get_provider(step, force_expensive=force_expensive)

        prompt = (
            f"Task: {step.description}\n\n"
            f"Context (recent history):\n{self.context.execution_history[-5:]}\n\n"
            f"System Instructions:\n"
            f"1. Provide a clear, actionable response.\n"
            f"2. Do NOT hallucinate data. Use ONLY values provided in the task.\n"
            f"3. For math problems, calculate explicitly. Do NOT use `input()`.\n"
            f"4. If writing code, ensure it is self-contained and prints the result.\n"
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

    def _get_code_from_llm(self, step: Step, force_expensive: bool = False) -> str:
        """Ask LLM to generate executable code for a tool step."""
        provider = self.escalation.get_provider(step, force_expensive=force_expensive)
        prompt = (
            f"Generate Python code to accomplish this task:\n{step.description}\n\n"
            f"Constraints:\n"
            f"1. Return ONLY the code, no explanation.\n"
            f"2. Do NOT use `input()`. Use variables for any values provided in the task.\n"
            f"3. The code must be self-contained and print the final result.\n"
            f"4. Use EXACT function names/signatures if specified in the task.\n"
        )
        code, usage = provider.generate(prompt)
        cost = provider.calculate_cost(usage)
        self.tracker.add_cost(cost)
        
        # Strip markdown formatting if present
        if "```" in code:
            code = code.replace("```python", "").replace("```py", "").replace("```", "").strip()
            
        return code
