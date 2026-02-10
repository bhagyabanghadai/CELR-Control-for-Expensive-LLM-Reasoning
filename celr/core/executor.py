from celr.core.types import TaskContext, Plan, Step, StepType, TaskStatus, ModelConfig
from celr.core.planner import Planner
from celr.core.llm import BaseLLMProvider, LiteLLMProvider
from celr.core.cost_tracker import CostTracker
import time

from celr.core.escalation import EscalationManager

class TaskExecutor:
    def __init__(self, context: TaskContext, planner: Planner, cost_tracker: CostTracker):
        self.context = context
        self.planner = planner
        self.tracker = cost_tracker
        self.escalation_manager = EscalationManager(cost_tracker)
        
        # Default LLM is just a placeholder access point for now
        self.default_llm = planner.reasoning.llm 

    def run(self, plan: Plan):
        """
        Main execution loop.
        1. Get runnable steps.
        2. Execute them.
        3. Update status.
        4. Repeat until done or failed.
        """
        self.context.log("Starting execution loop...")
        
        while True:
            runnable_steps = self.planner.get_ready_steps(plan)
            
            # Check for completion
            if not runnable_steps:
                # Are we done or stuck?
                if all(s.status in [TaskStatus.COMPLETED, TaskStatus.SKIPPED] for s in plan.items):
                    self.context.log("All steps completed successfully.")
                    return "SUCCESS"
                
                failed_steps = [s for s in plan.items if s.status == TaskStatus.FAILED]
                if failed_steps:
                    self.context.log(f"Plan failed. {len(failed_steps)} steps failed.")
                    return "FAILED"
                    
                # If we have PENDING steps but none are runnable, we have a cycle or bug
                self.context.log("Stuck! No runnable steps but plan is not complete.")
                return "STUCK"

            for step in runnable_steps:
                self.execute_step(step)
                
            # Basic throttler
            time.sleep(0.1)

    def execute_step(self, step: Step):
        self.context.log(f"Executing step: {step.description} ({step.id})")
        step.status = TaskStatus.RUNNING
        
        try:
            # 1. Check Budget
            estimated_cost = 0.01 # Mock estimate
            if not self.tracker.can_afford(estimated_cost):
                self.context.log("Critical: Budget exhausted before step execution.")
                step.status = TaskStatus.FAILED
                return

            # 2. Router / Escalation Logic
            selected_model_name = self.escalation_manager.select_model(step)
            step.assigned_agent = selected_model_name
            self.context.log(f"Routing: Step difficulty {step.estimated_difficulty} -> Selected Model: {selected_model_name}")
            
            # NOTE: In a real implementation, we would now instantiate the specific provider for 'selected_model_name'
            # For this prototype, we are just logging the decision.
            
            # 3. Execution (Mock for now)
            if step.step_type == StepType.EXECUTION:
                # Simulating tool use
                result = f"Executed {step.description} using tool."
            else:
                # Reasoning/LLM call
                prompt = f"Perform this task: {step.description}\nContext: {self.context.execution_history[-3:]}"
                result = self.default_llm.generate(prompt)
                cost = self.default_llm.calculate_cost(prompt, result)
                self.tracker.add_cost(cost)
                step.cost_usd = cost
            
            step.output = result
            step.status = TaskStatus.COMPLETED
            self.context.log(f"Step {step.id} finished. output_len={len(result)}")
            
        except Exception as e:
            self.context.log(f"Step {step.id} failed: {e}")
            step.status = TaskStatus.FAILED
