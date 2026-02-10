from celr.core.types import Step, TaskContext, ModelConfig
from celr.core.cost_tracker import CostTracker

class EscalationManager:
    def __init__(self, cost_tracker: CostTracker):
        self.tracker = cost_tracker
        
        # Hardcoded configs for now
        self.small_model_config = ModelConfig(
            name="ollama/llama3", 
            provider="ollama", 
            cost_per_million_input_tokens=0.0, 
            cost_per_million_output_tokens=0.0
        )
        self.large_model_config = ModelConfig(
            name="gpt-4o", 
            provider="openai", 
            cost_per_million_input_tokens=5.0, 
            cost_per_million_output_tokens=15.0
        )
        
        # Heuristics
        self.difficulty_threshold = 0.7
        self.min_budget_for_escalation_usd = 0.10 # Reserve $0.10 for the big gun

    def select_model(self, step: Step) -> str:
        """
        Returns the name of the model to use for this step.
        Decides based on:
        1. Step Difficulty
        2. Remaining Budget
        3. History (did we already try cheap model?)
        """
        
        remaining_budget = self.tracker.check_remaining_budget()
        
        # 1. Budget Safety Check
        # If we are effectively out of money, force small model (or fail if handled elsewhere)
        if remaining_budget <= 0:
            return self.small_model_config.name

        # 2. "Budget Drain" Loophole Mitigation
        # If we are getting close to the reserve, and the task is hard, SAVE the money for the big model.
        # Don't waste the last $0.15 on a Mistral 7B attempt if we really need GPT-4.
        if remaining_budget < 2 * self.min_budget_for_escalation_usd:
             if step.estimated_difficulty > 0.4:
                 return self.large_model_config.name

        # 3. Standard Logic
        if step.estimated_difficulty >= self.difficulty_threshold:
            # It's a hard task. Can we afford the big model?
            if self.tracker.can_afford(self.min_budget_for_escalation_usd):
                return self.large_model_config.name
            else:
                # We need intelligence but can't afford it. 
                # Log warning and try best effort with small model.
                # In a real system, we might ask user for more budget here.
                return self.small_model_config.name
        
        # Default to cheap model
        return self.small_model_config.name
