import logging

from celr.core.types import TaskContext

logger = logging.getLogger(__name__)

class CostTracker:
    def __init__(self, context: TaskContext):
        self.context = context
        
    def add_cost(self, amount_usd: float):
        self.context.current_spread_usd += amount_usd
        if self.context.current_spread_usd > self.context.budget_limit_usd:
            self.context.log(f"WARNING: Budget exceeded! Spent ${self.context.current_spread_usd:.4f} > Limit ${self.context.budget_limit_usd:.4f}")
            logger.warning(f"Budget exceeded: ${self.context.current_spread_usd:.4f} > ${self.context.budget_limit_usd:.4f}")
            
    def check_remaining_budget(self) -> float:
        return self.context.budget_limit_usd - self.context.current_spread_usd
        
    def can_afford(self, estimated_cost: float) -> bool:
        return self.check_remaining_budget() >= estimated_cost
