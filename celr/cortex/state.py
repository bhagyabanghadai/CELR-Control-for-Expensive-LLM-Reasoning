import numpy as np
from typing import Dict, List, Any
from celr.core.types import TaskContext, Plan, Step, TaskStatus


class StateExtractor:
    """
    Extracts RL state vectors from CELR execution logs.
    State Dimension: 8
    [
      0: Budget Remaining Ratio (0.0 - 1.0)
      1: Current Step Index Ratio (0.0 - 1.0)
      2: Current Retry Count (Normalized by Max Retries)
      3: Verification Failure Count (Normalized)
      4: Estimated Difficulty Score (0.0 - 1.0)
      5: Previous Action (One-hot or Enum Index) - Placeholder
      6: Time Elapsed Ratio (0.0 - 1.0, assuming max 300s)
      7: Is Coding Task (0.0 or 1.0)
    ]
    """

    def __init__(self, max_retries: int = 3, max_time_s: int = 300):
        self.max_retries = max_retries
        self.max_time = max_time_s
        self.state_dim = 8

    def extract(self, context: TaskContext, plan: Plan = None, current_step_idx: int = 0, retry_count: int = 0) -> np.ndarray:
        """
        Converts current execution context into a state vector.
        """
        # 1. Budget Remaining
        budget_ratio = 0.0
        if context.budget_limit_usd > 0:
            remaining = context.budget_limit_usd - context.current_spread_usd
            budget_ratio = max(0.0, remaining / context.budget_limit_usd)

        # 2. Step Progress
        progress_ratio = 0.0
        if plan and len(plan.items) > 0:
            progress_ratio = current_step_idx / len(plan.items)

        # 3. Retry Count
        retry_ratio = min(1.0, retry_count / self.max_retries)

        # 4. Verification Failures (Approximated by total retries for now)
        ver_fail_ratio = retry_ratio

        # 5. Difficulty — extract from current step if available
        difficulty = 0.5  # Default
        if plan and plan.items and current_step_idx < len(plan.items):
            difficulty = plan.items[current_step_idx].estimated_difficulty

        # 6. Previous Action (Placeholder)
        prev_action = 0.0

        # 7. Time Elapsed (Placeholder — context doesn't track start time yet)
        time_ratio = 0.0

        # 8. Task Type (Heuristic)
        prompt = context.original_request.lower()
        is_coding = 1.0 if any(kw in prompt for kw in ["code", "python", "script", "function", "class", "program"]) else 0.0

        state = np.array([
            budget_ratio,
            progress_ratio,
            retry_ratio,
            ver_fail_ratio,
            difficulty,
            prev_action,
            time_ratio,
            is_coding
        ], dtype=np.float32)

        return state
