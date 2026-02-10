"""
Trajectory Scorer — rates completed execution trajectories on quality.

Scoring dimensions:
  1. success_score  — did the task succeed? (1.0 or 0.0)
  2. efficiency_score — fewer retries = better (1.0 / (1 + total_retries))
  3. cost_score — lower cost relative to budget = better
  4. step_completion_rate — what fraction of steps completed?
  5. composite_score — weighted combination → single 0.0–1.0 metric

Used by the training pipeline to:
  - Filter high-quality trajectories for SFT
  - Rank trajectories for DPO preference pairs (chosen vs rejected)
  - Weight training examples by quality
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryScore:
    """Scored trajectory with individual and composite quality metrics."""

    trajectory_id: str
    original_request: str
    final_status: str

    # Individual scores (0.0 to 1.0)
    success_score: float = 0.0
    efficiency_score: float = 0.0
    cost_score: float = 0.0
    step_completion_rate: float = 0.0

    # Composite (weighted combination)
    composite_score: float = 0.0

    # Raw data for reference
    total_retries: int = 0
    total_cost_usd: float = 0.0
    total_steps: int = 0
    completed_steps: int = 0


class TrajectoryScorer:
    """
    Scores execution trajectories on multiple quality dimensions.

    Usage:
        scorer = TrajectoryScorer()
        score = scorer.score(trajectory_dict)
        ranked = scorer.rank(list_of_trajectories)
    """

    def __init__(
        self,
        weight_success: float = 0.40,
        weight_efficiency: float = 0.25,
        weight_cost: float = 0.20,
        weight_completion: float = 0.15,
    ):
        self.weights = {
            "success": weight_success,
            "efficiency": weight_efficiency,
            "cost": weight_cost,
            "completion": weight_completion,
        }
        # Validate weights sum to ~1.0
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            logger.warning(f"Score weights sum to {total:.2f}, expected 1.0")

    def score(self, trajectory: dict) -> TrajectoryScore:
        """
        Score a single trajectory.

        Expected trajectory dict keys:
            - task_id: str
            - original_request: str
            - final_status: "SUCCESS" | "FAILED" | "STUCK"
            - plan: list of step dicts
            - budget_limit_usd: float
            - total_cost_usd: float
        """
        task_id = trajectory.get("task_id", "unknown")
        request = trajectory.get("original_request", "")
        status = trajectory.get("final_status", "FAILED")
        plan = trajectory.get("plan", [])
        budget = trajectory.get("budget_limit_usd", 1.0)
        cost = trajectory.get("total_cost_usd", 0.0)

        # 1. Success score — binary
        success = 1.0 if status == "SUCCESS" else 0.0

        # 2. Efficiency — fewer retries is better
        total_retries = sum(
            step.get("retry_count", 0) for step in plan
        )
        efficiency = 1.0 / (1.0 + total_retries)

        # 3. Cost score — lower cost relative to budget is better
        if budget > 0:
            cost_ratio = min(cost / budget, 1.0)
            cost_score = 1.0 - cost_ratio  # $0 spent = 1.0, full budget = 0.0
        else:
            cost_score = 1.0  # Free is perfect

        # 4. Step completion rate
        total_steps = len(plan)
        completed = sum(
            1 for step in plan if step.get("status") == "COMPLETED"
        )
        completion_rate = completed / max(total_steps, 1)

        # 5. Composite score (weighted)
        composite = (
            self.weights["success"] * success
            + self.weights["efficiency"] * efficiency
            + self.weights["cost"] * cost_score
            + self.weights["completion"] * completion_rate
        )

        result = TrajectoryScore(
            trajectory_id=task_id,
            original_request=request,
            final_status=status,
            success_score=success,
            efficiency_score=efficiency,
            cost_score=cost_score,
            step_completion_rate=completion_rate,
            composite_score=round(composite, 4),
            total_retries=total_retries,
            total_cost_usd=cost,
            total_steps=total_steps,
            completed_steps=completed,
        )

        logger.debug(
            f"Scored trajectory {task_id}: "
            f"composite={composite:.3f} "
            f"(success={success}, efficiency={efficiency:.2f}, "
            f"cost={cost_score:.2f}, completion={completion_rate:.2f})"
        )

        return result

    def rank(self, trajectories: List[dict]) -> List[TrajectoryScore]:
        """Score and rank trajectories by composite score (highest first)."""
        scored = [self.score(t) for t in trajectories]
        scored.sort(key=lambda s: s.composite_score, reverse=True)
        logger.info(
            f"Ranked {len(scored)} trajectories. "
            f"Top={scored[0].composite_score:.3f}, "
            f"Bottom={scored[-1].composite_score:.3f}"
            if scored else "No trajectories to rank"
        )
        return scored

    def filter_high_quality(
        self, trajectories: List[dict], threshold: float = 0.6
    ) -> List[TrajectoryScore]:
        """Return only trajectories above the quality threshold."""
        ranked = self.rank(trajectories)
        high_quality = [s for s in ranked if s.composite_score >= threshold]
        logger.info(
            f"Filtered {len(high_quality)}/{len(ranked)} trajectories "
            f"above threshold {threshold}"
        )
        return high_quality
