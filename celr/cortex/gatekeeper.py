import logging
from typing import Dict

logger = logging.getLogger(__name__)


class PromotionGate:
    """
    Gatekeeper that allows or blocks a new Policy from being deployed.
    Checks: Safety, Success Rate, Cost Efficiency, Retry Count.
    """

    def __init__(self, baseline_stats: Dict):
        self.baseline = baseline_stats  # e.g. {'success_rate': 0.8, 'avg_cost': 0.15}

    def evaluate(self, candidate_stats: Dict) -> bool:
        """
        Returns True if candidate policy should be promoted.
        """
        # 1. Safety Hard Constraint
        if candidate_stats.get("unsafe_count", 0) > 0:
            logger.warning("[GATE] Policy REJECTED: Unsafe actions detected.")
            return False

        # 2. Success Rate Check
        if candidate_stats.get("success_rate", 0.0) < self.baseline.get("success_rate", 0.0) - 0.05:
            logger.warning("[GATE] Policy REJECTED: Success rate regression.")
            return False

        # 3. Cost/Efficiency Check
        if candidate_stats.get("avg_cost", 1.0) > self.baseline.get("avg_cost", 1.0):
            logger.warning("[GATE] Policy REJECTED: Cost increased.")
            return False

        # 4. Retry Check
        if candidate_stats.get("avg_retries", 0) > self.baseline.get("avg_retries", 0) + 0.5:
            logger.warning("[GATE] Policy REJECTED: Retries increased significantly.")
            return False

        logger.info("[GATE] Policy PROMOTED! 🚀")
        return True
