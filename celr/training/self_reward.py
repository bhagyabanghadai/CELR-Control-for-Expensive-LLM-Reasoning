"""
Self-Reward Scorer — LLM-as-Judge for trajectory quality.

Implements Meta's Self-Rewarding Language Model concept:
The LLM rates its OWN outputs on quality dimensions, creating
reward signals without human annotation.

The scoring prompt asks the model to rate on a 1-5 scale across:
  - Accuracy: Is the response correct?
  - Completeness: Does it fully address the task?
  - Efficiency: Was it achieved with minimal retries/cost?

Returns a normalized 0.0-1.0 score for use in DPO training.
"""

import json
import logging
import re
from typing import List, Optional, Tuple

from celr.core.llm import BaseLLMProvider, LLMUsage
from celr.training.data_generator import PreferencePair
from celr.training.scorer import TrajectoryScore

logger = logging.getLogger(__name__)

REWARD_PROMPT_TEMPLATE = """You are evaluating the quality of an AI agent's task execution.

## Task
{prompt}

## Agent's Response
{response}

## Scoring Instructions
Rate the response on three dimensions (1-5 each):

1. **Accuracy** (1-5): Is the response correct and factually accurate?
2. **Completeness** (1-5): Does it fully address all parts of the task?
3. **Efficiency** (1-5): Was it achieved with minimal steps and retries?

Respond ONLY in this exact format:
Accuracy: <score>
Completeness: <score>
Efficiency: <score>
Total: <sum>
"""

SCORE_PATTERN = re.compile(
    r"Accuracy:\s*(\d)\s*\n"
    r"Completeness:\s*(\d)\s*\n"
    r"Efficiency:\s*(\d)\s*\n"
    r"Total:\s*(\d+)",
    re.IGNORECASE,
)


class SelfRewardScorer:
    """
    Uses an LLM to score responses (LLM-as-Judge pattern from Meta).

    Usage:
        scorer = SelfRewardScorer(llm=provider)
        score = scorer.score_response("Write hello world", "print('Hello!')")
        pairs = scorer.generate_reward_pairs(trajectories)
    """

    def __init__(self, llm: BaseLLMProvider):
        self.llm = llm
        self.total_scoring_cost = 0.0

    def score_response(self, prompt: str, response: str) -> float:
        """
        Score a single prompt-response pair using LLM-as-Judge.

        Returns:
            Normalized score between 0.0 and 1.0
        """
        scoring_prompt = REWARD_PROMPT_TEMPLATE.format(
            prompt=prompt[:500],      # Truncate for token budget
            response=response[:1000],
        )

        try:
            text, usage = self.llm.generate(scoring_prompt)
            cost = self.llm.calculate_cost(usage)
            self.total_scoring_cost += cost

            score = self._parse_score(text)
            logger.debug(
                f"Self-reward score: {score:.2f} "
                f"(tokens={usage.total_tokens}, cost=${cost:.6f})"
            )
            return score

        except Exception as e:
            logger.warning(f"Self-reward scoring failed: {e}. Returning 0.5")
            return 0.5  # Neutral fallback

    def score_trajectory(self, trajectory: dict) -> float:
        """Score a full trajectory by evaluating the overall execution."""
        request = trajectory.get("original_request", "")
        plan = trajectory.get("plan", [])
        status = trajectory.get("final_status", "FAILED")

        # Build a summary of the execution for the judge
        step_summaries = []
        for step in plan[:10]:  # Cap at 10 steps
            step_summaries.append(
                f"- [{step.get('status', 'UNKNOWN')}] {step.get('description', 'N/A')[:80]}"
            )

        response = (
            f"Final Status: {status}\n"
            f"Steps Executed:\n" + "\n".join(step_summaries)
        )

        return self.score_response(request, response)

    def generate_reward_pairs(
        self,
        trajectories: List[dict],
        min_gap: float = 0.2,
    ) -> List[PreferencePair]:
        """
        Score all trajectories with LLM-as-Judge, then create
        preference pairs from high/low scored ones.
        """
        # Score each trajectory
        scored_items: List[Tuple[dict, float]] = []
        for traj in trajectories:
            score = self.score_trajectory(traj)
            scored_items.append((traj, score))

        # Sort by score (highest first)
        scored_items.sort(key=lambda x: x[1], reverse=True)

        # Create pairs: top half vs bottom half
        pairs = []
        mid = len(scored_items) // 2

        for (chosen_traj, chosen_score), (rejected_traj, rejected_score) in zip(
            scored_items[:mid], reversed(scored_items[mid:])
        ):
            if chosen_score - rejected_score < min_gap:
                continue

            pair = PreferencePair(
                prompt=chosen_traj.get("original_request", ""),
                chosen=self._trajectory_summary(chosen_traj),
                rejected=self._trajectory_summary(rejected_traj),
                chosen_score=chosen_score,
                rejected_score=rejected_score,
                pair_type="self_reward",
            )
            pairs.append(pair)

        logger.info(
            f"Generated {len(pairs)} self-reward pairs from "
            f"{len(trajectories)} trajectories. "
            f"Scoring cost: ${self.total_scoring_cost:.4f}"
        )
        return pairs

    def _parse_score(self, text: str) -> float:
        """Parse the LLM's structured score response into 0.0-1.0."""
        match = SCORE_PATTERN.search(text)
        if match:
            accuracy = int(match.group(1))
            completeness = int(match.group(2))
            efficiency = int(match.group(3))
            total = accuracy + completeness + efficiency
            # Normalize: max total = 15, min = 3
            normalized = (total - 3) / 12.0
            return max(0.0, min(1.0, normalized))

        # Fallback: try to find any number
        numbers = re.findall(r"(\d+)/(?:15|5)", text)
        if numbers:
            value = int(numbers[0])
            return min(value / 15.0, 1.0)

        logger.warning(f"Could not parse self-reward score from: {text[:100]}")
        return 0.5  # Neutral fallback

    @staticmethod
    def _trajectory_summary(trajectory: dict) -> str:
        """Create a concise text summary of a trajectory."""
        plan = trajectory.get("plan", [])
        completed = sum(1 for s in plan if s.get("status") == "COMPLETED")
        return (
            f"Task: {trajectory.get('original_request', 'N/A')}\n"
            f"Status: {trajectory.get('final_status', 'UNKNOWN')}\n"
            f"Steps: {completed}/{len(plan)} completed\n"
            f"Cost: ${trajectory.get('total_cost_usd', 0):.4f}"
        )
