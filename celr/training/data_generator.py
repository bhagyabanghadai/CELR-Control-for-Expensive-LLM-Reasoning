"""
Preference Pair Generator — creates training data for DPO and SPIN.

Two modes:
  1. DPO pairs: From scored trajectories, pair high-quality (chosen) with
     low-quality (rejected) responses for the same or similar tasks
  2. SPIN pairs: Real SFT data (chosen) vs model-generated responses (rejected)

Export formats:
  - TRL format (for Hugging Face TRL library)
  - ShareGPT format (for Axolotl, Llama-Factory)
  - SPIN format (for UCLA SPIN training)
"""

import json
import logging
from dataclasses import dataclass
from typing import List, Optional

from celr.training.scorer import TrajectoryScore

logger = logging.getLogger(__name__)


@dataclass
class PreferencePair:
    """A single preference pair for DPO/SPIN training."""

    prompt: str
    chosen: str       # Better response (wins)
    rejected: str     # Worse response (loses)
    chosen_score: float = 0.0
    rejected_score: float = 0.0
    pair_type: str = "dpo"  # "dpo" or "spin"


class PreferencePairGenerator:
    """
    Generates preference pairs from scored trajectories.

    Usage:
        gen = PreferencePairGenerator()
        pairs = gen.create_dpo_pairs(scored_trajectories)
        gen.export_trl_format(pairs, "dpo_train.json")
    """

    def __init__(self, min_score_gap: float = 0.15):
        """
        Args:
            min_score_gap: Minimum composite score difference between
                           chosen and rejected to form a valid pair.
        """
        self.min_score_gap = min_score_gap

    def create_dpo_pairs(
        self, scored: List[TrajectoryScore]
    ) -> List[PreferencePair]:
        """
        Create DPO pairs by pairing high-quality with low-quality trajectories.

        Strategy: Sort by score, pair top half with bottom half.
        Only creates pairs where score gap >= min_score_gap.
        """
        if len(scored) < 2:
            logger.warning("Need at least 2 scored trajectories for DPO pairs")
            return []

        # Sort by composite score (highest first)
        sorted_scores = sorted(
            scored, key=lambda s: s.composite_score, reverse=True
        )

        pairs = []
        mid = len(sorted_scores) // 2

        # Pair top scoring with bottom scoring
        for chosen, rejected in zip(sorted_scores[:mid], reversed(sorted_scores[mid:])):
            gap = chosen.composite_score - rejected.composite_score
            if gap < self.min_score_gap:
                continue

            pair = PreferencePair(
                prompt=chosen.original_request,
                chosen=self._trajectory_to_response(chosen),
                rejected=self._trajectory_to_response(rejected),
                chosen_score=chosen.composite_score,
                rejected_score=rejected.composite_score,
                pair_type="dpo",
            )
            pairs.append(pair)

        logger.info(
            f"Created {len(pairs)} DPO pairs from "
            f"{len(scored)} trajectories (min_gap={self.min_score_gap})"
        )
        return pairs

    def create_spin_pairs(
        self,
        sft_data: List[dict],
        model_responses: List[dict],
    ) -> List[PreferencePair]:
        """
        Create SPIN pairs: real SFT data = chosen, model-generated = rejected.

        Args:
            sft_data: List of {"prompt": str, "response": str} from real data
            model_responses: List of {"prompt": str, "response": str} from model
        """
        # Match by prompt
        model_map = {r["prompt"]: r["response"] for r in model_responses}

        pairs = []
        for sft in sft_data:
            prompt = sft["prompt"]
            if prompt not in model_map:
                continue

            pair = PreferencePair(
                prompt=prompt,
                chosen=sft["response"],       # Real human/expert data
                rejected=model_map[prompt],   # Model's own generation
                chosen_score=1.0,
                rejected_score=0.0,
                pair_type="spin",
            )
            pairs.append(pair)

        logger.info(
            f"Created {len(pairs)} SPIN pairs from "
            f"{len(sft_data)} SFT examples and "
            f"{len(model_responses)} model responses"
        )
        return pairs

    def export_trl_format(
        self, pairs: List[PreferencePair], output_file: str
    ) -> int:
        """
        Export pairs in TRL DPOTrainer format.

        Format: [{"prompt": str, "chosen": str, "rejected": str}, ...]
        """
        data = []
        for pair in pairs:
            data.append({
                "prompt": pair.prompt,
                "chosen": pair.chosen,
                "rejected": pair.rejected,
            })

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported {len(data)} pairs to TRL format: {output_file}")
        return len(data)

    def export_sharegpt_format(
        self, pairs: List[PreferencePair], output_file: str
    ) -> int:
        """
        Export pairs in ShareGPT format (for Axolotl/Llama-Factory).

        Creates conversations with chosen responses marked.
        """
        data = []
        for pair in pairs:
            data.append({
                "conversations": [
                    {"from": "system", "value": "You are an expert task-solving AI."},
                    {"from": "human", "value": pair.prompt},
                    {"from": "gpt", "value": pair.chosen},
                ],
                "rejected_conversations": [
                    {"from": "system", "value": "You are an expert task-solving AI."},
                    {"from": "human", "value": pair.prompt},
                    {"from": "gpt", "value": pair.rejected},
                ],
                "pair_type": pair.pair_type,
                "chosen_score": pair.chosen_score,
                "rejected_score": pair.rejected_score,
            })

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported {len(data)} pairs to ShareGPT format: {output_file}")
        return len(data)

    def export_spin_format(
        self, pairs: List[PreferencePair], output_file: str
    ) -> int:
        """
        Export pairs in SPIN training format.

        Format: [{"real": [messages], "generated": [messages]}, ...]
        """
        data = []
        for pair in pairs:
            if pair.pair_type != "spin":
                continue
            data.append({
                "real": [
                    {"role": "user", "content": pair.prompt},
                    {"role": "assistant", "content": pair.chosen},
                ],
                "generated": [
                    {"role": "user", "content": pair.prompt},
                    {"role": "assistant", "content": pair.rejected},
                ],
            })

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported {len(data)} SPIN pairs: {output_file}")
        return len(data)

    @staticmethod
    def _trajectory_to_response(score: TrajectoryScore) -> str:
        """Convert a scored trajectory into a text response for training."""
        return (
            f"Task: {score.original_request}\n"
            f"Status: {score.final_status}\n"
            f"Steps: {score.completed_steps}/{score.total_steps}\n"
            f"Retries: {score.total_retries}\n"
            f"Cost: ${score.total_cost_usd:.4f}\n"
            f"Quality Score: {score.composite_score:.3f}"
        )
