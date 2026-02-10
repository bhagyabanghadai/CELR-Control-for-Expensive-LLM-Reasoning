"""
Training Pipeline — orchestrates the full self-improvement cycle.

The pipeline runs 5 steps:
  1. Collect — load successful trajectories from .celr_logs/
  2. Score   — rate each trajectory with TrajectoryScorer + SelfRewardScorer
  3. Pair    — generate DPO/SPIN preference pairs
  4. Export  — save to training format (TRL/ShareGPT/SPIN)
  5. Report  — print stats (total pairs, quality distribution)

This is the "meta-learning" loop that makes CELR self-improving:
successful execution trajectories are turned into training data
that makes the small model better over time.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional

from celr.core.config import CELRConfig
from celr.core.llm import BaseLLMProvider
from celr.training.scorer import TrajectoryScorer, TrajectoryScore
from celr.training.data_generator import PreferencePairGenerator, PreferencePair
from celr.training.self_reward import SelfRewardScorer

logger = logging.getLogger(__name__)


@dataclass
class TrainingReport:
    """Summary of a training pipeline run."""

    trajectories_loaded: int = 0
    trajectories_scored: int = 0
    high_quality_count: int = 0
    dpo_pairs_created: int = 0
    spin_pairs_created: int = 0
    self_reward_pairs_created: int = 0
    total_pairs_exported: int = 0
    scoring_cost_usd: float = 0.0
    output_files: List[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"=== Training Pipeline Report ===\n"
            f"Trajectories loaded:  {self.trajectories_loaded}\n"
            f"Trajectories scored:  {self.trajectories_scored}\n"
            f"High quality (≥0.6):  {self.high_quality_count}\n"
            f"DPO pairs created:    {self.dpo_pairs_created}\n"
            f"SPIN pairs created:   {self.spin_pairs_created}\n"
            f"Self-reward pairs:    {self.self_reward_pairs_created}\n"
            f"Total exported:       {self.total_pairs_exported}\n"
            f"Scoring cost:         ${self.scoring_cost_usd:.4f}\n"
            f"Output files:         {', '.join(self.output_files)}"
        )


class TrainingPipeline:
    """
    Full self-improvement training orchestrator.

    Usage:
        pipeline = TrainingPipeline(config=config, llm=provider)
        report = pipeline.run()
        print(report.summary())
    """

    def __init__(
        self,
        config: CELRConfig,
        llm: Optional[BaseLLMProvider] = None,
        quality_threshold: float = 0.6,
        output_dir: str = "training_data",
    ):
        self.config = config
        self.llm = llm
        self.quality_threshold = quality_threshold
        self.output_dir = output_dir
        self.log_dir = config.log_dir

        # Initialize components
        self.scorer = TrajectoryScorer()
        self.pair_gen = PreferencePairGenerator()
        self.self_reward = SelfRewardScorer(llm) if llm else None

    def run(self) -> TrainingReport:
        """Execute the full training pipeline."""
        report = TrainingReport()

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Step 1: Collect trajectories
        logger.info("Step 1/5: Loading trajectories...")
        trajectories = self._load_trajectories()
        report.trajectories_loaded = len(trajectories)

        if not trajectories:
            logger.warning("No trajectories found. Nothing to train on.")
            return report

        # Step 2: Score trajectories
        logger.info("Step 2/5: Scoring trajectories...")
        scored = self.scorer.rank(trajectories)
        report.trajectories_scored = len(scored)

        high_quality = [s for s in scored if s.composite_score >= self.quality_threshold]
        report.high_quality_count = len(high_quality)

        # Step 3: Generate preference pairs
        logger.info("Step 3/5: Generating preference pairs...")

        # 3a. DPO pairs from trajectory scoring
        dpo_pairs = self.pair_gen.create_dpo_pairs(scored)
        report.dpo_pairs_created = len(dpo_pairs)

        # 3b. Self-reward pairs (if LLM available)
        self_reward_pairs = []
        if self.self_reward:
            logger.info("Running LLM-as-Judge scoring...")
            self_reward_pairs = self.self_reward.generate_reward_pairs(trajectories)
            report.self_reward_pairs_created = len(self_reward_pairs)
            report.scoring_cost_usd = self.self_reward.total_scoring_cost

        # Combine all pairs
        all_pairs = dpo_pairs + self_reward_pairs

        # Step 4: Export
        logger.info("Step 4/5: Exporting training data...")
        total_exported = 0

        if all_pairs:
            # TRL format
            trl_path = os.path.join(self.output_dir, "dpo_train.json")
            count = self.pair_gen.export_trl_format(all_pairs, trl_path)
            total_exported += count
            report.output_files.append(trl_path)

            # ShareGPT format
            sgpt_path = os.path.join(self.output_dir, "sharegpt_train.json")
            count = self.pair_gen.export_sharegpt_format(all_pairs, sgpt_path)
            report.output_files.append(sgpt_path)

        # Export high-quality only as SFT data
        if high_quality:
            sft_path = os.path.join(self.output_dir, "sft_high_quality.json")
            self._export_sft(high_quality, sft_path)
            report.output_files.append(sft_path)

        report.total_pairs_exported = total_exported

        # Step 5: Report
        logger.info("Step 5/5: Pipeline complete.")
        logger.info(report.summary())

        return report

    def _load_trajectories(self) -> List[dict]:
        """Load trajectory JSONL files from log directory."""
        trajectories = []

        if not os.path.exists(self.log_dir):
            logger.warning(f"Log directory not found: {self.log_dir}")
            return []

        for filename in os.listdir(self.log_dir):
            if not filename.endswith(".jsonl"):
                continue

            filepath = os.path.join(self.log_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        trajectories.append(record)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping malformed line in {filename}: {e}")

        logger.info(f"Loaded {len(trajectories)} trajectories from {self.log_dir}")
        return trajectories

    def _export_sft(self, scored: List[TrajectoryScore], output_file: str) -> int:
        """Export high-quality trajectories as SFT training data."""
        data = []
        for score in scored:
            data.append({
                "conversations": [
                    {"from": "system", "value": "You are an expert task-solving AI."},
                    {"from": "human", "value": score.original_request},
                    {"from": "gpt", "value": (
                        f"Completed successfully with {score.completed_steps} steps, "
                        f"{score.total_retries} retries, "
                        f"and ${score.total_cost_usd:.4f} cost."
                    )},
                ],
                "quality_score": score.composite_score,
            })

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported {len(data)} high-quality SFT examples: {output_file}")
        return len(data)
