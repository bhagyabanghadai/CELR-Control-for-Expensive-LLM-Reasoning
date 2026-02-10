"""
CELR Training Pipeline — Self-improvement through trajectory learning.

Implements ideas from:
  - SPIN (Self-Play Fine-Tuning) — UCLA
  - Self-Rewarding Language Models — Meta AI
  - Tiny Recursive Models — Samsung SAIL
"""

from celr.training.scorer import TrajectoryScorer, TrajectoryScore
from celr.training.data_generator import PreferencePairGenerator, PreferencePair
from celr.training.self_reward import SelfRewardScorer
from celr.training.pipeline import TrainingPipeline

__all__ = [
    "TrajectoryScorer",
    "TrajectoryScore",
    "PreferencePairGenerator",
    "PreferencePair",
    "SelfRewardScorer",
    "TrainingPipeline",
]
