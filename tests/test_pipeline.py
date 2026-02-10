"""Tests for celr.training.pipeline — Training orchestration."""

import json
import os
import pytest
import tempfile
import shutil

from celr.core.config import CELRConfig
from celr.training.pipeline import TrainingPipeline, TrainingReport
from tests.conftest import MockLLMProvider


def _create_test_log_dir(trajectories):
    """Create a temp dir with JSONL trajectory files."""
    tmp_dir = tempfile.mkdtemp()
    log_file = os.path.join(tmp_dir, "test_trajectories.jsonl")
    with open(log_file, "w", encoding="utf-8") as f:
        for traj in trajectories:
            f.write(json.dumps(traj) + "\n")
    return tmp_dir


class TestTrainingPipeline:
    def test_empty_log_dir(self):
        """Pipeline with no trajectories should return empty report."""
        config = CELRConfig(log_dir=tempfile.mkdtemp())
        pipeline = TrainingPipeline(config=config, output_dir=tempfile.mkdtemp())
        report = pipeline.run()
        assert report.trajectories_loaded == 0
        assert report.dpo_pairs_created == 0

    def test_pipeline_with_data(self):
        """Pipeline with trajectories should produce training data."""
        trajectories = [
            {
                "task_id": "t1", "original_request": "Good task",
                "final_status": "SUCCESS", "budget_limit_usd": 1.0,
                "total_cost_usd": 0.01,
                "plan": [{"id": "s1", "status": "COMPLETED", "retry_count": 0}],
            },
            {
                "task_id": "t2", "original_request": "Bad task",
                "final_status": "FAILED", "budget_limit_usd": 1.0,
                "total_cost_usd": 0.90,
                "plan": [{"id": "s1", "status": "FAILED", "retry_count": 3}],
            },
            {
                "task_id": "t3", "original_request": "Medium task",
                "final_status": "SUCCESS", "budget_limit_usd": 1.0,
                "total_cost_usd": 0.30,
                "plan": [
                    {"id": "s1", "status": "COMPLETED", "retry_count": 1},
                    {"id": "s2", "status": "COMPLETED", "retry_count": 0},
                ],
            },
            {
                "task_id": "t4", "original_request": "Another bad",
                "final_status": "FAILED", "budget_limit_usd": 1.0,
                "total_cost_usd": 0.95,
                "plan": [{"id": "s1", "status": "FAILED", "retry_count": 5}],
            },
        ]

        log_dir = _create_test_log_dir(trajectories)
        output_dir = tempfile.mkdtemp()

        try:
            config = CELRConfig(log_dir=log_dir)
            pipeline = TrainingPipeline(config=config, output_dir=output_dir)
            report = pipeline.run()

            assert report.trajectories_loaded == 4
            assert report.trajectories_scored == 4
            assert report.high_quality_count >= 1
        finally:
            shutil.rmtree(log_dir)
            shutil.rmtree(output_dir)

    def test_pipeline_with_self_reward(self):
        """Pipeline with LLM should also generate self-reward pairs."""
        trajectories = [
            {
                "task_id": "t1", "original_request": "Task A",
                "final_status": "SUCCESS", "budget_limit_usd": 1.0,
                "total_cost_usd": 0.01,
                "plan": [{"id": "s1", "status": "COMPLETED", "retry_count": 0}],
            },
            {
                "task_id": "t2", "original_request": "Task B",
                "final_status": "FAILED", "budget_limit_usd": 1.0,
                "total_cost_usd": 0.90,
                "plan": [{"id": "s1", "status": "FAILED", "retry_count": 3}],
            },
        ]

        log_dir = _create_test_log_dir(trajectories)
        output_dir = tempfile.mkdtemp()
        mock_llm = MockLLMProvider(
            response="Accuracy: 4\nCompleteness: 4\nEfficiency: 4\nTotal: 12"
        )

        try:
            config = CELRConfig(log_dir=log_dir)
            pipeline = TrainingPipeline(
                config=config, llm=mock_llm, output_dir=output_dir
            )
            report = pipeline.run()

            assert report.trajectories_loaded == 2
            assert report.scoring_cost_usd >= 0.0
        finally:
            shutil.rmtree(log_dir)
            shutil.rmtree(output_dir)


class TestTrainingReport:
    def test_summary(self):
        report = TrainingReport(
            trajectories_loaded=10,
            dpo_pairs_created=5,
            total_pairs_exported=5,
        )
        summary = report.summary()
        assert "10" in summary
        assert "5" in summary
        assert "Training Pipeline Report" in summary
