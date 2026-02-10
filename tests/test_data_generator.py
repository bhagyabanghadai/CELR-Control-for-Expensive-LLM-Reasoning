"""Tests for celr.training.data_generator — Preference pair generation."""

import json
import os
import pytest
import tempfile

from celr.training.scorer import TrajectoryScore
from celr.training.data_generator import PreferencePairGenerator, PreferencePair


def _make_scored(tid, request, score, status="SUCCESS"):
    return TrajectoryScore(
        trajectory_id=tid,
        original_request=request,
        final_status=status,
        composite_score=score,
        total_retries=0,
        total_cost_usd=0.01,
        total_steps=3,
        completed_steps=3 if status == "SUCCESS" else 1,
    )


class TestDPOPairs:
    def test_creates_pairs(self):
        gen = PreferencePairGenerator(min_score_gap=0.1)
        scored = [
            _make_scored("t1", "Task A", 0.95),
            _make_scored("t2", "Task B", 0.80),
            _make_scored("t3", "Task C", 0.40, "FAILED"),
            _make_scored("t4", "Task D", 0.20, "FAILED"),
        ]
        pairs = gen.create_dpo_pairs(scored)

        assert len(pairs) > 0
        for pair in pairs:
            assert pair.chosen_score > pair.rejected_score
            assert pair.pair_type == "dpo"

    def test_min_score_gap(self):
        gen = PreferencePairGenerator(min_score_gap=0.5)
        scored = [
            _make_scored("t1", "A", 0.60),
            _make_scored("t2", "B", 0.55),
        ]
        pairs = gen.create_dpo_pairs(scored)
        assert len(pairs) == 0  # Gap too small

    def test_not_enough_trajectories(self):
        gen = PreferencePairGenerator()
        pairs = gen.create_dpo_pairs([_make_scored("t1", "A", 0.9)])
        assert len(pairs) == 0

    def test_empty_input(self):
        gen = PreferencePairGenerator()
        pairs = gen.create_dpo_pairs([])
        assert len(pairs) == 0


class TestSPINPairs:
    def test_creates_spin_pairs(self):
        gen = PreferencePairGenerator()
        sft = [
            {"prompt": "What is 2+2?", "response": "4"},
            {"prompt": "Capital of France?", "response": "Paris"},
        ]
        model = [
            {"prompt": "What is 2+2?", "response": "22"},
            {"prompt": "Capital of France?", "response": "Lyon"},
        ]
        pairs = gen.create_spin_pairs(sft, model)

        assert len(pairs) == 2
        assert pairs[0].chosen == "4"
        assert pairs[0].rejected == "22"
        assert pairs[0].pair_type == "spin"

    def test_unmatched_prompts(self):
        gen = PreferencePairGenerator()
        sft = [{"prompt": "A", "response": "answer A"}]
        model = [{"prompt": "B", "response": "answer B"}]
        pairs = gen.create_spin_pairs(sft, model)
        assert len(pairs) == 0


class TestExport:
    def test_export_trl_format(self):
        gen = PreferencePairGenerator()
        pairs = [
            PreferencePair(prompt="Q", chosen="good", rejected="bad"),
        ]
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name

        try:
            count = gen.export_trl_format(pairs, path)
            assert count == 1

            with open(path, "r") as f:
                data = json.load(f)
            assert len(data) == 1
            assert "prompt" in data[0]
            assert "chosen" in data[0]
            assert "rejected" in data[0]
        finally:
            os.unlink(path)

    def test_export_sharegpt_format(self):
        gen = PreferencePairGenerator()
        pairs = [
            PreferencePair(prompt="Q", chosen="good", rejected="bad"),
        ]
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name

        try:
            count = gen.export_sharegpt_format(pairs, path)
            assert count == 1

            with open(path, "r") as f:
                data = json.load(f)
            assert "conversations" in data[0]
            assert "rejected_conversations" in data[0]
        finally:
            os.unlink(path)
