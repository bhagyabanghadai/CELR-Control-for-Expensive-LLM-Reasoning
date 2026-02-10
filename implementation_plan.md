# Phase 7: Training Pipeline Integration (IMPLEMENTED ✅)

CELR's self-improvement loop: successful execution trajectories are scored, turned into preference pairs, and used to fine-tune the small model so it gets better over time — without human annotation.

CELR's self-improvement loop: successful execution trajectories are scored, turned into preference pairs, and used to fine-tune the small model so it gets better over time — without human annotation.

Inspired by three research lines:
- **SPIN** (UCLA): Self-play where model discriminates its own outputs from real SFT data
- **Self-Rewarding LM** (Meta): LLM scores its own outputs → DPO training
- **TinyRecursiveModels** (Samsung): Recursive refinement (already in CELR's verify→reflect→retry loop)

## Proposed Changes

### New Package: `celr/training/`

#### [NEW] [\_\_init\_\_.py](file:///F:/LLM%20CELR/celr/training/__init__.py)
Re-exports for clean imports.

---

#### [NEW] [scorer.py](file:///F:/LLM%20CELR/celr/training/scorer.py)
**Trajectory quality scorer.** Scores completed trajectories on multiple dimensions:
- `success_score`: 1.0 if SUCCESS, 0.0 if FAILED
- `efficiency_score`: fewer retries = higher score (1.0 / (1 + retry_count))
- `cost_score`: lower cost = higher score (budget_remaining / budget_limit)
- `composite_score`: weighted combination → single 0.0–1.0 quality metric

```python
class TrajectoryScorer:
    def score(self, trajectory: dict) -> TrajectoryScore
    def rank(self, trajectories: List[dict]) -> List[TrajectoryScore]
```

---

#### [NEW] [data_generator.py](file:///F:/LLM%20CELR/celr/training/data_generator.py)
**Preference pair generator for DPO/SPIN training.** Creates `(chosen, rejected)` pairs from trajectory data:

1. **DPO pairs**: High-scoring trajectory = chosen, low-scoring = rejected (same task)
2. **SPIN pairs**: Real SFT data = chosen, model-generated response = rejected
3. Export formats: ShareGPT (Axolotl), DPO pairs (TRL), SPIN pairs

```python
class PreferencePairGenerator:
    def create_dpo_pairs(self, scored: List[TrajectoryScore]) -> List[PreferencePair]
    def create_spin_pairs(self, sft_data: List, model_responses: List) -> List[PreferencePair]
    def export_trl_format(self, pairs: List[PreferencePair], output: str) -> int
    def export_sharegpt_format(self, pairs: List[PreferencePair], output: str) -> int
```

---

#### [NEW] [self_reward.py](file:///F:/LLM%20CELR/celr/training/self_reward.py)
**LLM-as-Judge scoring.** The model rates its own outputs on a 1–5 scale, creating reward signals without human annotation:

```python
class SelfRewardScorer:
    def __init__(self, llm: BaseLLMProvider)
    def score_response(self, prompt: str, response: str) -> float  # 0.0-1.0
    def score_trajectory(self, trajectory: dict) -> float
    def generate_reward_pairs(self, trajectories: List[dict]) -> List[PreferencePair]
```

Uses a structured prompt: "Rate this response on accuracy, completeness, and efficiency (1-5)."

---

#### [NEW] [pipeline.py](file:///F:/LLM%20CELR/celr/training/pipeline.py)
**Training orchestration.** Runs the full self-improvement cycle:

```
Step 1: Collect → Load successful trajectories from .celr_logs/
Step 2: Score   → Score each trajectory with TrajectoryScorer + SelfRewardScorer  
Step 3: Pair    → Generate DPO/SPIN preference pairs
Step 4: Export  → Save to training format (ShareGPT/TRL/SPIN)
Step 5: Report  → Print stats (total pairs, quality distribution)
```

```python
class TrainingPipeline:
    def __init__(self, config: CELRConfig, llm: BaseLLMProvider)
    def run(self) -> TrainingReport
```

---

#### [MODIFY] [trainer.py](file:///F:/LLM%20CELR/celr/core/trainer.py)
Update existing trainer to use the new pipeline components. Add `export_to_dpo()` and `export_to_spin()` methods alongside existing `export_to_sharegpt()`.

---

### Tests

#### [NEW] [test_scorer.py](file:///F:/LLM%20CELR/tests/test_scorer.py)
Test scoring logic: success/failure trajectories, efficiency calculation, cost scoring, composite ranking.

#### [NEW] [test_data_generator.py](file:///F:/LLM%20CELR/tests/test_data_generator.py)
Test DPO pair creation, SPIN pair creation, export formats.

#### [NEW] [test_self_reward.py](file:///F:/LLM%20CELR/tests/test_self_reward.py)
Test LLM-as-Judge scoring with mock LLM, reward pair generation.

#### [NEW] [test_pipeline.py](file:///F:/LLM%20CELR/tests/test_pipeline.py)
Test end-to-end pipeline with mock data.

---

## Verification Plan

### Automated Tests
```bash
python -m pytest tests/test_scorer.py tests/test_data_generator.py tests/test_self_reward.py tests/test_pipeline.py -v
```

### Manual Verification
- Create sample trajectory data, run pipeline, inspect exported files
- Verify DPO pair format matches TRL expectations
- Verify ShareGPT format matches Axolotl expectations
