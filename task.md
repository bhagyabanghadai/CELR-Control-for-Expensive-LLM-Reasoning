# Market-Ready CELR Overhaul — Task Tracker

## Overhaul Phases (O-1 through O-6) ✅ ALL COMPLETE

### Phase O-1: Foundation Fixes
- [x] Create `celr/core/__init__.py` with re-exports <!-- id: 18 -->
- [x] Create `celr/core/exceptions.py` with custom hierarchy <!-- id: 19 -->
- [x] Add `EscalationTier` + retry fields to `types.py` <!-- id: 20 -->

### Phase O-2: Error Handling & Resilience
- [x] Rewrite `llm.py` — `(text, usage)` return, tenacity retry <!-- id: 21 -->
- [x] Fix bare `except:` in `reasoning.py` and `trainer.py` <!-- id: 22 -->
- [x] Add safety to `tools.py` (builtins whitelist, output limit, timeout) <!-- id: 23 -->

### Phase O-3: Wire the Full Loop
- [x] Refactor `executor.py` — inject Verifier, Reflection, ToolRegistry <!-- id: 24 -->
- [x] Replace mock execution with real LLM dispatch + tool calls <!-- id: 25 -->
- [x] Add verify-after-execute and retry-with-reflection loop <!-- id: 26 -->
- [x] Refactor `escalation.py` — configurable models, `get_provider()` <!-- id: 27 -->

### Phase O-4: Configuration System
- [x] Create `celr/core/config.py` with Pydantic `BaseSettings` <!-- id: 28 -->
- [x] Update `cli.py` — wire verbose, config file, Rich progress bars <!-- id: 29 -->

### Phase O-5: Structured Logging
- [x] Replace `context.log()`/`print()` with Python `logging` across all files <!-- id: 30 -->

### Phase O-6: Comprehensive Test Suite
- [x] Create `tests/conftest.py` with shared fixtures <!-- id: 31 -->
- [x] Write unit tests for types, cost_tracker, escalation, reasoning <!-- id: 32 -->
- [x] Write unit tests for executor, verifier, reflection, tools, trainer <!-- id: 33 -->
- [x] Run `pytest --cov=celr --cov-fail-under=80` and ensure all pass <!-- id: 34 -->

### Sign-Off
- [x] Write final walkthrough.md with proof-of-work <!-- id: 35 -->
- [x] Commit all changes to GitHub in incremental commits <!-- id: 36 -->

---

## Phase 7: Training Pipeline Integration 🚀

### 7A: Trajectory Scoring
- [x] Create `celr/training/__init__.py` with re-exports <!-- id: 40 -->
- [x] Create `celr/training/scorer.py` — TrajectoryScorer with composite scoring <!-- id: 41 -->

### 7B: DPO/SPIN Data Generation
- [x] Create `celr/training/data_generator.py` — PreferencePairGenerator <!-- id: 42 -->
- [x] DPO pairs: high-score=chosen, low-score=rejected from same task <!-- id: 43 -->
- [x] SPIN pairs: real SFT=chosen, model-generated=rejected <!-- id: 44 -->

### 7C: Self-Reward Integration (LLM-as-Judge)
- [x] Create `celr/training/self_reward.py` — SelfRewardScorer <!-- id: 45 -->

### 7D: Training Orchestration
- [x] Create `celr/training/pipeline.py` — TrainingPipeline <!-- id: 46 -->
- [x] Update `celr/core/trainer.py` with `export_to_dpo()` + `export_to_spin()` <!-- id: 47 -->

### 7E: Tests
- [x] Create `tests/test_scorer.py` <!-- id: 48 -->
- [x] Create `tests/test_data_generator.py` <!-- id: 49 -->
- [x] Create `tests/test_self_reward.py` <!-- id: 50 -->
- [x] Create `tests/test_pipeline.py` <!-- id: 51 -->

### 7F: Sign-Off
- [x] Run all tests, commit, push to GitHub <!-- id: 52 -->

## Phase 8: Live Testing & Scripts 🚀
- [x] Create `demo.py` (Mock LLM) and fix bugs
- [x] Create `run_demo.bat` (Dry Run)
- [x] Create `run_ollama.bat` (Local LLM Execution)
- [x] Create `train_ollama.bat` (Self-Improvement Pipeline)
- [x] Verify end-to-end execution with local Llama 3
- [x] Update documentation (README, Walkthrough)

## Phase 8: Adaptive Cortex (Meta-Learning Control) 🧠
- [x] Create `celr/cortex/` package
- [x] Implement `StateExtractor` (logs -> RL state vector)
- [x] Implement `MetaPolicy` (Rule-based baseline + RL support)
- [x] Implement `OfflineTrainer` (Compute rewards from logs)
- [x] Implement `PromotionGate` (Benchmark suite validation)
- [x] Integrate Cortex into `Executor` (Control loop)
- [x] Verify Policy Promotion (Train on logs -> Pass Gate -> Run)
- [x] Add "Adaptive Cortex" section to README
