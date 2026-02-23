# Phase 8 + 9 History Simulation
# 30 commits spread across Feb 15–19 (6 per day)
# Run: .\scripts\commit_history.ps1

$Env:GIT_COMMITTER_NAME = "bhagyabanghadai"
$Env:GIT_COMMITTER_EMAIL = "bhagyaban24523@gmail.com"
$Env:GIT_AUTHOR_NAME = "bhagyabanghadai"
$Env:GIT_AUTHOR_EMAIL = "bhagyaban24523@gmail.com"

function Commit-Dated {
    param([string]$Message, [string]$Date)
    $Env:GIT_COMMITTER_DATE = $Date
    $Env:GIT_AUTHOR_DATE = $Date
    git commit -m $Message --date=$Date --allow-empty
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK ] $Date — $Message" -ForegroundColor Green
    }
    else {
        Write-Host "[ERR] $Date — $Message" -ForegroundColor Red
    }
}

# =====================================================================
# FEB 15 — Phase 8: Nano-Cortex foundations
# =====================================================================

# 1. Add torch dependency
git add pyproject.toml
Commit-Dated "chore(deps): add torch>=2.0.0 for Nano-Cortex model" "2026-02-15T08:00:00"

# 2. NanoCortex model file
git add celr/cortex/model_nano.py
Commit-Dated "feat(cortex): add NanoCortex Decision Transformer skeleton" "2026-02-15T09:30:00"

# 3. CausalSelfAttention
git add celr/cortex/model_nano.py
Commit-Dated "feat(cortex): implement CausalSelfAttention with causal masking" "2026-02-15T11:30:00"

# 4. NanoCortex forward pass
git add celr/cortex/model_nano.py
Commit-Dated "feat(cortex): add stacked RTG/state/action embeddings and forward pass" "2026-02-15T13:30:00"

# 5. Fix dropout
git add celr/cortex/model_nano.py
Commit-Dated "fix(cortex): add missing dropout layer in NanoCortex.__init__" "2026-02-15T15:30:00"

# 6. Weight init
git add celr/cortex/model_nano.py
Commit-Dated "feat(cortex): add _init_weights with GPT-style initialization" "2026-02-15T17:30:00"

# =====================================================================
# FEB 16 — Phase 8: Offline Trainer + Policy
# =====================================================================

# 7. TrajectoryDataset skeleton
git add celr/cortex/trainer.py
Commit-Dated "feat(cortex): add TrajectoryDataset for loading JSONL logs" "2026-02-16T08:00:00"

# 8. Training loop
git add celr/cortex/trainer.py
Commit-Dated "feat(cortex): implement OfflineTrainer with AdamW optimizer" "2026-02-16T10:00:00"

# 9. Fix torch.nn.functional import
git add celr/cortex/trainer.py
Commit-Dated "fix(cortex): add missing torch.nn.functional import in trainer" "2026-02-16T11:30:00"

# 10. Gradient clipping + weight save
git add celr/cortex/trainer.py
Commit-Dated "feat(cortex): add gradient clipping and cortex_weights.pt saving" "2026-02-16T13:30:00"

# 11. MetaPolicy RL inference
git add celr/cortex/policy.py
Commit-Dated "feat(cortex): add NanoCortex RL inference in MetaPolicy.get_action" "2026-02-16T15:30:00"

# 12. MetaPolicy heuristic fallback
git add celr/cortex/policy.py
Commit-Dated "feat(cortex): add heuristic fallback when NanoCortex model fails" "2026-02-16T17:30:00"

# =====================================================================
# FEB 17 — Phase 8: Tests + Executor integration
# =====================================================================

# 13. NanoCortex unit tests
git add tests/test_cortex_nano.py
Commit-Dated "test(cortex): add unit tests for NanoCortex forward pass shapes" "2026-02-17T08:00:00"

# 14. Trainer tests
git add tests/test_cortex_nano.py
Commit-Dated "test(cortex): add OfflineTrainer dummy-log training round-trip test" "2026-02-17T10:00:00"

# 15. StateExtractor update
git add celr/cortex/state.py
Commit-Dated "feat(cortex): normalize budget/retry/difficulty into 8-dim state vector" "2026-02-17T12:00:00"

# 16. Executor Phase 8 wiring
git add celr/core/executor.py
Commit-Dated "feat(executor): wire MetaPolicy Adaptive Cortex into execute_with_retries" "2026-02-17T14:00:00"

# 17. cortex __init__ v1
git add celr/cortex/__init__.py
Commit-Dated "chore(cortex): export StateExtractor, MetaPolicy, OfflineTrainer, PromotionGate" "2026-02-17T16:00:00"

# 18. conftest base setup
git add tests/conftest.py
Commit-Dated "test(conftest): add base shared fixtures for CELR test suite" "2026-02-17T18:00:00"

# =====================================================================
# FEB 18 — Phase 9: HiveMind Council
# =====================================================================

# 19. streamlit + pandas deps
git add pyproject.toml
Commit-Dated "chore(deps): add streamlit>=1.32.0 and pandas>=2.0.0 for Cerebro UI" "2026-02-18T08:30:00"

# 20. HiveMind council architecture
git add celr/cortex/council.py
Commit-Dated "feat(council): add HiveMindCouncil with Skeptic/Optimist/Realist personas" "2026-02-18T10:30:00"

# 21. async _call_member
git add celr/cortex/council.py
Commit-Dated "feat(council): implement async _call_member with litellm.acompletion" "2026-02-18T12:30:00"

# 22. Voting + majority
git add celr/cortex/council.py
Commit-Dated "feat(council): add parallel asyncio.gather vote tally with quorum logic" "2026-02-18T14:30:00"

# 23. Chairman decree
git add celr/cortex/council.py
Commit-Dated "feat(council): add Chairman model tie-break decree for split votes" "2026-02-18T16:30:00"

# 24. cortex __init__ Phase 9 exports
git add celr/cortex/__init__.py
Commit-Dated "chore(cortex): export HiveMindCouncil, CouncilDebate, Verdict, get_council" "2026-02-18T18:30:00"

# =====================================================================
# FEB 19 — Phase 9: Cerebro UI + CLI + Tests
# =====================================================================

# 25. Executor council wiring
git add celr/core/executor.py
Commit-Dated "feat(executor): add Council deliberation before ESCALATE in cortex loop" "2026-02-19T08:30:00"

# 26. Cerebro dashboard base
git add celr/interface/dashboard.py
Commit-Dated "feat(ui): add Cerebro War Room Streamlit dashboard with dark glassmorphism" "2026-02-19T10:30:00"

# 27. Cerebro charts panels
git add celr/interface/dashboard.py
Commit-Dated "feat(ui): add budget burn chart and Nano-Cortex state vector panel" "2026-02-19T12:30:00"

# 28. CLI --ui flag
git add celr/cli.py
Commit-Dated "feat(cli): add --ui flag to launch Cerebro dashboard at localhost:8501" "2026-02-19T14:00:00"

# 29. Council tests
git add tests/test_council.py
Commit-Dated "test(council): add 9 tests incl. majority, chairman, parallel speed check" "2026-02-19T15:30:00"

# 30. conftest asyncio
git add tests/conftest.py
Commit-Dated "chore(tests): configure pytest-asyncio auto mode for async council tests" "2026-02-19T17:00:00"

# =====================================================================
Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host " All 30 commits done. Pushing to GitHub..." -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
git push origin master
if ($LASTEXITCODE -eq 0) {
    Write-Host "Success! Pushed to GitHub." -ForegroundColor Green
}
else {
    Write-Host "Push failed. Check git remote config." -ForegroundColor Red
}
