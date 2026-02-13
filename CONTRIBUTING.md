# Contributing to CELR

Thank you for your interest in contributing to CELR! This document outlines how to get started.

## Development Setup

```bash
# Clone the repo
git clone https://github.com/your-org/CELR-Control-for-Expensive-LLM-Reasoning.git
cd CELR-Control-for-Expensive-LLM-Reasoning

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install in development mode
pip install -e ".[dev]"
# Or: pip install -r requirements.txt
```

## Running Tests

```bash
# All tests
python -m pytest tests/ -v

# Quick check
python -m pytest tests/ -q

# Specific module
python -m pytest tests/test_verifier.py -v
```

## Running the Demo

```bash
# Dry-run demo (no API key needed)
python demo.py
```

## Running Benchmarks

```bash
# Dry-run: list benchmark tasks
python -m benchmarks.benchmark_runner --dry-run

# Run with real LLM (requires API key in .env)
python -m benchmarks.benchmark_runner --model gpt-4o-mini --budget 0.50

# Filter by difficulty
python -m benchmarks.benchmark_runner --difficulty easy
```

## Code Style

- **Type hints**: Required on all public methods
- **Docstrings**: Required on all classes and public methods
- **Logging**: Use `logging.getLogger(__name__)`, never `print()`
- **Exceptions**: Use custom exceptions from `celr.core.exceptions`
- **Config**: All env vars use `CELR_` prefix

## Architecture Overview

```
celr/
├── core/          # Core execution engine
│   ├── executor.py     # Main execution loop
│   ├── planner.py      # DAG-based task planning
│   ├── escalation.py   # Model routing
│   ├── verifier.py     # Output verification
│   └── reflection.py   # Failure analysis
├── cortex/        # Adaptive Cortex (Phase 8)
│   ├── state.py        # RL state extraction
│   ├── policy.py       # Meta-policy decisions
│   └── gatekeeper.py   # Policy promotion gate
└── training/      # Self-improvement pipeline
    ├── pipeline.py     # Training orchestration
    ├── scorer.py       # Trajectory scoring
    └── data_generator.py  # DPO/SPIN data generation
```

## Pull Request Guidelines

1. Write tests for new features
2. All 85+ existing tests must pass
3. Keep PRs focused — one feature per PR
4. Include a clear description of what changed and why
