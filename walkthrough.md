# CELR Walkthrough & Demo

This guide shows you how to use your new "Meta-Brain" system.

## 1. Setup

First, install the package in editable mode:
```bash
pip install -e .
```

Copy the environment example and set your keys:
```bash
cp .env.example .env
# Edit .env and allow at least OPENAI_API_KEY
```

## 2. Running a Task

Run the CLI with a budget and a complex task:
```bash
python -m celr.cli "Analyze the trade-offs between Rust and C++ for embedded systems" --budget 0.20
```

### What Happens Internally?
1.  **Thinking Phase:** The local model (or cheap API) decomposes this into:
    *   Research Rust features.
    *   Research C++ features.
    *   Compare safety vs performance.
    *   Synthesize report.
2.  **Planning Phase:** A DAG (Directed Acyclic Graph) is created.
3.  **Execution Loop:** 
    *   It tries to answer each step using cheap models.
    *   If a step is "Hard" (difficulty > 0.7) and budget allows, it **escalates** to GPT-4o.
4.  **Logging:** The full trace is saved to `.celr_logs/` for future training.

## 3. Training (Self-Improvement)

After using the system for a few days, you can export your data to train a better small model:

```python
from celr.core.trainer import Trainer

trainer = Trainer()
count = trainer.export_to_sharegpt("my_training_data.json")
print(f"Exported {count} successful reasoning chains!")
```
