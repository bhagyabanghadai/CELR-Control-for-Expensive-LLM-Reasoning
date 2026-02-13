# Changelog

All notable changes to CELR will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-02-12

### Added
- Core execution engine with DAG-based planning
- Cost-aware model escalation (small → mid → large)
- LiteLLM integration for multi-provider support (OpenAI, Anthropic, Groq, DeepSeek, Ollama)
- Verifier with confidence scoring and specific Python error detection
- Self-reflection with smart retry logic (transient vs permanent errors)
- Adaptive Cortex (Phase 8): state extraction, meta-policy, promotion gate
- Training pipeline: trajectory scoring, DPO/SPIN data generation, self-reward scoring
- Benchmark suite with 12 standardized tasks and comparison runner
- CLI for quick task execution

### Fixed
- `CortexAction.RETy_COMPRESSED` typo that crashed policy decisions
- `StateExtractor` referencing non-existent model attributes
- `demo.py` missing `import os` and using non-existent `context.to_dict()`
- Duplicate import in `reasoning.py`
- `PromotionGate` using `print()` instead of logger

### Changed
- Verifier now rejects empty/None output (security improvement)
- Reflection uses error-type-aware retry decisions
- Config supports `o3-*`, `groq/*`, and `deepseek/*` model prefixes
- Package exports version and full public API via `celr.__init__`
