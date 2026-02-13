# CELR: Control for Expensive LLM Reasoning

> **A meta-brain for your AI agents.**

CELR is a local, cost-aware reasoning system that sits in front of powerful (and expensive) LLMs. It doesn't just pass your prompt to GPT-4. It thinks, plans, attempts to solve problems cheaply, and only escalates when necessary.

**Status:** 🚀 Production Ready (v0.1.0) | **Tests:** 85 Passing | **License:** MIT

## 💡 The Core Application

Most agent systems blindly call the smartest model available for every sub-task. This is slow and expensive.
CELR changes the equation:

1.  **Reason First:** Uses a small, local model (or cheap API) to analyze difficulty and decompose the task.
2.  **Budget Aware:** You set a budget (e.g., $0.50). CELR optimizes its strategy to stay within it.
3.  **Escalation Protocol:** Tries simple solutions first. If verified as incorrect or low-confidence, it escalates to "SOTA-class" models (Opus, GPT-4, etc.).
4.  **Verification:** Uses Code Execution and logic checks to verify answers locally before accepting them.
5.  **Self-Improvement:** Logs successful paths and trains itself to be smarter (Phase 7).

## 🚀 Getting Started

### 1. Prerequisites
*   Python 3.10+
*   (Optional) [Ollama](https://ollama.com/) running locally for the "Small Brain"

### 2. Installation
```bash
# Clone the repo
git clone https://github.com/bhagyabanghadai/CELR-Control-for-Expensive-LLM-Reasoning.git
cd CELR

# Install dependencies
pip install -r requirements.txt
```

### 3. Quick Start (Interactive Chat)
The easiest way to use CELR is the interactive chat interface. It auto-detects your API keys or local Ollama models.

1.  **Launch it:**
    *   **Windows:** Double-click `run_chat.bat`
    *   **Mac/Linux:** Run `./run_chat.sh`
    *   **Manual:** `python celr_chat.py`

2.  **Pick your AI:**
    *   Select **Ollama** (free, local)
    *   Or cloud models (GPT-4o, Claude, Groq)

3.  **Start chatting!**
    *   Type `/system You cover code like a pirate` to change persona.
    *   Type `/save` to save your conversation to `logs/chats/`.

### 4. Running with Real APIs (CLI)
For scripted or automation tasks, use the CLI directly:

1.  Copy `.env.example` to `.env`.
2.  Add your keys (`OPENAI_API_KEY=sk-...`).
3.  Run a specific task:
    ```bash
    python -m celr.cli "Write a snake game in Python" --budget 0.50
    ```


## 🧠 Phase 8: Adaptive Cortex (Meta-Learning Control) 🆕

**It’s not about better text. It’s about better decisions.**

Most agents fail because they don't know *when* to stop, *when* to verifying, or *how* to spend their budget.
CELR implements an **Offline RL Controller** (Adaptive Cortex) that learns a policy from your execution logs.

1.  **Observe:** Budget, Risk, Difficulty, History.
2.  **Decide:** Escalate? Recurse? Verify? Stop?
3.  **Learn:** Trained offline to maximize success while minimizing cost/retries.
4.  **Gate:** New policies are only deployed if they pass strict safety & efficiency benchmarks.

**No online training.** The system gets smarter at *controlling* the agent, not just writing prompts.

## 🏗 Architecture

```mermaid
graph TD
    User[User Prompt + Budget] --> Controller
    Controller --> Planner[Local Small Brain]
    Planner --> Plan[Execution Plan (DAG)]
    Plan --> Executor
    
    Executor -->|Easy Step| LocalLLM[Llama 3 / Mistral]
    Executor -->|Hard Step| SmartLLM[GPT-4o / Claude 3.5]
    
    LocalLLM --> Verifier
    SmartLLM --> Verifier
    
    Verifier -->|Success| Controller
    Verifier -->|Failure| Replanner[Refinement Strategy]
    Replanner --> Executor
    
    Controller -.->|Log Trajectory| TrainingPipe[Self-Improvement Pipeline]
    TrainingPipe -.->|DPO Data| FineTuner
```

## 🛠 Tech Stack

| Component | Technology |
| :--- | :--- |
| **Framework** | Python 3.10+, Pydantic |
| **LLM Interface** | LiteLLM (OpenAI, Anthropic, Ollama, vLLM, etc.) |
| **Orchestrator** | CELR Custom Engine |
| **Scoring** | LLM-as-Judge (Self-Reward) |
| **Optimization** | DPO / SPIN Data Generation |

## 📚 Research Concepts
*   **"System 2" Reasoning:** Recursive decomposition of tasks.
*   **Test-Time Compute:** Trading inference time for accuracy.
*   **Self-Correction:** Reflexion-style loops.
*   **Escalation:** Mixture-of-Agents routing.

## 📊 Benchmarking

CELR includes a benchmark suite to compare its pipeline vs direct LLM calls:

```bash
# See available tasks
python -m benchmarks.benchmark_runner --dry-run

# Run with GPT-4o-mini (requires OPENAI_API_KEY in .env)
python -m benchmarks.benchmark_runner --model gpt-4o-mini --budget 0.50

# Filter by difficulty
python -m benchmarks.benchmark_runner --difficulty easy
```

Results are saved to `benchmarks/results/` with accuracy, cost, and latency comparisons.

---
**License**: MIT
