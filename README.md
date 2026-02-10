# CELR: Control for Expensive LLM Reasoning

> **A meta-brain for your AI agents.**

CELR is a local, cost-aware reasoning system that sits in front of powerful (and expensive) LLMs. It doesn't just pass your prompt to GPT-4. It thinks, plans, attempts to solve problems cheaply, and only escalates when necessary.

**Status:** 🚧 Early Data / Prototype

## 💡 The Core Application

Most agent systems blindly call the smartest model available for every sub-task. This is slow and expensive.
CELR changes the equation:

1.  **Reason First:** Uses a small, local model (or cheap API) to analyze difficulty and decompose the task.
2.  **Budget Aware:** You set a budget (e.g., $0.50). CELR optimizes its strategy to stay within it.
3.  **Escalation Protocol:** Tries simple solutions first. If verified as incorrect or low-confidence, it escalates to "SOTA-class" models (Opus, GPT-4, etc.).
4.  **Verification:** Uses Code Execution and logic checks to verify answers locally before accepting them.

## 🏗 Architecture

```mermaid
graph TD
    User[User Prompt + Budget] --> Controller
    Controller --> Planner[Local "Small" Brain]
    Planner --> Plan[Execution Plan (DAG)]
    Plan --> Executor
    
    Executor -->|Easy Step| LocalLLM[Llama 3 / Mistral]
    Executor -->|Hard Step| SmartLLM[GPT-4o / Claude 3.5]
    
    LocalLLM --> Verifier
    SmartLLM --> Verifier
    
    Verifier -->|Success| Controller
    Verifier -->|Failure| Replanner[Refinement Strategy]
    Replanner --> Executor
```

## 🚀 Getting Started

### Prerequisites
*   Python 3.10+
*   (Optional) Ollama running locally for the "Small Brain"
*   API Keys for "Big Brains" (OpenAI, Anthropic)

### Usage

```bash
# Clone the repo
git clone https://github.com/bhagyabanghadai/CELR-Control-for-Expensive-LLM-Reasoning.git
cd CELR-Control-for-Expensive-LLM-Reasoning

# Install dependencies
pip install -r requirements.txt

# Run the experimental CLI
python -m celr.main "Analyze this data file and find the outliers" --budget 0.20
```

## 🧠 Design Philosophy

*   **Recursive Reasoning:** Small models can equal big models if given time to think and refine.
## 🛠 Tech Stack & Open Source Status

| Component | Technology | License / Status |
| :--- | :--- | :--- |
| **Framework** | Python 3.10+ | Open Source (PSF) |
| **Orchestrator** | CELR (this repo) | Open Source (MIT) | 
| **Small Brain** | Llama 3 / Mistral (via Ollama) | Open Weights (Apache 2.0 / Llama Community) |
| **Big Brain** | GPT-4o / Claude 3.5 | **Proprietary APIs** (Paid) |
| **Big Brain (Local Alt)** | Llama-3-70B (via vLLM) | Open Weights (Llama Community) |

**Note:** The system is designed to be 100% Open Source compatible if you have the hardware to run a large local model (e.g., 70B+ parameters) as your "Big Brain", otherwise it defaults to paid APIs for the high-intelligence capability.

## 📚 Research & Concept Attribution

The *ideas* behind this system are based on open research papers and community concepts. They are public knowledge, not proprietary software.

1.  **"Tiny Recursive Models" (TRM) / "System 2":** Inspired by Andrej Karpathy's "Baby Llama" experiments and the *Chain of Thought* papers (Wei et al., 2022).
2.  **Test-Time Compute:** Based on the principle that "more inference time = better reasoning", popularized by OpenAI's *o1* technical reports and various "Self-Consistency" papers (Wang et al., 2022).
3.  **Self-Correction / Verification:** Loosely based on *Reflexion* (Shinn et al., 2023) and *Language Models Can Solve Computer Tasks* (Kim et al., 2023).
4.  **Escalation / Routing:** Common pattern in "Mixture of Agents" and "Cascade" architectures (e.g., FrugalGPT).

We are **implementing** these open concepts in a new, unified framework. We are not "stealing" closed code; we are building on shared scientific knowledge.

## ❓ FAQ

### Q: Do I need to train a model?
**A: No.**


2.  **Day 100 (Optional):** If you *want* to make the small model faster and smarter, the system can save its own successful reasoning chains. You can then use this **self-generated data** to fine-tune a local model. **You do not need external datasets.**

## 🔌 Compatible Models

Yes, CELR is **model-agnostic**. We use `LiteLLM` under the hood, which means it works with almost **any** LLM provider or local runner.

*   **100% Local (Free):**
    *   **Ollama** (Llama 3, Mistral, Gemma, Phi-3)
    *   **LM Studio** (Any GGUF model)
    *   **vLLM** (Production-grade local serving)
    *   **LocalAI**
*   **Cloud (Paid/Free Tiers):**
    *   **OpenAI** (GPT-4o, GPT-3.5)
    *   **Anthropic** (Claude 3.5 Sonnet, Haiku)
    *   **Groq** (Llama 3 70B at 800 tokens/sec)
    *   **DeepSeek** (DeepSeek Coder / Chat)
    *   **Together AI**, **Mistral API**, **Cohere**...

You just change one line in the `.env` file to switch brains.




