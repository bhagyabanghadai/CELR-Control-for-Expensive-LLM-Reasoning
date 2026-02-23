# 🧠 The Science Behind CELR (Research Inspirations)

This file documents the key research papers and academic concepts that form the cognitive architecture of **CELR** (Control for Expensive LLM Reasoning).

## 1. **Meta (Facebook AI Research)**
### **"Self-Rewarding Language Models" (Yuan et al., 2024)**

*   **Concept:**
    Standard RLHF (Reinforcement Learning from Human Feedback) is expensive and slow because it relies on humans. This paper proposes that LLMs can act as their own reward models, judging the quality of their own outputs during training to improve iteratively.
*   **CELR Implementation:**
    *   **Module:** `celr.training.self_reward.SelfRewardScorer`
    *   **How it works:** We implemented an "LLM-as-Judge" system where the model rates its own execution trajectories on Accuracy (1-5), Completeness (1-5), and Efficiency (1-5).
    *   **Code:** The `score_response()` method uses a specific prompt template to generate these self-supervised rewards, creating a dataset for future fine-tuning without human labelers.

## 2. **Northeastern University / MIT**
### **"Reflexion: Language Agents with Verbal Reinforcement Learning" (Shinn et al., 2023)**

*   **Concept:**
    Instead of just updating weights (like traditional RL), agents can improve by verbally reflecting on their mistakes. When a task fails, the agent generates a "reflection" summary of *why* it failed and stores it in memory context for the next attempt.
*   **CELR Implementation:**
    *   **Module:** `celr.core.reflection.SelfReflection`
    *   **How it works:** The `_execute_with_retries` loop in `executor.py` catches failures. It then calls the `Reflection` module, which analyzes the error trace and injects a "Constructive Criticism" prompt into the context for the next retry attempt. This turns a "crash" into a "learning moment."

## 3. **LMSYS / UC Berkeley**
### **"RouteLLM: Learning to Route LLMs with Preference Data" (Ong et al., 2024)**

*   **Concept:**
    Using a single large model (like GPT-4) for everything is wasteful. A "Router" can predict the difficulty of a prompt and dynamically assign it to a cheaper model (like Llama 3) or an expensive model (GPT-4) to optimize cost-performance trade-offs.
*   **CELR Implementation:**
    *   **Module:** `celr.cortex.router.Router` & `celr.cortex.policy.MetaPolicy`
    *   **How it works:** Our **Adaptive Cortex** calculates a "State Vector" (Difficulty, Depth, Budget Pressure). The Router uses this valid signal to route simple steps to `ollama/llama3.2` and complex/risky steps to `gpt-4o`.
    *   **Result:** We achieve near-GPT-4 performance at a fraction of the cost.

## 4. **OpenAI / "Process Supervision"**
### **"Let's Verify Step by Step" (Lightman et al., 2023)**

*   **Concept:**
    Outcome supervision (checking only the final answer) is insufficient for complex reasoning. "Process supervision" involves verifying *each individual step* of a chain-of-thought to catch errors early.
*   **CELR Implementation:**
    *   **Module:** `celr.core.verifier.Verifier`
    *   **How it works:** Every single step in the `Plan` is subjected to a verification pass. For code steps, we run the code and check stderr. For reasoning steps, we use a lightweight "Critic" model to sanity-check the logic before proceeding. This prevents error cascading.

## 5. **Google DeepMind**
### **"Tree of Thoughts: Deliberate Problem Solving" (Yao et al., 2023)**

*   **Concept:**
    Standard prompting is linear (Chain of Thought). Difficult problems require exploring multiple possible branches/options and backtracking if a path looks unpromising.
*   **CELR Implementation:**
    *   **Module:** `celr.core.planner.Planner` & `celr.core.executor.TaskExecutor`
    *   **How it works:** Explicit Planning Step. The implementation of `TaskStatus.FAILED` triggering a re-planning or backtracking (though currently linear-with-retries in Phase 1, the architecture supports tree expansion). The **Council** mechanism (Phase 9) also simulates the "multiple thoughts" aspect by generating 3 distinct critiques in parallel.

## 6. **Samsung (SR-Lab)**
### **"Tiny Recursive Models (TRM)"** (Conceptual)

*   **Concept:**
    Deploying massive LLMs recursively is inefficient. Instead, use a **Tiny** specialized model (TRM) that runs recursively at every step to guide the larger model's execution. This acts as a lightweight "prefrontal cortex" for the heavy "limbic system" LLM.
*   **CELR Implementation:**
    *   **Module:** `celr.cortex.model_nano` (The **Nano-Cortex**)
    *   **How it works:** Our Nano-Cortex is a tiny neural network (or small LLM) that runs *recursively* before every single action. It analyzes the state vector and decides whether to *Proceed*, *Escalate*, or *Abort*. It is the direct implementation of the "Tiny Recursive Model" philosophy: high-frequency, low-latency control.

## 7. **Constitutional AI (Anthropic)**
### **"Constitutional AI: Harmlessness from AI Feedback" (Bai et al., 2022)**

*   **Concept:**
    Using a set of principles (a "Constitution") to guide AI behavior rather than just human labels.
*   **CELR Implementation:**
    *   **Module:** `celr.cortex.council.HiveMindCouncil`
    *   **How it works:** The "Personas" (Skeptic, Optimist, Realist) act as a distributed constitution. They enforce specific behavioral guardrails (Safety, Cost, Feasibility) during the high-stakes escalation debates.

## 8. **DeepMind / "Test-Time Compute"**
### **"Scaling Laws for Test-Time Compute"**

*   **Concept:**
    Allowing a model to "think longer" (spend more tokens/time) during inference can improve performance on hard tasks more than just increasing model size.
*   **CELR Implementation:**
    *   **Module:** `celr.core.executor.TaskExecutor`
    *   **How it works:** Our **Budget-Aware Recursion** and **Reflexion Loop** dynamicially allocate more compute (retries, reflection steps) to harder problems, while fast-tracking easy ones.

## 9. **UCLA / SPIN**
### **"Self-Play Fine-Tuning (SPIN)"**

*   **Concept:**
    Iterative self-play where the LLM generates data, and a reward model selects the best trajectories to fine-tune the LLM, creating a virtuous cycle.
*   **CELR Implementation:**
    *   **Module:** `celr.training.data_generator.PreferencePairGenerator`
    *   **How it works:** We generate **SPIN Pairs** (`create_spin_pairs`) where the "Chosen" response is a high-quality trajectory (often from a stronger model or successful run) and the "Rejected" is a failed run. This creates the exact dataset needed for SPIN/DPO training.

## 10. **Meta-RL (DynaMITE / General)**
### **"Meta-Reinforcement Learning"**

*   **Concept:**
    Learning a policy that can adapt to new tasks by managing the agent's own learning process or strategy selection.
*   **CELR Implementation:**
    *   **Module:** `celr.cortex.policy` & `celr.cortex.trainer`
    *   **How it works:** The **Adaptive Cortex** is a meta-controller. It doesn't solve the task itself; it solves the problem of "How should I solve this task?" (Which model? Which path?). It learns from offline logs (`celr.cortex.trainer.OfflineTrainer`) to optimize this decision-making policy.

## 11. **Online Self-Reward (Quality Gating)**
*   **Concept:** Using the model as its own judge to reject low-quality reasoning branches in real-time.
*   **CELR Implementation:**
    *   **Module:** `celr.core.executor.TaskExecutor` & `celr.training.self_reward.SelfRewardScorer`
    *   **Status:** **🟢 ACTIVE (ONLINE)**. 
    *   **How it works:** After output generation, the `SelfRewardScorer` performs a recursive check. If the score is `< 0.6`, the output is discarded and the `Reflexion` loop is triggered. This forces the model to "think harder" until it meets its own quality standards.
