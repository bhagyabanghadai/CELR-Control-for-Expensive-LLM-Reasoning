# ⚡ CPU/GPU Optimization Guide

If CELR is using too much VRAM or running slowly on your local machine, use these settings.

## 1. Reduce Context Window
High VRAM usage usually comes from the context window (memory of conversation).
By default, Ollama uses 4096 (or more). Reducing this drastically cuts VRAM usage.

**In your `.env` file:**
```bash
# Default is 4096. Try 2048 for 8GB VRAM cards.
CELR_OLLAMA_NUM_CTX=2048
```

## 2. Aggressive Model Unloading
Ollama keeps models loaded for 5 minutes by default. If you need to free GPU for other tasks (like games or rendering) immediately after CELR replies:

**In your `.env` file:**
```bash
# Unload model immediately after response
CELR_OLLAMA_KEEP_ALIVE=0

# Or keep for just 1 minute
CELR_OLLAMA_KEEP_ALIVE=1m
```

## 3. Use Quantized Models (Best Performance)
Ensure you are using 4-bit quantized models. They use 50% less RAM than 8-bit.
CELR defaults to `llama3.2` which is usually 4-bit, but verify you haven't pulled an `fp16` version.

```bash
# Run this to ensure you have the efficient version
ollama pull llama3.2:3b-instruct-q4_k_m
```

## 4. Troubleshooting
If you still crash:
1.  **Close other apps:** Web browsers use GPU acceleration.
2.  **Use Cloud Models:** Switch to `gpt-4o-mini` (Option 1 in chat). It costs pennies and uses 0% of your GPU.
