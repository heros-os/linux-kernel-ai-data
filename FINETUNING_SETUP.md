# Installation & Setup Guide for Fine-tuning

## Prerequisites

- **GPU Required**: NVIDIA GPU with at least 8GB VRAM
- **CUDA**: Version 11.8 or 12.1
- **Python**: 3.10 or 3.11

---

## Step 1: Install Dependencies

```powershell
# Install Unsloth and dependencies
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Install additional requirements
pip install --no-deps xformers trl peft accelerate bitsandbytes

# Install dataset library
pip install datasets
```

---

## Step 2: Export Training Data

```powershell
python scripts/export_reviewer_training.py
```

This creates `reviewer_training/reviewer_training.jsonl` with your AI's best reviews.

---

## Step 3: Run Fine-tuning

```powershell
python scripts/finetune_reviewer.py
```

**Expected time:** 30-60 minutes on RTX 3060/4060
**Memory usage:** ~6-8GB VRAM

---

## Step 4: Convert to Ollama (Optional)

### Install llama.cpp

```powershell
pip install llama-cpp-python
```

### Convert to GGUF

```powershell
python -m llama_cpp.convert qwen-reviewer-v2 --outfile qwen-reviewer-v2.gguf
```

### Create Ollama Model

Create `Modelfile`:

```
FROM ./qwen-reviewer-v2.gguf

SYSTEM You are an expert code reviewer evaluating commits for AI training quality.

PARAMETER temperature 0.3
PARAMETER top_p 0.9
```

Import to Ollama:

```powershell
ollama create qwen-reviewer:v2 -f Modelfile
```

---

## Step 5: Test Improved Model

Update `config/settings.py`:

```python
model_name = "qwen-reviewer:v2"
```

Test:

```powershell
python scripts/score_commits.py --limit 10
```

---

## Troubleshooting

### Out of Memory

- Reduce `per_device_train_batch_size` to 1
- Reduce `max_seq_length` to 2048
- Use `load_in_8bit=True` instead of 4bit

### CUDA Not Available

- Verify: `python -c "import torch; print(torch.cuda.is_available())"`
- Reinstall PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu121`

### Slow Training

- Normal! 200 steps takes 30-60 minutes
- Reduce `max_steps` to 100 for faster testing

---

## Configuration Options

Edit `scripts/finetune_reviewer.py`:

```python
# Faster training (less accurate)
max_steps=100
learning_rate=3e-4

# Better quality (slower)
max_steps=500
learning_rate=1e-4
per_device_train_batch_size=1
```

---

## Expected Results

| Metric           | Before     | After           |
| ---------------- | ---------- | --------------- |
| Reasoning length | 1 sentence | 2-3 sentences   |
| Consistency      | Variable   | More consistent |
| Detail level     | Basic      | Detailed        |

**Your AI will be smarter!** 🚀
