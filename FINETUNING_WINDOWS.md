# Windows Fine-tuning Guide

## Issue: Unsloth doesn't work on Windows

Unsloth has Linux-only dependencies (vllm). Use this Windows-compatible alternative instead.

---

## Step 1: Install Dependencies

```powershell
# Install transformers + PEFT
pip install transformers peft accelerate bitsandbytes datasets

# Install PyTorch with CUDA (if you have NVIDIA GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## Step 2: Export Training Data

```powershell
python scripts/export_reviewer_training.py
```

---

## Step 3: Run Fine-tuning (Windows Compatible)

```powershell
python scripts/finetune_reviewer_windows.py
```

**Time:** 30-60 minutes with GPU, 2-4 hours on CPU

---

## Differences from Unsloth

| Feature       | Unsloth     | Windows Script |
| ------------- | ----------- | -------------- |
| Speed         | Faster      | Standard       |
| Memory        | 4-bit quant | 8-bit quant    |
| Compatibility | Linux only  | Windows ✅     |
| Quality       | Same        | Same           |

---

## If You Don't Have a GPU

Fine-tuning on CPU is **very slow** (hours). Alternatives:

### Option 1: Use Google Colab (Free GPU)

1. Upload `reviewer_training.jsonl` to Google Drive
2. Open Colab: https://colab.research.google.com
3. Use the original Unsloth script (works on Linux)
4. Download the fine-tuned model

### Option 2: Use Smaller Model

Fine-tune `qwen2.5-coder:1.5b` instead (much faster):

```python
MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
```

### Option 3: Skip Fine-tuning

Just use the improved prompts! The better context (4000 chars) already improves quality significantly.

---

## Testing the Fine-tuned Model

After training completes, test it:

```powershell
python scripts/test_finetuned_model.py
```

---

## Troubleshooting

### "CUDA out of memory"

- Reduce `MAX_SEQ_LENGTH` to 1024
- Reduce `per_device_train_batch_size` to 1
- Close other applications

### "No module named 'bitsandbytes'"

- Install: `pip install bitsandbytes`
- If still fails, remove `load_in_8bit=True` (uses more memory)

### Training is very slow

- Normal on CPU (2-4 hours)
- Use Google Colab for free GPU
- Or use smaller model (1.5B)

---

## Expected Results

Same quality improvement as Unsloth, just takes a bit longer to train.

**Your AI will still get smarter!** 🚀
