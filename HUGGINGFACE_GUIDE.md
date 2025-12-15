# HuggingFace Dataset Publication Guide

## Step 1: Export the Dataset

Run the preparation script:

```powershell
python scripts/prepare_huggingface.py
```

This creates:

- `huggingface_dataset/premium_train.jsonl` (~4,500 examples)
- `huggingface_dataset/premium_val.jsonl` (~500 examples)
- `huggingface_dataset/premium_ai_train.jsonl` (~1,350 examples)
- `huggingface_dataset/premium_ai_val.jsonl` (~150 examples)
- `huggingface_dataset/high_quality_train.jsonl` (~18,000 examples)
- `huggingface_dataset/high_quality_val.jsonl` (~2,000 examples)

## Step 2: Create HuggingFace Account

1. Go to https://huggingface.co
2. Sign up for a free account
3. Verify your email

## Step 3: Install HuggingFace CLI

```powershell
pip install huggingface_hub
```

## Step 4: Login to HuggingFace

```powershell
huggingface-cli login
```

Enter your access token (create one at https://huggingface.co/settings/tokens)

## Step 5: Create Dataset Repository

1. Go to https://huggingface.co/new-dataset
2. Name: `linux-kernel-code`
3. License: `gpl-2.0`
4. Make it public

## Step 6: Upload Dataset

```powershell
cd huggingface_dataset
huggingface-cli upload your-username/linux-kernel-code . --repo-type dataset
```

## Step 7: Add Dataset Card

Copy the content from `huggingface_card.md` to your dataset's README.md on HuggingFace.

## Step 8: Test Loading

```python
from datasets import load_dataset

dataset = load_dataset("your-username/linux-kernel-code", split="premium_ai_train")
print(dataset[0])
```

## Optional: Add Dataset Preview

HuggingFace will automatically generate a preview. You can customize it by adding:

```yaml
# In your dataset card
configs:
  - config_name: premium_ai
    data_files:
      - split: train
        path: premium_ai_train.jsonl
      - split: validation
        path: premium_ai_val.jsonl
```

## Promotion

Share on:

- Twitter/X with #HuggingFace #CodeLLM #LinuxKernel
- Reddit r/MachineLearning
- HuggingFace Discord
- LinkedIn

---

**You're done!** 🎉 Your dataset is now public and ready for the community to use.
