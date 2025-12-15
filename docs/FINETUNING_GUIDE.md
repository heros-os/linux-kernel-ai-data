# Fine-Tuning Qwen2.5-Coder 7B for Code Review

## Overview

Ollama runs pre-trained models but doesn't support fine-tuning directly. You need to:

1. **Fine-tune** the base model using a training framework
2. **Convert** the fine-tuned model to GGUF format
3. **Import** into Ollama

## Recommended Options

### Option 1: Unsloth (Fastest, Easiest)

**Pros**: 2x faster, uses 60% less memory, easy setup
**Cons**: Linux/Colab only (WSL works)

```bash
# Install in WSL
pip install unsloth

# Create training script
python train_reviewer.py
```

### Option 2: Axolotl (Most Flexible)

**Pros**: Many options, well-documented
**Cons**: More complex setup

### Option 3: LLaMA-Factory (GUI Available)

**Pros**: User-friendly web UI
**Cons**: Slightly slower

---

## Step-by-Step: Unsloth in WSL

### 1. Setup WSL Environment

```bash
# In WSL terminal
conda create -n finetune python=3.11
conda activate finetune
pip install unsloth torch transformers datasets trl
```

### 2. Prepare Dataset

Copy your JSONL to WSL:

```bash
cp /mnt/c/Users/ewert/.gemini/antigravity/scratch/linux-kernel-ai-data/huggingface_datasets/*.jsonl ~/training_data/
```

### 3. Create Training Script

Create `train_reviewer.py`:

```python
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# Load model with 4-bit quantization
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-Coder-7B",
    max_seq_length=4096,
    load_in_4bit=True,
)

# Apply LoRA for efficient fine-tuning
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)

# Load your dataset
dataset = load_dataset("json", data_files="~/training_data/premium_reasoning.jsonl")

# Format for instruction tuning
def format_instruction(example):
    return {
        "text": f"""### Instruction
{example['instruction']}

### Input
{example['input']}

### Response
{example['output']}"""
    }

dataset = dataset.map(format_instruction)

# Training
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    dataset_text_field="text",
    max_seq_length=4096,
    args=TrainingArguments(
        output_dir="./reviewer_finetuned",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_steps=500,
    ),
)

trainer.train()

# Save LoRA adapter
model.save_pretrained("./reviewer_finetuned")
```

### 4. Run Training

```bash
python train_reviewer.py
```

Training time estimate:

- **premium_reasoning** (2,765 examples): ~1-2 hours on RTX 3080+
- **high_reasoning** (8,339 examples): ~3-5 hours

### 5. Merge LoRA and Convert to GGUF

```python
# merge_and_export.py
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Qwen2.5-Coder-7B",
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(model)
model.load_adapter("./reviewer_finetuned")

# Merge adapters
model = model.merge_and_unload()

# Export to GGUF for Ollama
model.save_pretrained_gguf(
    "reviewer_gguf",
    tokenizer,
    quantization_method="q4_k_m"
)
```

### 6. Import into Ollama

Create a `Modelfile`:

```
FROM ./reviewer_gguf/model-q4_k_m.gguf

SYSTEM "You are a Linux kernel developer. Given a commit message describing a bug fix, feature, or improvement, and the relevant source code context, generate the appropriate kernel patch."

PARAMETER temperature 0.3
PARAMETER top_p 0.9
```

Import:

```bash
ollama create kernel-reviewer -f Modelfile
```

### 7. Test Your Model

```bash
ollama run kernel-reviewer
```

---

## Quick Alternative: Google Colab

If you don't have a powerful GPU, use Colab's free T4:

1. Upload `premium_reasoning.jsonl` to Google Drive
2. Open [Unsloth Colab Notebook](https://colab.research.google.com/drive/1Ys44kVvmeZtnICzWz0xgpRnrIOjZAuxp)
3. Follow the notebook steps
4. Download the GGUF file
5. Import into Ollama

---

## Which Dataset to Use?

| Dataset             | Use Case                  | Training Time |
| ------------------- | ------------------------- | ------------- |
| `premium_reasoning` | Best quality, AI-verified | ~1-2 hours    |
| `high_reasoning`    | More data, good quality   | ~3-5 hours    |
| `premium_score`     | Large set, no reasoning   | ~2-3 hours    |

**Recommendation**: Start with `premium_reasoning` for highest quality results.

---

## Expected Results

After fine-tuning, your model should:

- Generate better kernel patches
- Understand Linux kernel coding conventions
- Provide more accurate bug fixes
- Match the style of high-quality kernel commits

## Need Help?

Let me know if you want me to:

1. Set up the WSL environment
2. Create the training scripts
3. Test the fine-tuning process
