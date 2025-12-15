#!/usr/bin/env python3
"""
Fine-tune qwen2.5-coder:7b using Unsloth for better code review.

This script uses your AI's best reviews to improve its scoring capabilities.
"""

import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import json
from pathlib import Path

# Configuration
MAX_SEQ_LENGTH = 4096
MODEL_NAME = "Qwen/Qwen2.5-Coder-7B"
OUTPUT_DIR = "qwen-reviewer-v2"
TRAINING_DATA = "reviewer_training/reviewer_training.jsonl"

def format_prompt(example):
    """Format training examples for instruction tuning."""
    return f"""<|im_start|>system
{example['system']}<|im_end|>
<|im_start|>user
{example['instruction']}<|im_end|>
<|im_start|>assistant
{example['output']}<|im_end|>"""

def main():
    print("=" * 70)
    print(" FINE-TUNING QWEN2.5-CODER FOR CODE REVIEW")
    print("=" * 70)
    
    # Check if training data exists
    if not Path(TRAINING_DATA).exists():
        print(f"\n[ERROR] Training data not found: {TRAINING_DATA}")
        print("Run: python scripts/export_reviewer_training.py")
        return
    
    print(f"\n[1/6] Loading base model: {MODEL_NAME}")
    
    # Load model with 4-bit quantization
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,  # Auto-detect
        load_in_4bit=True,  # Use 4-bit quantization to save memory
    )
    
    print("[2/6] Adding LoRA adapters")
    
    # Add LoRA adapters for efficient fine-tuning
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # LoRA rank
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Memory efficient
        random_state=42,
    )
    
    print("[3/6] Loading training data")
    
    # Load dataset
    dataset = load_dataset("json", data_files=TRAINING_DATA, split="train")
    
    # Format for training
    def formatting_func(examples):
        texts = []
        for i in range(len(examples['system'])):
            text = format_prompt({
                'system': examples['system'][i],
                'instruction': examples['instruction'][i],
                'output': examples['output'][i]
            })
            texts.append(text)
        return {"text": texts}
    
    dataset = dataset.map(formatting_func, batched=True)
    
    print(f"  Training examples: {len(dataset)}")
    
    print("[4/6] Configuring training")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=200,  # Adjust based on dataset size
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        save_strategy="steps",
        save_steps=50,
    )
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=training_args,
    )
    
    print("[5/6] Training (this may take 30-60 minutes)")
    print(f"  Steps: {training_args.max_steps}")
    print(f"  Batch size: {training_args.per_device_train_batch_size}")
    print(f"  Learning rate: {training_args.learning_rate}")
    
    # Train!
    trainer.train()
    
    print("[6/6] Saving fine-tuned model")
    
    # Save model
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("\n" + "=" * 70)
    print(" TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nModel saved to: {OUTPUT_DIR}")
    print("\nNext steps:")
    print("1. Convert to GGUF for Ollama:")
    print(f"   python -m llama_cpp.convert {OUTPUT_DIR}")
    print("\n2. Create Ollama model:")
    print("   ollama create qwen-reviewer:v2 -f Modelfile")
    print("\n3. Test improved model:")
    print("   Update config/settings.py model_name to 'qwen-reviewer:v2'")
    print("   python scripts/score_commits.py --limit 10")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Training stopped by user")
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
