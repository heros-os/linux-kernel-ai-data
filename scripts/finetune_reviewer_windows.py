#!/usr/bin/env python3
"""
Fine-tune qwen2.5-coder:7b using standard transformers + PEFT (Windows compatible).

This is an alternative to Unsloth that works on Windows.
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from pathlib import Path
import json

# Configuration
MAX_SEQ_LENGTH = 2048  # Reduced for Windows compatibility
MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
OUTPUT_DIR = "qwen-reviewer-v2"
TRAINING_DATA = "reviewer_training/reviewer_training.jsonl"

def format_prompt(example):
    """Format training examples."""
    return f"""<|im_start|>system
{example['system']}<|im_end|>
<|im_start|>user
{example['instruction']}<|im_end|>
<|im_start|>assistant
{example['output']}<|im_end|>"""

def main():
    print("=" * 70)
    print(" FINE-TUNING QWEN2.5-CODER (Windows Compatible)")
    print("=" * 70)
    
    # Check training data
    if not Path(TRAINING_DATA).exists():
        print(f"\n[ERROR] Training data not found: {TRAINING_DATA}")
        print("Run: python scripts/export_reviewer_training.py")
        return
    
    print(f"\n[1/6] Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"[2/6] Loading model (this may take a few minutes)")
    
    # Load model in 8-bit (more Windows-compatible than 4-bit)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        load_in_8bit=True,
        device_map="auto",
        trust_remote_code=True,
    )
    
    print("[3/6] Preparing model for training")
    model = prepare_model_for_kbit_training(model)
    
    # LoRA configuration
    lora_config = LoraConfig(
        r=8,  # Reduced for faster training
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    print("[4/6] Loading and formatting training data")
    
    # Load dataset
    dataset = load_dataset("json", data_files=TRAINING_DATA, split="train")
    
    def tokenize_function(examples):
        # Format prompts
        texts = []
        for i in range(len(examples['system'])):
            text = format_prompt({
                'system': examples['system'][i],
                'instruction': examples['instruction'][i],
                'output': examples['output'][i]
            })
            texts.append(text)
        
        # Tokenize
        return tokenizer(
            texts,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding=False,
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    print(f"  Training examples: {len(tokenized_dataset)}")
    
    print("[5/6] Configuring training")
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=1,  # Small batch for Windows
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        warmup_steps=10,
        optim="adamw_torch",  # More compatible
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    print("[6/6] Training (this may take 30-60 minutes)")
    print(f"  Epochs: {training_args.num_train_epochs}")
    print(f"  Batch size: {training_args.per_device_train_batch_size}")
    print(f"  GPU available: {torch.cuda.is_available()}")
    
    # Train
    trainer.train()
    
    print("\nSaving model...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("\n" + "=" * 70)
    print(" TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nModel saved to: {OUTPUT_DIR}")
    print("\nNext steps:")
    print("1. Test the model with transformers")
    print("2. Or convert to GGUF for Ollama (requires llama.cpp)")
    print("\nTo test:")
    print("  python scripts/test_finetuned_model.py")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Training stopped")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
