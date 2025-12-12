#!/usr/bin/env python3
"""
XBMC Coder Training Script
Fine-tune DeepSeek/Qwen on XBMC commit data using QLoRA
Works on Windows with RTX 3090/4090/5080 (16-24GB VRAM)
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model options
MODELS = {
    "deepseek": "deepseek-ai/deepseek-coder-6.7b-instruct",
    "qwen": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "qwen-3b": "Qwen/Qwen2.5-Coder-3B-Instruct",  # Smaller, faster
}


def load_training_data(data_path: str, max_samples: int = None):
    """Load JSONL training data."""
    logger.info(f"Loading training data from {data_path}")
    
    examples = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            examples.append(json.loads(line))
    
    logger.info(f"Loaded {len(examples)} examples")
    return Dataset.from_list(examples)


def format_prompt(example):
    """Format example for instruction tuning."""
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")
    
    if input_text:
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    
    return {"prompt": prompt, "completion": output}


def tokenize_function(examples, tokenizer, max_length):
    """Tokenize prompts and completions."""
    prompts = examples["prompt"]
    completions = examples["completion"]
    
    # Combine prompt and completion
    texts = [p + c + tokenizer.eos_token for p, c in zip(prompts, completions)]
    
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )
    
    # Create labels (same as input_ids for causal LM)
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized


def main():
    parser = argparse.ArgumentParser(description="Train XBMC Coder")
    parser.add_argument("--model", choices=list(MODELS.keys()), default="qwen-3b",
                        help="Base model to fine-tune")
    parser.add_argument("--data", default="exports/xbmc_training.jsonl",
                        help="Path to training data")
    parser.add_argument("--output", default="models/xbmc-coder",
                        help="Output directory for model")
    parser.add_argument("--max-samples", type=int, default=10000,
                        help="Max training samples (use fewer for testing)")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Batch size per GPU")
    parser.add_argument("--max-length", type=int, default=1024,
                        help="Max sequence length")
    parser.add_argument("--lora-r", type=int, default=16,
                        help="LoRA rank")
    
    args = parser.parse_args()
    
    # Check GPU
    if not torch.cuda.is_available():
        logger.error("CUDA not available! Training on CPU would be too slow.")
        sys.exit(1)
    
    device = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    logger.info(f"GPU: {device} ({vram:.1f} GB)")
    
    # Model path
    model_name = MODELS[args.model]
    output_dir = f"{args.output}-{args.model}-lora"
    
    logger.info(f"Base model: {model_name}")
    logger.info(f"Output: {output_dir}")
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Quantization config for 4-bit loading
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load model in 4-bit
    logger.info("Loading model in 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    
    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # LoRA config
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load and process data
    dataset = load_training_data(args.data, args.max_samples)
    dataset = dataset.map(format_prompt)
    
    # Tokenize
    logger.info("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, args.max_length),
        batched=True,
        remove_columns=dataset.column_names,
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        report_to="none",
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    # Train!
    logger.info("Starting training...")
    trainer.train()
    
    # Save
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    logger.info("Training complete!")
    logger.info(f"LoRA weights saved to: {output_dir}")


if __name__ == "__main__":
    main()
