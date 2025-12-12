#!/usr/bin/env python3
"""
Test the trained XBMC Coder model.
"""

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Model paths
BASE_MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"
LORA_PATH = "models/xbmc-coder-qwen-3b-lora"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(LORA_PATH, trust_remote_code=True)

print("Loading base model...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, LORA_PATH)

print("\n" + "="*50)
print("XBMC Coder Ready!")
print("="*50)

# Test prompts
test_prompts = [
    "Fix the buffer overflow in the video player seek function",
    "Add error handling for failed network connections in the addon downloader",
    "Refactor the audio decoder to support multiple formats",
]

for prompt in test_prompts:
    print(f"\n### Prompt:\n{prompt}\n")
    
    full_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("### Response:\n")[-1]
    print(f"### Response:\n{response[:500]}...")
    print("-"*50)
