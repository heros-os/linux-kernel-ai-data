# LLM Fine-Tuning Guide

After exporting your training data, here's how to fine-tune open-source models.

## Model Options

| Model                   | Size  | VRAM Required        | Quality    |
| ----------------------- | ----- | -------------------- | ---------- |
| **Qwen 2.5 Coder 14B**  | ~28GB | 48GB (24GB w/ QLoRA) | ⭐⭐⭐⭐⭐ |
| **DeepSeek Coder 6.7B** | ~13GB | 24GB (16GB w/ QLoRA) | ⭐⭐⭐⭐   |
| **Qwen 2.5 Coder 7B**   | ~14GB | 24GB (16GB w/ QLoRA) | ⭐⭐⭐⭐   |
| **CodeLlama 7B**        | ~14GB | 24GB (16GB w/ QLoRA) | ⭐⭐⭐     |
| **Mistral 7B**          | ~14GB | 24GB (16GB w/ QLoRA) | ⭐⭐⭐     |

## Quick Start with Axolotl

[Axolotl](https://github.com/axolotl-ai-cloud/axolotl) is the easiest way to fine-tune.

### 1. Install Axolotl

```bash
# Clone axolotl
git clone https://github.com/axolotl-ai-cloud/axolotl.git
cd axolotl

# Install (requires PyTorch + CUDA)
pip install packaging ninja
pip install -e '.[flash-attn,deepspeed]'
```

### 2. Prepare Training Data

```bash
# In this project directory
python scripts/export_training_data.py \
    -o exports/training_data.jsonl \
    --mode diff \
    --min-lines 5 \
    --max-lines 200
```

### 3. Run Training

```bash
# For 14B model (requires 48GB VRAM or cloud GPU)
accelerate launch -m axolotl.cli.train training/axolotl_config.yaml

# For 7B model (works on RTX 3090/4090)
accelerate launch -m axolotl.cli.train training/axolotl_config_7b.yaml
```

### 4. Merge LoRA Weights

```bash
python -m axolotl.cli.merge_lora training/axolotl_config.yaml \
    --lora_model_dir ./models/kernel-coder-14b-lora \
    --output_dir ./models/kernel-coder-14b-merged
```

## Cloud Training Options

If you don't have sufficient local GPU:

| Provider             | Cost      | GPU       | Notes               |
| -------------------- | --------- | --------- | ------------------- |
| **RunPod**           | ~$1.50/hr | A100 80GB | Best value          |
| **Lambda Labs**      | ~$1.25/hr | A100 40GB | Fast setup          |
| **Vast.ai**          | ~$0.50/hr | RTX 4090  | Cheapest, community |
| **Google Colab Pro** | $10/mo    | A100      | Easy, time-limited  |

## Alternative: Unsloth (2x Faster)

[Unsloth](https://github.com/unslothai/unsloth) provides optimized training:

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-Coder-14B-Instruct",
    max_seq_length=4096,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=64,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

# Train with HuggingFace Trainer or TRL
```

## Inference After Training

````python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base + LoRA
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-14B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, "./models/kernel-coder-14b-lora")

# Or use merged model
model = AutoModelForCausalLM.from_pretrained(
    "./models/kernel-coder-14b-merged",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Generate
prompt = "Fix the race condition in this code:\n```c\nvoid foo() { ... }"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0]))
````

## Expected Training Time

| Dataset Size  | Model | GPU       | Time          |
| ------------- | ----- | --------- | ------------- |
| 100K examples | 7B    | RTX 4090  | ~4-6 hours    |
| 100K examples | 14B   | A100 80GB | ~8-12 hours   |
| 1M examples   | 7B    | RTX 4090  | ~40-60 hours  |
| 1M examples   | 14B   | A100 80GB | ~80-100 hours |

## Tips for Best Results

1. **Start small**: Train on 10K examples first to validate the pipeline
2. **Filter carefully**: Quality > quantity — remove trivial commits
3. **Subsystem focus**: Train on specific subsystems (mm, kernel/sched) for experts
4. **Evaluation**: Test on held-out commits before scaling up
