#!/bin/bash
# XBMC Coder Training Script
# Run on a machine with CUDA GPU (cloud or local)

set -e

echo "=== XBMC Coder Training Setup ==="

# Check GPU
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"

# Install dependencies
echo "Installing Axolotl..."
pip install axolotl[flash-attn] --quiet

# Choose model (DeepSeek or Qwen)
MODEL=${1:-deepseek}

if [ "$MODEL" == "qwen" ]; then
    CONFIG="training/axolotl_qwen.yaml"
    echo "Training with Qwen 2.5 Coder 7B"
else
    CONFIG="training/axolotl_config_7b.yaml"
    echo "Training with DeepSeek Coder 6.7B"
fi

# Start training
echo "Starting training..."
echo "Config: $CONFIG"
echo "Data: exports/xbmc_training.jsonl"
echo ""

accelerate launch -m axolotl.cli.train $CONFIG

echo ""
echo "=== Training Complete ==="
echo "LoRA weights saved to: models/xbmc-coder-*-lora"
