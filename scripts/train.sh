#!/bin/bash
# XBMC Coder Training Script
# Run on a machine with CUDA GPU (cloud or WSL)

set -e

echo "=== XBMC Coder Training Setup ==="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python not found. Installing..."
    sudo apt update
    sudo apt install -y python3 python3-pip python3-venv
fi

# Use python3 explicitly
PYTHON=python3
PIP="python3 -m pip"

# Check for CUDA/GPU
echo ""
echo "Checking GPU..."
$PYTHON -c "import torch; print(f'PyTorch CUDA: {torch.cuda.is_available()}')" 2>/dev/null || {
    echo "PyTorch not installed. Installing..."
    $PIP install torch --quiet
    $PYTHON -c "import torch; print(f'PyTorch CUDA: {torch.cuda.is_available()}')"
}

# Check if CUDA is available
CUDA_AVAILABLE=$($PYTHON -c "import torch; print(torch.cuda.is_available())")
if [ "$CUDA_AVAILABLE" = "False" ]; then
    echo ""
    echo "WARNING: No CUDA GPU detected!"
    echo "Training will be very slow on CPU."
    echo ""
    echo "For GPU training, you need:"
    echo "1. NVIDIA GPU with CUDA drivers"
    echo "2. WSL2 with CUDA support (nvidia-smi should work)"
    echo ""
    read -p "Continue with CPU training? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    $PYTHON -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"
fi

# Install dependencies
echo ""
echo "Installing Axolotl and dependencies..."
$PIP install axolotl accelerate transformers datasets peft bitsandbytes --quiet

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
echo ""
echo "Starting training..."
echo "Config: $CONFIG"
echo "Data: exports/xbmc_training.jsonl (173K examples)"
echo ""

$PYTHON -m accelerate.commands.launch -m axolotl.cli.train $CONFIG

echo ""
echo "=== Training Complete ==="
echo "LoRA weights saved to: models/xbmc-coder-*-lora"
