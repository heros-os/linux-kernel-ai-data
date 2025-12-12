# XBMC Coder Training Script (Windows PowerShell)
# Run on WSL2 or cloud with CUDA

Write-Host "=== XBMC Coder Training Setup ===" -ForegroundColor Cyan

# Check if running in WSL
$inWSL = $false
if ($env:WSL_DISTRO_NAME) {
    $inWSL = $true
    Write-Host "Running in WSL: $env:WSL_DISTRO_NAME" -ForegroundColor Green
}

# Check GPU
try {
    python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"
} catch {
    Write-Host "No CUDA GPU detected!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Options:" -ForegroundColor Yellow
    Write-Host "1. Use WSL2 with CUDA: wsl --install, then run scripts/train.sh"
    Write-Host "2. Use cloud GPU (RunPod, Lambda Labs, Vast.ai)"
    Write-Host "3. Use Google Colab (upload xbmc_training.jsonl)"
    Write-Host ""
    exit 1
}

# Install Axolotl
Write-Host "Installing Axolotl..." -ForegroundColor Yellow
pip install axolotl[flash-attn] --quiet

# Select model
$model = $args[0]
if ($model -eq "qwen") {
    $config = "training/axolotl_qwen.yaml"
    Write-Host "Training with Qwen 2.5 Coder 7B" -ForegroundColor Green
} else {
    $config = "training/axolotl_config_7b.yaml"
    Write-Host "Training with DeepSeek Coder 6.7B" -ForegroundColor Green
}

# Start training
Write-Host ""
Write-Host "Starting training..." -ForegroundColor Cyan
Write-Host "Config: $config"
Write-Host "Data: exports/xbmc_training.jsonl (173K examples)"
Write-Host ""

accelerate launch -m axolotl.cli.train $config

Write-Host ""
Write-Host "=== Training Complete ===" -ForegroundColor Green
