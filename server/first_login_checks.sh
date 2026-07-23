#!/bin/bash
# ============================================================
# Run these commands ON gpu1 the first time you connect,
# to inventory the environment before running experiments.
# Copy-paste them one block at a time.
# ============================================================

echo "===== 1. WHICH V100 (16GB or 32GB)? ====="
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv

echo ""
echo "===== 2. CUDA version available ====="
nvidia-smi | grep "CUDA Version"
nvcc --version 2>/dev/null || echo "nvcc not in PATH (usually fine; torch bundles CUDA)"

echo ""
echo "===== 3. Python + pip ====="
python3 --version
which python3
pip3 --version 2>/dev/null || echo "pip3 not found - may need: python3 -m ensurepip"

echo ""
echo "===== 4. Is conda/mamba available? (preferred for envs) ====="
which conda 2>/dev/null && conda --version || echo "no conda"
which mamba 2>/dev/null && mamba --version || echo "no mamba"

echo ""
echo "===== 5. Disk space (need ~30-50GB for models + datasets) ====="
df -h ~ 
echo "--- home quota (if any) ---"
quota -s 2>/dev/null || echo "no quota command / no quota set"

echo ""
echo "===== 6. RAM available ====="
free -h

echo ""
echo "===== 7. Is torch already installed with CUDA? ====="
python3 -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available(), '| device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')" 2>/dev/null || echo "torch not installed yet"

echo ""
echo "===== 8. Internet access from gpu1? (for pip / HF downloads) ====="
curl -sI https://huggingface.co 2>/dev/null | head -1 || echo "NO direct internet - may need proxy or download via hop"

echo ""
echo "===== 9. Existing CUDA modules (HPC clusters often use 'module') ====="
module avail 2>&1 | head -20 || echo "no 'module' system"

echo ""
echo "===== DONE - paste all output back to review ====="
