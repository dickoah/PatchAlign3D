#!/bin/bash
set -e  # Exit strictly on any error

echo "=== Installing PyTorch with CUDA 12.8 support ==="
pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 torchaudio==2.7.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128

echo "=== Installing core dependencies from requirements.txt ==="
pip install -r requirements.txt

echo "=== Installing Pointnet2 Ops (Custom Compile) ==="
# Set architectures for modern GPUs (Ampere, Ada, Hopper) to avoid compile errors
export TORCH_CUDA_ARCH_LIST="8.0 8.6 9.0"
pip install "./libraries/pointnet2_pytorch/pointnet2_ops_lib/" --no-build-isolation

echo "=== Installing KNN CUDA ==="
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl

echo "=== Installation steps completed successfully! ==="
