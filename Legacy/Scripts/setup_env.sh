#!/bin/bash

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA support (RTX 3060 — CUDA 12.4)
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install other dependencies
pip install -r requirements.txt

echo "Environment setup complete"
echo "Run: source venv/bin/activate"
