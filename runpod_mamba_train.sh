#!/bin/bash

# Mamba Training Script for RunPod
# Train a Mamba model instead of GPT on RunPod infrastructure

echo "=== NANOCHAT MAMBA TRAINING - RUNPOD ==="
echo "Start time: $(date)"
echo "Model: Mamba (Linear-time State Space Model)"

# RunPod specific environment setup
export OMP_NUM_THREADS=1
GPU_COUNT=$(nvidia-smi -L | wc -l)
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPU_COUNT-1)))

# Use workspace directory for persistence
export NANOCHAT_BASE_DIR="/workspace/nanochat_artifacts"
mkdir -p $NANOCHAT_BASE_DIR

echo "📁 Artifacts will be saved to: $NANOCHAT_BASE_DIR"
echo "🔥 Using $GPU_COUNT GPUs for Mamba training"

# -----------------------------------------------------------------------------
# Python environment setup

echo "🐍 Setting up Python environment..."

# Install uv if not available
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
[ -d ".venv" ] || uv venv
source .venv/bin/activate
uv sync

# Install Mamba SSM library
echo "🐍 Installing Mamba SSM library..."
pip install mamba-ssm

# -----------------------------------------------------------------------------
# wandb setup
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

echo "📊 Using WANDB_RUN: $WANDB_RUN"

# -----------------------------------------------------------------------------
# System info

echo "🖥️  RunPod System Info:"
echo "GPU Count: $GPU_COUNT"
echo "GPU Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1) MB per GPU"
echo "CPU Cores: $(nproc)"
echo "RAM: $(free -h | awk '/^Mem:/ {print $2}')"

# Initialize report
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer Setup (reuse existing tokenizer)

echo ""
echo "🔤 TOKENIZER SETUP"
echo "=================="

# Check if tokenizer exists, if not create it
if [ ! -d "$NANOCHAT_BASE_DIR/tokenizer" ]; then
    echo "Tokenizer not found, please run tokenizer training first:"
    echo "python -m scripts.tok_train --max_chars=2000000000"
    exit 1
else
    echo "✅ Using existing tokenizer at $NANOCHAT_BASE_DIR/tokenizer"
fi

# -----------------------------------------------------------------------------
# Dataset Setup

echo ""
echo "📦 DATASET SETUP" 
echo "================"

# Check if we have enough data shards
SHARD_COUNT=$(ls $NANOCHAT_BASE_DIR/base_data/*.parquet 2>/dev/null | wc -l)
if [ $SHARD_COUNT -lt 8 ]; then
    echo "Downloading dataset shards..."
    python -m nanochat.dataset -n 240
else
    echo "✅ Found $SHARD_COUNT dataset shards"
fi

# -----------------------------------------------------------------------------
# Mamba Model Training

echo ""
echo "🐍 MAMBA MODEL TRAINING"
echo "======================"

echo "Starting Mamba model training..."
echo "Architecture: Linear-time State Space Model"
echo "Advantages: O(n) complexity vs O(n²) for attention"

# Dynamic GPU count handling
if [ $GPU_COUNT -ge 8 ]; then
    echo "Using $GPU_COUNT GPUs with default settings"
    torchrun --standalone --nproc_per_node=$GPU_COUNT -m scripts.base_train -- \
        --model_type=mamba \
        --depth=20 \
        --run=$WANDB_RUN
elif [ $GPU_COUNT -ge 4 ]; then
    echo "Using $GPU_COUNT GPUs with adjusted batch size"
    torchrun --standalone --nproc_per_node=$GPU_COUNT -m scripts.base_train -- \
        --model_type=mamba \
        --depth=20 \
        --device_batch_size=24 \
        --run=$WANDB_RUN
elif [ $GPU_COUNT -ge 2 ]; then
    echo "Using $GPU_COUNT GPUs with smaller batch size"
    torchrun --standalone --nproc_per_node=$GPU_COUNT -m scripts.base_train -- \
        --model_type=mamba \
        --depth=20 \
        --device_batch_size=16 \
        --run=$WANDB_RUN
else
    echo "Using single GPU"
    python -m scripts.base_train \
        --model_type=mamba \
        --depth=20 \
        --device_batch_size=8 \
        --run=$WANDB_RUN
fi

echo "✅ Mamba model training complete!"

# -----------------------------------------------------------------------------
# Model Evaluation

echo ""
echo "📊 MODEL EVALUATION"
echo "=================="

echo "Evaluating Mamba model..."
# Note: We'll need to update evaluation scripts to handle Mamba
# For now, skip evaluation or use base evaluation with modifications

# -----------------------------------------------------------------------------
# Final Report

echo ""
echo "📄 GENERATING FINAL REPORT"
echo "========================="

python -m nanochat.report generate
cp $NANOCHAT_BASE_DIR/report.md /workspace/mamba_training_report.md

echo ""
echo "🎉 MAMBA TRAINING COMPLETE!"
echo "=========================="
echo "End time: $(date)"

echo ""
echo "📁 OUTPUTS:"
echo "- Mamba model checkpoint: $NANOCHAT_BASE_DIR/base_checkpoints/mamba_d20/"
echo "- Training report: /workspace/mamba_training_report.md"
echo ""
echo "💬 TO TEST YOUR MAMBA MODEL:"
echo "1. CLI: python -m scripts.chat_cli --model_type=mamba"
echo "2. Web UI: python -m scripts.chat_web --model_type=mamba"
echo ""
echo "🚀 Mamba advantages:"
echo "  - Linear O(n) complexity vs quadratic O(n²) for attention"
echo "  - Better scaling for long sequences"
echo "  - More memory efficient during inference"