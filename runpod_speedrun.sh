#!/bin/bash

# RunPod optimized version of speedrun.sh
# "Best ChatGPT clone that $100 can buy" - RunPod 8xH100 Edition
# Designed to run in ~4 hours on RunPod 8xH100 node

echo "=== NANOCHAT SPEEDRUN - RUNPOD 8XH100 ==="
echo "Start time: $(date)"
echo "Expected duration: ~4 hours"
echo "Expected cost: ~$100 (depending on RunPod pricing)"

# 1) Example launch (simplest):
# bash runpod_speedrun.sh
# 2) Example launch in a screen session (recommended for 4-hour run):
# screen -L -Logfile speedrun.log -S speedrun bash runpod_speedrun.sh
# 3) Example launch with wandb logging:
# WANDB_RUN=speedrun screen -L -Logfile speedrun.log -S speedrun bash runpod_speedrun.sh

# RunPod specific environment setup
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # Ensure all 8 GPUs are visible

# Use workspace directory instead of ~/.cache for RunPod persistence
export NANOCHAT_BASE_DIR="/workspace/nanochat_artifacts"
mkdir -p $NANOCHAT_BASE_DIR

echo "📁 Artifacts will be saved to: $NANOCHAT_BASE_DIR"
echo "💾 This directory persists across RunPod sessions"

# -----------------------------------------------------------------------------
# Python venv setup with uv

echo "🐍 Setting up Python environment..."

# install uv (if not already installed)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv
# install the repo dependencies
uv sync
# activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate

# -----------------------------------------------------------------------------
# wandb setup
if [ -z "$WANDB_RUN" ]; then
    # by default use "dummy" : it's handled as a special case, skips logging to wandb
    WANDB_RUN=dummy
fi

echo "📊 Using WANDB_RUN: $WANDB_RUN"

# -----------------------------------------------------------------------------
# System info and report initialization

echo "🖥️  RunPod System Info:"
echo "GPU Count: $(nvidia-smi -L | wc -l)"
echo "GPU Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1) MB per GPU"
echo "CPU Cores: $(nproc)"
echo "RAM: $(free -h | awk '/^Mem:/ {print $2}')"
echo "Disk Space: $(df -h /workspace | awk 'NR==2 {print $4}') available"

# Initialize report
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer Training

echo ""
echo "🔤 STAGE 1: TOKENIZER TRAINING"
echo "==============================="

echo "Installing Rust/Cargo..."
# Install Rust / Cargo
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

echo "Building rustbpe tokenizer..."
# Build the rustbpe Tokenizer
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

echo "Downloading dataset shards..."
# Download the first ~2B characters of pretraining dataset (8 shards for tokenizer)
python -m nanochat.dataset -n 8

echo "Starting background download of full dataset (240 shards)..."
# Kick off downloading more shards in the background while tokenizer trains
python -m nanochat.dataset -n 240 &
DATASET_DOWNLOAD_PID=$!

echo "Training tokenizer (vocab size 65536 on ~2B characters)..."
# train the tokenizer with vocab size 2**16 = 65536 on ~2B characters of data
python -m scripts.tok_train --max_chars=2000000000

echo "Evaluating tokenizer..."
# evaluate the tokenizer (report compression ratio etc.)
python -m scripts.tok_eval

echo "✅ Tokenizer training complete!"

# -----------------------------------------------------------------------------
# Base Model Pretraining

echo ""
echo "🧠 STAGE 2: BASE MODEL PRETRAINING"
echo "=================================="

echo "Downloading evaluation bundle..."
# Download the eval_bundle from s3 to evaluate CORE metric during training (~162MB)
EVAL_BUNDLE_URL=https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip
if [ ! -d "$NANOCHAT_BASE_DIR/eval_bundle" ]; then
    curl -L -o eval_bundle.zip $EVAL_BUNDLE_URL
    unzip -q eval_bundle.zip
    rm eval_bundle.zip
    mv eval_bundle $NANOCHAT_BASE_DIR
fi

echo "Waiting for complete dataset download..."
wait $DATASET_DOWNLOAD_PID

echo "Starting base model pretraining (d20, 561M parameters)..."
echo "This is the longest stage - will take 2-3 hours on 8xH100..."

# Verify we have 8 GPUs before starting expensive pretraining
GPU_COUNT=$(nvidia-smi -L | wc -l)
if [ $GPU_COUNT -ne 8 ]; then
    echo "❌ ERROR: Expected 8 GPUs, found $GPU_COUNT"
    echo "Please ensure you're using an 8xH100 RunPod instance"
    exit 1
fi

# pretrain the d20 model
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=20 --run=$WANDB_RUN

echo "Evaluating base model..."
# evaluate the model on a larger chunk of train/val data and draw some samples
torchrun --standalone --nproc_per_node=8 -m scripts.base_loss
# evaluate the model on CORE tasks
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval

echo "✅ Base model pretraining complete!"

# -----------------------------------------------------------------------------
# Midtraining

echo ""
echo "💬 STAGE 3: MIDTRAINING"
echo "======================="

echo "Teaching model conversation tokens, tool use, and multiple choice..."
# run midtraining and eval the model
torchrun --standalone --nproc_per_node=8 -m scripts.mid_train -- --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i mid

echo "✅ Midtraining complete!"

# -----------------------------------------------------------------------------
# Supervised Fine-tuning

echo ""
echo "🎯 STAGE 4: SUPERVISED FINE-TUNING"
echo "=================================="

echo "Running supervised fine-tuning..."
# train sft and re-eval right away (should see a small bump)
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft

echo "✅ Supervised fine-tuning complete!"

# -----------------------------------------------------------------------------
# Optional Reinforcement Learning

echo ""
echo "🤖 STAGE 5: REINFORCEMENT LEARNING (Optional)"
echo "============================================="

read -p "Run Reinforcement Learning training? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Running reinforcement learning (GSM8K)..."
    torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl -- --run=$WANDB_RUN
    torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i rl -a GSM8K
    echo "✅ Reinforcement learning complete!"
else
    echo "⏭️  Skipping reinforcement learning"
fi

# -----------------------------------------------------------------------------
# Final Report Generation

echo ""
echo "📊 GENERATING FINAL REPORT"
echo "========================="

# Generate the full report by putting together all the sections
python -m nanochat.report generate

# Copy report to workspace root for easy access
cp $NANOCHAT_BASE_DIR/report.md /workspace/nanochat_final_report.md

echo ""
echo "🎉 NANOCHAT SPEEDRUN COMPLETE!"
echo "============================="
echo "End time: $(date)"

# Calculate and display duration
START_TIME_FILE="/tmp/nanochat_start_time"
if [ -f "$START_TIME_FILE" ]; then
    START_TIME=$(cat "$START_TIME_FILE")
    DURATION=$(($(date +%s) - $START_TIME))
    HOURS=$((DURATION / 3600))
    MINUTES=$(((DURATION % 3600) / 60))
    echo "Total duration: ${HOURS}h ${MINUTES}m"
else
    echo "Duration: Check timestamps above"
fi

echo ""
echo "📁 OUTPUTS SAVED TO:"
echo "- Model checkpoints: $NANOCHAT_BASE_DIR/"
echo "- Final report: /workspace/nanochat_final_report.md"
echo "- Training logs: All stages logged to report"
echo ""
echo "💬 TO CHAT WITH YOUR MODEL:"
echo "1. CLI chat: python -m scripts.chat_cli -p \"Why is the sky blue?\""
echo "2. Interactive CLI: python -m scripts.chat_cli"
echo "3. Web UI: python -m scripts.chat_web"
echo ""
echo "🌐 For Web UI access:"
echo "   - Start: python -m scripts.chat_web"
echo "   - Visit: http://[YOUR_RUNPOD_IP]:8000/"
echo "   - Use RunPod's 'Connect' button for port forwarding"
echo ""
echo "💾 All artifacts are saved in /workspace/ and will persist"
echo "   across RunPod sessions if you don't terminate the pod."

# Save completion timestamp
date +%s > "$START_TIME_FILE"