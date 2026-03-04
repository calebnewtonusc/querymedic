# QueryMedic — GPU Setup Guide

## Target Hardware: 18x NVIDIA A6000 (48GB each)

## GPU Allocation
```
GPUs  0-3:  vLLM synthesis instance 1
GPUs  4-7:  vLLM synthesis instance 2
GPUs  8-15: SFT + RL training (8 GPUs, ZeRO-3)
GPUs 16-17: Validation agent (runs EXPLAIN ANALYZE, eval)
```

## Environment Setup
```bash
conda create -n querymedic python=3.11 && conda activate querymedic
pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu121
pip install transformers>=4.40.0 peft>=0.10.0 trl>=0.8.6
pip install deepspeed>=0.14.0 accelerate>=0.28.0
pip install datasets>=2.18.0 bitsandbytes>=0.43.0
pip install flash-attn --no-build-isolation
pip install psycopg2-binary pymysql wandb loguru
```

## Database Setup (Validation Agent)
```bash
# PostgreSQL (required for timing validation)
brew install postgresql@16  # macOS
sudo apt install postgresql-16  # Ubuntu

# Start PostgreSQL
pg_ctl -D /usr/local/var/postgresql@16 start

# Create test database
createdb querymedic_test
psql querymedic_test -c "CREATE EXTENSION IF NOT EXISTS pg_stat_statements;"
```

## Training Launch
```bash
# SFT
CUDA_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 \
deepspeed --num_gpus=8 training/train.py \
    --deepspeed training/configs/ds_config.json \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --output-dir checkpoints/querymedic-sft

# GRPO RL (with validation agent on GPUs 16-17)
CUDA_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 \
deepspeed --num_gpus=8 training/train_rl.py \
    --model checkpoints/querymedic-sft/final \
    --reward-backend postgresql  # connects to local PG for EXPLAIN validation

# DPO
CUDA_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 \
deepspeed --num_gpus=8 training/train_dpo.py \
    --model checkpoints/querymedic-rl/final
```
