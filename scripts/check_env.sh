#!/usr/bin/env bash
# check_env.sh — Verify QueryMedic environment before training

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass() { echo -e "${GREEN}[PASS]${NC} $1"; }
fail() { echo -e "${RED}[FAIL]${NC} $1"; ERRORS=$((ERRORS+1)); }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }

ERRORS=0

echo "QueryMedic Environment Check"
echo "============================"
echo ""

# ── Python packages ──────────────────────────────────────────
echo "Checking Python packages..."
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; pass('PyTorch CUDA')" 2>/dev/null && pass "PyTorch + CUDA" || fail "PyTorch CUDA not available"
python -c "import transformers; print(f'transformers {transformers.__version__}')" 2>/dev/null && pass "transformers" || fail "transformers not installed"
python -c "import peft" 2>/dev/null && pass "peft" || fail "peft not installed"
python -c "import trl" 2>/dev/null && pass "trl" || fail "trl not installed"
python -c "import deepspeed" 2>/dev/null && pass "deepspeed" || fail "deepspeed not installed"
python -c "import sqlglot" 2>/dev/null && pass "sqlglot" || fail "sqlglot not installed"
python -c "import sqlparse" 2>/dev/null && pass "sqlparse" || fail "sqlparse not installed"
python -c "import anthropic" 2>/dev/null && pass "anthropic SDK" || warn "anthropic not installed (API fallback disabled)"
echo ""

# ── GPU availability ─────────────────────────────────────────
echo "Checking GPU availability..."
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
if [ "$GPU_COUNT" -ge 14 ]; then
    pass "GPU count: $GPU_COUNT (sufficient for ZeRO-3 training)"
elif [ "$GPU_COUNT" -ge 4 ]; then
    warn "GPU count: $GPU_COUNT (training possible but slower)"
elif [ "$GPU_COUNT" -ge 1 ]; then
    warn "GPU count: $GPU_COUNT (single-GPU mode — use --num_gpus 1)"
else
    fail "No GPUs detected"
fi

VRAM=$(python -c "
import torch
if torch.cuda.is_available():
    gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f'{gb:.0f}GB')
else:
    print('N/A')
" 2>/dev/null || echo "N/A")
echo "  VRAM (GPU 0): $VRAM"
echo ""

# ── Environment variables ─────────────────────────────────────
echo "Checking environment variables..."
[ -n "${ANTHROPIC_API_KEY:-}" ] && pass "ANTHROPIC_API_KEY set" || warn "ANTHROPIC_API_KEY not set (API synthesis disabled)"
[ -n "${WANDB_API_KEY:-}" ] && pass "WANDB_API_KEY set" || warn "WANDB_API_KEY not set (W&B logging disabled)"
[ -n "${HF_TOKEN:-}" ] && pass "HF_TOKEN set" || warn "HF_TOKEN not set (private model access disabled)"
[ -n "${POSTGRES_URL:-}" ] && pass "POSTGRES_URL set" || warn "POSTGRES_URL not set (live validation disabled)"
[ -n "${MYSQL_URL:-}" ] && pass "MYSQL_URL set" || warn "MYSQL_URL not set (MySQL validation disabled)"
echo ""

# ── DeepSpeed ────────────────────────────────────────────────
echo "Checking DeepSpeed..."
if command -v deepspeed &>/dev/null; then
    DS_VERSION=$(deepspeed --version 2>/dev/null | head -1)
    pass "deepspeed CLI: $DS_VERSION"
else
    fail "deepspeed CLI not found"
fi
echo ""

# ── Storage ──────────────────────────────────────────────────
echo "Checking storage..."
FREE_GB=$(df -BG . | tail -1 | awk '{print $4}' | tr -d 'G')
if [ "$FREE_GB" -ge 100 ]; then
    pass "Free disk space: ${FREE_GB}GB"
elif [ "$FREE_GB" -ge 50 ]; then
    warn "Free disk space: ${FREE_GB}GB (minimum for training)"
else
    fail "Free disk space: ${FREE_GB}GB (need ≥100GB for model + data)"
fi
echo ""

# ── Summary ──────────────────────────────────────────────────
echo "============================"
if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}All checks passed. QueryMedic ready to train.${NC}"
else
    echo -e "${RED}$ERRORS check(s) failed. Fix above issues before training.${NC}"
    exit 1
fi
