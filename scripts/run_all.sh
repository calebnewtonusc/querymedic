#!/usr/bin/env bash
# run_all.sh — Full QueryMedic pipeline: collect → synth → train → eval

set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="querymedic-${TIMESTAMP}"
DATA_DIR="data/${RUN_NAME}"
CHECKPOINT_DIR="checkpoints/${RUN_NAME}"

echo "QueryMedic Full Pipeline Run: ${RUN_NAME}"
echo "Data:        ${DATA_DIR}"
echo "Checkpoints: ${CHECKPOINT_DIR}"
echo ""

mkdir -p "${DATA_DIR}" "${CHECKPOINT_DIR}"

# ── Step 1: Environment check ────────────────────────────────
echo "[1/6] Checking environment..."
bash scripts/check_env.sh

# ── Step 2: Collect data ─────────────────────────────────────
echo "[2/6] Collecting training data..."
python pipeline.py --collect-only

echo "  Collection complete"

# ── Step 3: Synthesize SFT pairs ─────────────────────────────
echo "[3/6] Synthesizing SFT pairs..."
python pipeline.py --synth-only

echo "  Synthesis complete"

# ── Step 4: Stage 1 — SFT training ───────────────────────────
echo "[4/6] Stage 1: SFT training..."
deepspeed --num_gpus 18 training/train.py \
    --deepspeed training/configs/deepspeed_zero3.json \
    --data-dir "${DATA_DIR}" \
    --output-dir "${CHECKPOINT_DIR}/sft"

echo "  SFT checkpoint: ${CHECKPOINT_DIR}/sft/final"

# ── Step 5: Stage 2 — GRPO RL training ───────────────────────
echo "[5/6] Stage 2: GRPO RL training..."
deepspeed --num_gpus 14 training/train_rl.py \
    --sft_checkpoint "${CHECKPOINT_DIR}/sft/final" \
    --data_path "${DATA_DIR}/sft/rl_scenarios.jsonl" \
    --output_dir "${CHECKPOINT_DIR}/rl" \
    --run_name "${RUN_NAME}-rl"

echo "  RL checkpoint: ${CHECKPOINT_DIR}/rl/final"

# ── Step 6: Stage 3 — DPO training ───────────────────────────
echo "[6/6] Stage 3: DPO training..."
deepspeed --num_gpus 14 training/train_dpo.py \
    --rl_checkpoint "${CHECKPOINT_DIR}/rl/final" \
    --data_path "${DATA_DIR}/sft/dpo_pairs.jsonl" \
    --output_dir "${CHECKPOINT_DIR}/dpo" \
    --run_name "${RUN_NAME}-dpo"

echo "  DPO checkpoint: ${CHECKPOINT_DIR}/dpo/final"

# ── Evaluation ───────────────────────────────────────────────
echo ""
echo "Running QueryBench evaluation..."
python pipeline.py --eval \
    --model "${CHECKPOINT_DIR}/dpo/final"

echo ""
echo "QueryMedic pipeline complete."
echo "Final model: ${CHECKPOINT_DIR}/dpo/final"
echo "Results:     results/${RUN_NAME}"
