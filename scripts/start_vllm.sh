#!/usr/bin/env bash
# Start 4 vLLM instances for synthesis (Qwen2.5-72B-Instruct)
# Each instance uses 4 GPUs (tensor-parallel-size 4) on 18x A6000 cluster
# Instance 1: GPUs 0,1,2,3   → port 8001
# Instance 2: GPUs 4,5,6,7   → port 8002
# Instance 3: GPUs 8,9,10,11 → port 8003
# Instance 4: GPUs 12,13,14,15 → port 8004
set -euo pipefail

MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-72B-Instruct}"
API_KEY="${VLLM_API_KEY:-synthesis}"

mkdir -p logs

echo "Starting 4 vLLM synthesis instances with $MODEL (tensor-parallel-size 4)..."

# Instance 1 — GPUs 0,1,2,3 — port 8001
CUDA_VISIBLE_DEVICES=0,1,2,3 \
	vllm serve "$MODEL" \
	--tensor-parallel-size 4 \
	--port 8001 \
	--api-key "$API_KEY" \
	--max-model-len 32768 \
	--dtype bfloat16 \
	--trust-remote-code \
	>logs/vllm_8001.log 2>&1 &
echo "Started instance 1 on GPUs 0-3, port 8001 (PID $!)"

# Instance 2 — GPUs 4,5,6,7 — port 8002
CUDA_VISIBLE_DEVICES=4,5,6,7 \
	vllm serve "$MODEL" \
	--tensor-parallel-size 4 \
	--port 8002 \
	--api-key "$API_KEY" \
	--max-model-len 32768 \
	--dtype bfloat16 \
	--trust-remote-code \
	>logs/vllm_8002.log 2>&1 &
echo "Started instance 2 on GPUs 4-7, port 8002 (PID $!)"

# Instance 3 — GPUs 8,9,10,11 — port 8003
CUDA_VISIBLE_DEVICES=8,9,10,11 \
	vllm serve "$MODEL" \
	--tensor-parallel-size 4 \
	--port 8003 \
	--api-key "$API_KEY" \
	--max-model-len 32768 \
	--dtype bfloat16 \
	--trust-remote-code \
	>logs/vllm_8003.log 2>&1 &
echo "Started instance 3 on GPUs 8-11, port 8003 (PID $!)"

# Instance 4 — GPUs 12,13,14,15 — port 8004
CUDA_VISIBLE_DEVICES=12,13,14,15 \
	vllm serve "$MODEL" \
	--tensor-parallel-size 4 \
	--port 8004 \
	--api-key "$API_KEY" \
	--max-model-len 32768 \
	--dtype bfloat16 \
	--trust-remote-code \
	>logs/vllm_8004.log 2>&1 &
echo "Started instance 4 on GPUs 12-15, port 8004 (PID $!)"

echo ""
echo "Waiting for vLLM instances to initialize (this takes ~60s for 72B model)..."
sleep 90

# Health check all instances
ALL_READY=true
for PORT in 8001 8002 8003 8004; do
	if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
		echo "[OK] Instance on port ${PORT} ready"
	else
		echo "[WARN] Instance on port ${PORT} not responding yet — check logs/vllm_${PORT}.log"
		ALL_READY=false
	fi
done

echo ""
if $ALL_READY; then
	echo "All 4 vLLM instances ready."
else
	echo "Some instances may need more time. Check logs/vllm_*.log"
fi
echo ""
echo "Synthesis URLs (pass to --vllm-urls):"
echo "  http://localhost:8001/v1"
echo "  http://localhost:8002/v1"
echo "  http://localhost:8003/v1"
echo "  http://localhost:8004/v1"
