#!/usr/bin/env bash
# One BFCL run = one vLLM server with a fixed KV-cache dtype + seed, then BFCL generate + evaluate.
# Usage: run_bfcl.sh <hf_model> <bfcl_model_key> <kv_dtype> <seed> <tag> [ids_file]
#   kv_dtype: auto (=bf16) | fp8 | nvfp4
#   ids_file: optional test_case_ids_to_generate.json (subsample); otherwise CATEGORIES below
# Weights are always BF16; only --kv-cache-dtype changes between configs.
set -euo pipefail
HF_MODEL=$1; BFCL_MODEL=$2; KV=$3; SEED=$4; TAG=$5; IDS=${6:-}
CATEGORIES=${CATEGORIES:-multi_turn_base,multi_turn_long_context}
THREADS=${THREADS:-32}
PORT=${PORT:-1053}
HERE=$(cd "$(dirname "$0")/.." && pwd)
RUN=$HERE/runs/$TAG
mkdir -p "$RUN"
export BFCL_PROJECT_ROOT=$RUN LOCAL_SERVER_ENDPOINT=localhost LOCAL_SERVER_PORT=$PORT

"$HERE/.venv/bin/vllm" serve "$HF_MODEL" --port "$PORT" --dtype bfloat16 \
  --kv-cache-dtype "$KV" --seed "$SEED" --max-model-len 40960 \
  --gpu-memory-utilization ${GPU_UTIL:-0.6} > "$RUN/vllm_server.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null; wait $SRV 2>/dev/null || true' EXIT
for i in $(seq 1 180); do
  curl -sf "localhost:$PORT/health" >/dev/null && break
  kill -0 $SRV 2>/dev/null || { echo "server died"; tail -30 "$RUN/vllm_server.log"; exit 1; }
  sleep 5
done
grep -E "GPU KV cache size|kv_cache_dtype=" "$RUN/vllm_server.log" | head -3 || true

START=$(date +%s)
cd "$RUN"
if [[ -n "$IDS" ]]; then
  cp "$IDS" "$RUN/test_case_ids_to_generate.json"
  "$HERE/bfcl_env/.venv/bin/bfcl" generate --model "$BFCL_MODEL" --run-ids \
    --skip-server-setup --num-threads "$THREADS" > "$RUN/generate.log" 2>&1
else
  "$HERE/bfcl_env/.venv/bin/bfcl" generate --model "$BFCL_MODEL" --test-category "$CATEGORIES" \
    --skip-server-setup --num-threads "$THREADS" > "$RUN/generate.log" 2>&1
fi
echo "generate_seconds=$(( $(date +%s) - START ))" | tee "$RUN/timing.txt"
"$HERE/bfcl_env/.venv/bin/bfcl" evaluate --model "$BFCL_MODEL" ${IDS:+--partial-eval} \
  --test-category "$CATEGORIES" > "$RUN/evaluate.log" 2>&1 || true
tail -5 "$RUN/evaluate.log"
