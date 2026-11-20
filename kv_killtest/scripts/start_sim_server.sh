#!/usr/bin/env bash
# Fixed tau2 user simulator + NL-assertion judge: Qwen3-8B, BF16 weights, BF16 KV, thinking off.
HERE=$(cd "$(dirname "$0")/.." && pwd)
SPORT=${SPORT:-1054}
mkdir -p "$HERE/runs"
exec "$HERE/.venv/bin/vllm" serve Qwen/Qwen3-8B --port "$SPORT" --dtype bfloat16 \
  --kv-cache-dtype auto --seed 0 --max-model-len 40960 \
  --gpu-memory-utilization ${GPU_UTIL:-0.25} \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --default-chat-template-kwargs '{"enable_thinking": false}' > "$HERE/runs/vllm_sim_server.log" 2>&1
