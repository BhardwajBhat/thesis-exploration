#!/usr/bin/env bash
# tau2-bench run: agent = local vLLM server with a given KV dtype; user simulator (and the
# retail NL-assertion judge) = a separate FIXED Qwen3-8B server with bf16 KV. No paid APIs.
# Usage: run_tau2.sh <agent_hf_model> <kv_dtype> <seed> <tag> <domain> [extra tau2 args...]
# Requires the simulator server to already be running (start_sim_server.sh).
set -euo pipefail
AGENT=$1; KV=$2; SEED=$3; TAG=$4; DOMAIN=$5; shift 5
HERE=$(cd "$(dirname "$0")/.." && pwd)
APORT=${APORT:-1055}; SPORT=${SPORT:-1054}; CONC=${CONC:-16}
RUN=$HERE/runs/$TAG; mkdir -p "$RUN"
curl -sf "localhost:$SPORT/health" >/dev/null || { echo "simulator server on :$SPORT is not up"; exit 1; }

"$HERE/.venv/bin/vllm" serve "$AGENT" --port "$APORT" --dtype bfloat16 \
  --kv-cache-dtype "$KV" --seed "$SEED" --max-model-len 40960 \
  --gpu-memory-utilization ${GPU_UTIL:-0.3} \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --default-chat-template-kwargs '{"enable_thinking": false}' > "$RUN/vllm_agent_${DOMAIN}.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null; wait $SRV 2>/dev/null || true' EXIT
for i in $(seq 1 180); do
  curl -sf "localhost:$APORT/health" >/dev/null && break
  kill -0 $SRV 2>/dev/null || { echo "agent server died"; tail -30 "$RUN/vllm_agent_${DOMAIN}.log"; exit 1; }
  sleep 5
done

export HOSTED_VLLM_API_KEY=local
export TAU2_NL_JUDGE=hosted_vllm/Qwen/Qwen3-8B
export TAU2_NL_JUDGE_ARGS="{\"api_base\": \"http://localhost:$SPORT/v1\", \"temperature\": 0}"
START=$(date +%s)
"$HERE/tau2_env/.venv/bin/tau2" run --benchmark none --domain "$DOMAIN" \
  --agent-llm "hosted_vllm/$AGENT" \
  --agent-llm-args "{\"api_base\": \"http://localhost:$APORT/v1\", \"temperature\": 0}" \
  --user-llm hosted_vllm/Qwen/Qwen3-8B \
  --user-llm-args "{\"api_base\": \"http://localhost:$SPORT/v1\", \"temperature\": 0}" \
  --communicate-scorer substring --num-trials 1 --seed 300 \
  --max-concurrency "$CONC" --save-to "kvkill_${TAG}_${DOMAIN}" "$@" > "$RUN/tau2_${DOMAIN}.log" 2>&1
echo "${DOMAIN}_seconds=$(( $(date +%s) - START ))" | tee -a "$RUN/timing.txt"
tail -25 "$RUN/tau2_${DOMAIN}.log"
