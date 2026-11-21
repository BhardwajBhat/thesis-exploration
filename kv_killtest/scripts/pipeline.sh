#!/usr/bin/env bash
# Whole kill-test pipeline, resumable (finished steps are skipped):
#   1. BFCL sweep (8B then 4B; 4 KV configs each)
#   2. batch-1 decode speed vs KV dtype (Qwen3-8B; idle GPU)
#   3. tau2 retail + airline, 8B agent, 4 KV configs; fixed bf16 simulator/judge server
set -uo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd); cd "$HERE"
log() { echo "=== $* $(date '+%F %T')"; }

log "PIPELINE start"
scripts/queue_bfcl.sh
log "BFCL done"

mkdir -p results
if [[ ! -s results/decode_speed.csv ]]; then
  for kv in auto fp8 nvfp4; do
    .venv/bin/python scripts/decode_speed.py Qwen/Qwen3-8B $kv results/decode_speed.csv \
      8192,32768,130000 > runs/speed_$kv.log 2>&1 || log "SPEED $kv FAILED"
  done
fi
log "SPEED done"

GPU_UTIL=0.3 setsid scripts/start_sim_server.sh &
for i in $(seq 1 180); do curl -sf localhost:1054/health >/dev/null && break; sleep 5; done
curl -sf localhost:1054/health >/dev/null || { log "SIM server failed"; exit 1; }
log "SIM up"
for cfg in "auto 0 bf16_s0" "nvfp4 0 nvfp4_s0" "fp8 0 fp8_s0" "auto 1 bf16_s1"; do
  set -- $cfg
  for dom in retail airline; do
    [[ -s ~/clod/tau2-bench/data/simulations/kvkill_tau2_8B_$3_$dom/results.json ]] && { log "skip tau2 $3 $dom"; continue; }
    log "TAU2 8B $3 $dom start"
    GPU_UTIL=0.3 CONC=16 scripts/run_tau2.sh Qwen/Qwen3-8B "$1" "$2" "tau2_8B_$3" "$dom" > /dev/null 2>&1 \
      || log "TAU2 8B $3 $dom FAILED"
    log "TAU2 8B $3 $dom done"
  done
done
pkill -f -- "--port 1054" || true
log "PIPELINE done"
