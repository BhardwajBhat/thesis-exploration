#!/usr/bin/env bash
# Full BFCL multi-turn + memory_kv sweep, thinking off, BF16 weights, KV dtype varied.
# bf16 is run twice (seed 0 and 1) to measure run-to-run noise.
set -uo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
export CATEGORIES=multi_turn_base,multi_turn_long_context,memory_kv THREADS=32 GPU_UTIL=0.8
for M in 8B 4B; do
  for cfg in "auto 0 bf16_s0" "nvfp4 0 nvfp4_s0" "fp8 0 fp8_s0" "auto 1 bf16_s1"; do
    set -- $cfg
    TAG=bfcl_${M}_$3
    [[ -s $HERE/runs/$TAG/evaluate.log ]] && { echo "skip $TAG"; continue; }
    echo "=== $TAG start $(date +%T)"
    "$HERE/scripts/run_bfcl.sh" Qwen/Qwen3-$M Qwen/Qwen3-$M-FC-nothink "$1" "$2" "$TAG"
    echo "=== $TAG done $(date +%T)"
  done
done
