#!/bin/bash
# Edge-STRIDE kill test, step 1: Qwen3-4B single-agent d1 traces (probe guess + timing) on tau2 air+ret, telecom, BFCL.
# Server: Qwen3-4B on GPU 2, port 8012. Shards run concurrently on that server; replay uses per-turn timings,
# and concurrency only scales them (we re-cost everything with the KV260 model anyway).
cd /data/home/bhat/clod/ETA-dev
export SPORK_TAU2_ROOT=/data/home/bhat/clod/tau2-bench SPORK_BFCL_ROOT=/data/home/bhat/clod/gorilla/berkeley-function-call-leaderboard
PY=~/.venvs/spork-eval/bin/python
OUT=~/edge_stride/results
COMMON="--model-url http://127.0.0.1:8012/v1 --model-name qwen3-4b --tokenizer-path Qwen/Qwen3-4B --configs d1 --enable-thinking --tau2-floor 2.0 --n 999 --results-root $OUT"
pids=()
for s in 0 15 30 45; do
  $PY run_eval.py $COMMON --benchmark tau2 --start $s --end $((s+15)) > ~/edge_stride/logs/tau2_$s.log 2>&1 & pids+=($!)
done
for s in 0 10; do
  SPORK_TAU2_DOMAINS=telecom $PY run_eval.py $COMMON --benchmark tau2 --start $s --end $((s+10)) > ~/edge_stride/logs/tel_$s.log 2>&1 & pids+=($!)
done
for s in 0 15 30; do
  $PY run_eval.py $COMMON --benchmark bfcl --start $s --end $((s+15)) > ~/edge_stride/logs/bfcl_$s.log 2>&1 & pids+=($!)
done
wait "${pids[@]}"
echo DONE > ~/edge_stride/logs/ALL_DONE
