# Edge STRIDE kill test (2026-11-26, yellowstone GPU 2): killed

Idea: run STRIDE's lossless look-ahead (guess the tool call, run the tool early, decode the next turn on the guessed
result in a second stream, commit only on an exact match) on a batch-1, bandwidth-bound edge accelerator (KV260), where
a second stream shares each weight read and should cost ~+20% per step instead of competing for compute.
Novelty note (hub search): the lossless agent-fork algorithm itself is claimed by Speculative Actions (2510.04371);
GDN/Mamba state rollback by SpecMamba (2509.19873) and ReplaySSM (2610.13778). Only the edge co-design was open.

**Verdict: dead.** Even with a free second stream and free prefill, the predicted single-agent speedup is
**1.10x [1.06, 1.16] on τ²** and **1.27x [1.20, 1.35] on BFCL**, below the 1.3x threshold. At a realistic +20%
batch-2 step cost it is 1.03x / 1.10x, with energy savings of 3% / 8% (threshold 15%).

## Step 1: traces (Qwen3-4B, single agent, `d1` arm = probe + speculative dispatch, 2 s tool floor)

vLLM 0.19.1 on GPU 2, `run_4b.sh`, ETA-dev harness unchanged. Stopped early (owner/hub rule: stop if the upper CI is
below 1.3x): 26 τ² airline+retail tasks and 17 BFCL multi-turn tasks (telecom shards were killed before finishing).

| | τ² air+ret | BFCL |
|---|---|---|
| tasks / turns / tool turns | 26 / 98 / 72 | 17 / 174 / 116 |
| exact guess (α_coverage) | 0.40 | 0.73 |
| fork-eligible (exact and no visible text before the call) | 0.40 | 0.63 |
| median decode tokens per turn | 685 | 299 |
| probe dispatch point, committed turns (fraction of the turn) | median 0.05 | median 0.13 |
| share of all decode tokens in committed tool turns | 15% | 35% |
| τ² reward (4B) | 0.12 | - |

The guess comes very early (5-13% into the turn), so the fork could overlap almost the whole parent turn. That is not
the limit. The limit is how much of the work sits in committed turns: on τ², 59% of decode tokens are in reply or
no-tool turns and 25% in missed tool turns; on BFCL a committed turn is followed by an equally short next turn.

## Step 2: KV260 replay (`kv260_replay.py`, no GPU)

Re-costs every recorded turn on a batch-1 device. Per step: t1 = 58 ms (measured KV260 8/6); a two-stream step costs
t1·(1+s); prefill at `prate` tok/s; the probe is a 24-step side stream; energy = 4.0 W idle + 2.69 W dynamic while
computing (0.388 J/token measured). Depth 1 (at most one unverified turn). Ratio of sums over tasks, 2,000 bootstrap
resamples, 95% CI. Full grid: `results/kv260_replay_out.txt`.

| Setting (T = 2 s tools) | τ² time | τ² energy saving | BFCL time | BFCL energy saving |
|---|---|---|---|---|
| baseline (tool speculation only) | 1.01x | ~0% | 1.03x | +2% |
| LA, s = 0, free prefill (ideal upper bound) | 1.10x [1.06, 1.16] | +9% [+6, +13] | 1.27x [1.20, 1.35] | +20% [+16, +25] |
| LA, s = 0.1 | 1.06x [1.03, 1.10] | +6% | 1.18x [1.11, 1.26] | +14% |
| LA, s = 0.2 (KV260 estimate: 50 ms weights + 10 ms per stream) | 1.03x [1.00, 1.06] | +3% | 1.10x [1.03, 1.18] | +8% |
| LA, s = 0.5 | 0.94x | -7% | 0.92x | -10% |

The ideal bound does not move with tool latency (T = 0 / 2 / 5 s gives 1.10 / 1.10 / 1.11 on τ²), because LA hides
decode, not tool time. More tool latency only helps the baseline.

## Why the datacenter number (1.49-1.73x for Qwen3-32B) does not carry over

1. **Small models guess worse:** τ² exact 0.40 for 4B vs ~0.76 for 32B (BFCL 0.73 vs 0.70: no gain where it matters).
2. **Depth 1 caps the overlap** at min(parent remaining, next turn). The datacenter headline used depth 2, which on
   the edge needs three streams (s grows) and a deeper guess chain (α^2 = 0.16 on τ²).
3. **The batch-2 step is not free on the KV260.** ~10-12 ms of each 58 ms step is per-stream (DeltaNet recurrence, 9.4 MB
   FP16 state read and written, vector ops), so s ≈ 0.2 even before attention and KV traffic.

## Caveats

- Qwen3-4B stands in for an edge-size model; the KV260 runs Qwen3.5-0.8B, which would likely guess worse still.
- Token counts come from GPU traces; KV260 decode would produce the same tokens (greedy is not guaranteed equal, but
  the counts are what matter).
- Not measured: multi-candidate forks (several guesses in parallel streams; the τ² name-only hit rate is higher) and
  depth 2. Both need s to stay near 0 with 3 or more streams, which the per-stream fixed cost rules out.
