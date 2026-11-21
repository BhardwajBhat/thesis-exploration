# KV kill test for idea C (KV precision by prompt role)

**Question:** is uniform low-precision KV cache already near-lossless on multi-turn tool-calling and agent
benchmarks? If uniform NVFP4 KV loses about 1 point or less versus BF16, assigning precision by role has
no headroom and idea C is dead.

**Answer: C is not dead. There is room, but it is narrow.** Uniform NVFP4 KV costs Qwen3-8B about 3–3.5 pt
on BFCL multi-turn. Against the setting-matched bf16 run that difference excludes 0, so it is outside the
noise. FP8 KV is within noise on the pooled set, but it also loses 3–4 pt on `multi_turn_long_context`,
the category that matters most for C. On tau2, NVFP4 scores below both bf16 runs in every comparison, but
tau2 is too noisy at these sample sizes to settle anything. Qwen3-4B shows no effect, but it sits at the
floor (about 10%), so that is weak evidence. At batch 1, NVFP4 decode is barely faster than FP8 below about
64k context; its main advantage is capacity.

Setup: DGX Spark (GB10, sm_121), vLLM 0.34.0 with FlashInfer (xqa decode for quantized KV), BF16
weights, `--kv-cache-dtype {auto, fp8, nvfp4}`. Weights are identical across configs; only the KV cache dtype changes. Thinking is off
(empty `<think></think>` prefill, the same as `enable_thinking=False`). Each bf16 config runs twice
(seeds 0 and 1) to measure noise.
CIs are Wilson 95% for accuracy and paired bootstrap 95% (10k resamples, over identical items) for
differences.

## BFCL v4 (full sets, n=200 per multi-turn category, 155 for memory_kv)

Acc % [Wilson 95%] · paired diff vs bf16_s0 [95%] · down/up flips · diff vs bf16_s1 [95%]

**Qwen3-8B**

| category | bf16_s0 | bf16_s1 (noise) | fp8 | nvfp4 |
|---|---|---|---|---|
| multi_turn_long_context | 16.5 [12.0,22.3] | 17.5 · +1.0 [0.0,+2.5] · 0↓/2↑ | 13.5 · **−3.0** [−6.0,0.0] · 8↓/2↑ · vs s1 −4.0 [−7.0,−1.0] | 13.0 · **−3.5** [−7.5,+0.5] · 12↓/5↑ · vs s1 −4.5 [−8.5,−0.5] |
| multi_turn_base | 24.5 [19.1,30.9] | 24.5 · 0.0 [−2.0,+2.0] · 2↓/2↑ | 25.5 · +1.0 [−1.5,+4.0] · 3↓/5↑ | 22.0 · −2.5 [−7.0,+2.0] · 13↓/8↑ |
| pooled multi-turn (n=400) | 20.5 [16.8,24.7] | 21.0 · +0.5 [−0.8,+1.8] | 19.5 · −1.0 [−3.2,+1.0] · vs s1 −1.5 [−3.8,+0.8] | 17.5 · **−3.0** [−6.0,0.0] · 25↓/13↑ · vs s1 **−3.5 [−6.8,−0.5]** |
| memory_kv | 9.7 [6.0,15.3] | 5.8 · −3.9 [−7.7,0.0] | 11.0 · +1.3 | 12.3 · +2.6 |

**Qwen3-4B** (near the floor)

| category | bf16_s0 | bf16_s1 | fp8 | nvfp4 |
|---|---|---|---|---|
| multi_turn_long_context | 9.5 | 9.0 · −0.5 | 9.5 · 0.0 [−3.0,+2.5] · 4↓/4↑ | 9.0 · −0.5 [−3.5,+2.5] · 5↓/4↑ |
| multi_turn_base | 12.5 | 12.5 · 0.0 | 12.0 · −0.5 [−3.5,+2.5] | 11.0 · −1.5 [−4.5,+1.5] |
| pooled | 11.0 | 10.8 · −0.2 | 10.8 · −0.2 [−2.2,+1.8] | 10.0 · −1.0 [−3.0,+1.0] |
| memory_kv | 6.5 | 6.5 · 0.0 | 7.1 · +0.7 | 6.5 · 0.0 |

Notes:
- At a fixed dtype, multi-turn decoding is almost deterministic: only 2–4 of 200 items flip between seeds,
  so the paired test is sensitive. Changing the KV dtype at all adds perturbation flips. On 4B these are
  symmetric (4↓/4↑ on long_context). On 8B they skew downward (fp8 8↓/2↑, nvfp4 12↓/5↑), which indicates a
  real loss.
- memory_kv's noise floor (bf16 s0 vs s1: −3.9, 8↓/2↑ on 8B) is as large as any effect, so it is excluded
  from the verdict.
- Failure types (`results/bfcl_error_types.csv`): the error mix does not change with dtype. Items that fail
  under quantized KV but pass in both bf16 runs are all `instance_state_mismatch` or
  `execution_response_mismatch`, plus one empty turn. There are no new format or parse failures, so the
  losses are wrong actions or state (reasoning / state tracking), not broken schema copying. nvfp4 has
  13 such items on base and 12 on long_context; fp8 has 3 and 8.

## tau2-bench (Qwen3-8B agent, retail base n=114, airline base n=50, 1 trial)

The user simulator and the retail NL-assertion judge are a fixed local Qwen3-8B server (bf16 KV, thinking
off); no paid APIs are used. Communicate scoring uses substring matching (the original scorer). The
reward is the task's own reward basis. "DB-only" counts only a final DB-state match (a missing DB check
counts as 0).

| | bf16_s0 | bf16_s1 | fp8 | nvfp4 |
|---|---|---|---|---|
| retail reward | 15.8 [10.2,23.6] | 23.7 · +7.9 [+0.9,+14.9] vs s0 | 15.8 · vs s0 0.0 [−7.9,+7.0] · vs s1 −7.9 [−15.8,0.0] | 11.4 · vs s0 −4.4 [−12.3,+2.6] · vs s1 **−12.3 [−19.3,−5.3]** |
| retail DB-only | 21.1 | 24.6 · +3.5 | 20.2 · −0.9 / −4.4 | 16.7 · −4.4 / −7.9 [−15.8,0.0] |
| airline reward | 22.0 [12.8,35.2] | 22.0 · 0.0 [−10,+10] | 16.0 · −6.0 / −6.0 | 16.0 · −6.0 [−18,+6] / −6.0 |
| airline DB-only | 24.0 | 24.0 | 20.0 · −4.0 / −4.0 | 18.0 · −6.0 / −6.0 |

The tau2 noise floor is large: the two bf16 runs differ by +7.9 on retail. bf16_s1 is the setting-matched
baseline for fp8 and nvfp4 (see caveats). nvfp4 is below both bf16 runs on every metric. fp8 is mixed.

## Batch-1 decode speed (Qwen3-8B, BF16 weights)

| ctx | bf16 tok/s | fp8 | nvfp4 | nvfp4 / fp8 |
|---|---|---|---|---|
| 8k | 13.18 | 13.64 | 13.77 | 1.01× |
| 32k | 10.97 | 12.36 | 12.84 | 1.04× |
| 130k | 6.59 | 9.06 | 10.40 | 1.15× |

KV pool at 0.6 GPU utilization: bf16 393k tokens, fp8 773k, nvfp4 1.37M. At batch 1, the 16 GB of
BF16 weights dominate the bytes read per decoded token, so NVFP4 over FP8 only helps at very long
context. Its real advantage is about 1.8× FP8's capacity (more sessions or longer contexts). This was
not measured with quantized weights, which would raise the KV share of traffic. The 130k point uses
130,000 tokens with YaRN ×4, which is enabled for all speed points.

## Verdict against the kill criteria

- **"C is dead" (nvfp4 within noise or within 1 pt on BFCL multi-turn AND tau2): not met.** On 8B,
  pooled BFCL is −3.0 / −3.5 pt and the CI against bf16_s1 excludes 0. tau2 is negative in every
  comparison.
- **"Room" (nvfp4 down ≥3 pt while fp8 stays lossless): borderline.** On pooled BFCL this holds:
  nvfp4 −3.0 to −3.5, fp8 −1.0 to −1.5 (within noise). On `multi_turn_long_context`, though, fp8 is not
  lossless (−3.0 / −4.0), so part of the long-context damage comes from any KV quantization, not from
  4-bit specifically.
- **Implication for C:** the gap between uniform nvfp4 and bf16 is about 3–4.5 pt on 8B multi-turn, and
  the failures are state and reasoning errors spread across turns. Role-based precision is worth one
  targeted follow-up test: keep tool-schema, system or recent turns in bf16/fp8 and put older history in
  nvfp4, and see whether that recovers the long-context loss. On this hardware the motivation is mainly
  capacity, not batch-1 speed.

## Caveats

- Thinking is off; BFCL's stock Qwen3 handler would think (about 15× slower). Absolute accuracies are
  therefore low (8B: 16–25% BFCL multi-turn, 16–24% tau2), which limits statistical power.
- **vLLM bug:** with vLLM 0.34 on sm_121, fp8 and nvfp4 KV crash with a CUDA illegal memory access under
  tau2's chat/tool workload when async scheduling is on; bf16 does not crash, and BFCL's completions
  workload did not trigger it. A repro on a side server showed that prefix caching off and eager mode both
  still crash, while `--no-async-scheduling` fixes it. tau2 fp8, nvfp4 and bf16_s1 ran with
  `--no-async-scheduling`. bf16_s0 ran before the bug was found, with async on. The crashed runs
  (100% infrastructure errors) were archived and excluded.
- tau2 uses `--max-steps 120` (default 200) for fp8, nvfp4 and bf16_s1. In bf16_s0 (run at 200), every
  trajectory longer than 118 messages was a loop that hit the cap with reward 0; the longest success was
  64 messages. The analysis applies the same cap to bf16_s0, with no score changes. Because of this and
  the async difference, bf16_s1 is the setting-matched tau2 baseline.
- BFCL GPU utilization was raised from 0.45 to 0.8 partway through (the KV pool saturated on
  long_context). This changes only scheduling and preemption, not the math. bf16 8B seed 0 ran its first
  273 entries at 0.45.
- Patches against upstream (in `patches/`): `gorilla_qwen3_nothink.patch` (gorilla 6ea5797) registers
  Qwen/Qwen3-4B and adds no-think FC handlers. `tau2_local_nl_judge.patch` (tau2-bench c06b07d) lets
  `TAU2_NL_JUDGE` point the NL-assertion judge at a local model.
- memory_kv covers only the KV backend (vector and rec_sum were skipped for time).

## Files

- `scripts/pipeline.sh`: whole resumable pipeline (BFCL → speed → tau2). `run_bfcl.sh`, `run_tau2.sh`,
  `start_sim_server.sh`, `queue_bfcl.sh` are its steps. `decode_speed.py` is the speed benchmark,
  `analyze.py` the statistics, and `kv_smoke.py` checks the dtypes.
- `results/summary.csv|json`: per benchmark × model × config: n, accuracy, CI, paired diffs vs bf16_s0 and
  bf16_s1, flips. `per_item.csv`: every item's score. `bfcl_error_types.csv`. `decode_speed.csv`.
- Raw runs (server logs, BFCL results, tau2 trajectories) are kept on the Spark under `kv_killtest/runs/`
  and `~/clod/tau2-bench/data/simulations/kvkill_*`; they are not committed.
