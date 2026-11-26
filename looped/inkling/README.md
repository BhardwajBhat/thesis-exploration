# Inkling 2 Mini: expert overlap across loop passes (2026-11-26, yellowstone GPU 2)

Question 1 of `../INKLING.md`: does a token reuse its experts on passes 2 and 3 of the loop (layers 6-17)?
**Verdict: dead by the pre-set criterion.** Mean unique experts per token per looped layer over the 3 passes is
**14.1** (floor 8, disjoint 24, random 22.5); the kill line was ~12. Already published in more depth (see Prior work).

## Setup

- `thinkingmachines/Inkling-2-Mini` (official, 20 GB NVFP4/MXFP8). transformers 5.24.1 has no `modelopt` quantizer, so it
  dequantizes on load: **52.8 GB BF16** on one A100 80GB. No vLLM: Inkling needs vLLM >= 0.35 nightly (CUDA 13,
  driver r580+); yellowstone has driver 535. Venv `~/.venvs/inkling` (torch 2.10 cu128, transformers 5.24.1).
- 16 prompts, 4 each of chat / code / math / tool-call (4 tools in the chat template). Greedy generation of up to 384
  tokens, then one teacher-forced forward over prompt + response with hooks on every router. A looped router is called 3
  times per forward (asserted), so the call order gives the pass. 6,779 tokens x 12 layers = 81,348 token-layers.
- Responses are correct and the tool calls are well formed (`results/inkling_expert_overlap.json`, `items`).
- Speed: about 5 tok/s generation (transformers reference MoE loops over experts in Python, about 28% GPU utilization).
  The measurement itself is a single forward pass; larger runs should feed existing traces without generation.

Run: `CUDA_VISIBLE_DEVICES=2 python -I expert_overlap.py --out raw_routing.pt` (~15 min), then
`python -I analyze_overlap.py raw_routing.pt summary.json`. The raw routing file (19 MB) is not committed.

## Results (all 81,348 token-layers)

| Metric | Value |
|---|---|
| **Unique experts over 3 passes** (8 = same, 24 = disjoint, random 22.5) | **14.11** (chat 14.1, code 14.1, math 13.3, tool 14.7) |
| Experts shared, pass 1 vs 2 / 2 vs 3 / 1 vs 3 (of 8) | 4.73 / 4.67 / 3.65 |
| Jaccard 1-2 / 2-3 / 1-3 | 0.451 / 0.445 / 0.324 |
| Identical expert set, 1-2 / 2-3 / all three | 1.6% / 2.0% / 0.2% |
| Baseline: next token, same layer, pass 1 (normal MoE temporal reuse) | 3.74 of 8 shared |
| Pass-1 top-K ranking recall of pass-2 experts, K = 8/12/16/24/32 | 0.59 / 0.71 / 0.78 / 0.85 / 0.89 |
| Same for pass-3 experts | 0.46 / 0.57 / 0.64 / 0.72 / 0.77 |
| Routing-weight mass of pass 2 / pass 3 covered by pass-1 set | 0.65 / 0.50 |
| Distinct experts per layer over a window of 1 / 4 / 16 / 64 tokens, pass 1 only vs all passes | 8 vs 14.1 / 18.4 vs 30.8 / 37.6 vs 57.6 / 65.1 vs 88.5 |
| Per layer (6 ... 17) | 13.1-15.0, except **layer 17 = 18.1** |
| Prompt vs generated tokens | 14.3 vs 14.0 |

Distribution of unique experts: mode 13, only 0.15% of token-layers at 8, 1.6% at 9, 19% above 16.

## What it means

- **Looping costs real expert traffic.** A token touches ~14 distinct experts per looped layer, not 8. Over a 4-token
  window the loop raises distinct expert loads 1.68x over pass 1 alone, over 64 tokens 1.36x (128 is the ceiling). The
  HF-card premise that unique weights fall "toward 3.4B" does not hold.
- **Exact reuse is useless (2%)**; pass 1 is only a weak predictor: prefetching pass-1's top 16 covers 78% of pass 2's
  experts and 64% of pass 3's. Normal cross-layer expert predictors already report similar recall, so this is not a new lever.
- Layer 17 is the outlier, matching the published finding that L18 (1-based) uses 2-3 near-shared experts on passes 1-2
  and spreads out on pass 3.

## Prior work (found after the plan was written)

- **noamshh.github.io/blog/inkling-loop-routing** ("Does Inkling 2 Mini use the same experts on every loop?", Nov 2026):
  14.5M prefill tokens across 23 benchmarks. 3.4-5.6 of 8 experts shared between passes (we get 4.7); fp32-vs-bf16
  rounding alone keeps 6.9-7.9; passes 2-3 agree more than with pass 1; reuse falls with position; L18 mode switch;
  message-read ablations; balancing bias appears to follow only pass 3. Far larger than this run, and consistent with it.
- How to Loop MoE / Foil (2609.35751): looped-MoE design, routing probes and expert masking.
- LoopMoE (2606.04438, EMNLP'26).
- llama.cpp #32755: wrong output when loop-layer experts are CPU-offloaded (engineering bug, not a research result).

## Not done (by instruction)

KV sharing across passes and per-token pass count (`../INKLING.md` questions 2-3) were not started.
