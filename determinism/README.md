# Batch-invariance kill test

`batch_invariance.py` runs the same Qwen3.5-0.8B prompt alone and as row 0 of batches containing unrelated equal-length token sequences. It compares 64-token greedy trajectories.

## RTX 4090 result

BF16, batch 4:

- 5 prompts
- 3/5 trajectories diverged
- 136/320 generated token positions differed
- first divergence occurred at token 1, 18, or 35
- maximum selected-logit difference after trajectory divergence: 14.375

Across batches 2/4/8, 7/15 comparisons diverged and 284/960 token positions differed.

FP32, batch 4:

- 0/5 trajectories diverged
- maximum selected-logit delta: 2.67e-5

This reproduces the known batch-dependence problem, but follow-up literature review shows that it does **not** establish a thesis gap. vLLM and SGLang already provide batch-invariant modes; LLM-42 provides selective verified determinism; Detent reports bitwise-replayable RL rollouts and exact rollout/trainer log-probs at about 1.03x stock rollout time; CertiSum covers selective exact accumulation for MXFP4 training across vendors/layouts.

**Final verdict: killed as the primary thesis direction.** See `BACKGROUND.md`. The local result remains useful evidence and a regression test.
