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

This establishes real demand for batch-invariant arithmetic in low precision on the local hardware. It does **not** yet establish a thesis contribution: this Transformers path uses reference PyTorch Gated DeltaNet kernels rather than vLLM/FlashInfer, and no deterministic-BF16 implementation or overhead measurement has been built yet.

Next gate: reproduce with production kernels and compare against current software batch-invariant modes. Kill if production divergence is absent or deterministic software overhead is below about 5%.
