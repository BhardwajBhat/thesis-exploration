# Background review: batch-invariant inference, RL, and speculation (2026-11-25)

## Current software solutions

### vLLM

vLLM has beta batch invariance enabled with `VLLM_BATCH_INVARIANT=1`. It uses deterministic kernels, disables incompatible optimizations (custom all-reduce, sequence parallelism / async TP paths), and on tuned Ada/Hopper/Blackwell BF16 linear layers uses breakable CUDA graphs so tile configurations follow runtime shapes. It supports a broad list of dense, MoE, VLM, and quantized models on compute capability >= 8.0. The reproducibility guarantee is scoped to the same hardware and vLLM version.

Sources:
- https://docs.vllm.ai/en/v0.31.0/features/batch_invariance/
- https://docs.vllm.ai/en/stable/usage/reproducibility/

### SGLang

SGLang exposes deterministic inference and batch-invariant operations. Its earlier published Qwen3-8B/H200 measurements reported roughly 24-55% slowdown depending on attention backend and workload. These numbers describe the earlier conservative deterministic kernel stack, not the current best known solution.

Sources:
- https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/sglang_for_rl.mdx
- https://github.com/lm-sys/lm-sys.github.io/blob/main/blog/2025-09-22-sglang-deterministic.md

### Thinking Machines batch-invariant kernels

The original public batch-invariance implementation fixes reduction order in matmul, attention, and normalization. The authors reported approximately 20% matmul slowdown in one test and explicitly noted the Triton implementation was not fully optimized.

Source: https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/

### LLM-42

LLM-42 runs ordinary fast decoding, periodically replays windows under a fixed shape, and commits or rolls back tokens. It avoids forcing every request through deterministic kernels. The paper reports that the then-current SGLang deterministic mode could reduce whole-batch throughput by 56% when one request required determinism; LLM-42 makes overhead proportional to deterministic traffic.

Source: https://arxiv.org/abs/2601.17768

### Detent

Detent directly removes most of the remaining thesis opportunity. It treats engine kernel selection as a program, identifies the few batch-dependent selection values, fixes only those values, and tunes them for the deployment. It reports batch-invariant vLLM and SGLang serving at 1.04-1.17x stock latency, versus 1.67-10.3x for conservative deterministic modes. More importantly for RL, seeded rollouts replay bit-for-bit under different batches and cache splits, and a trainer using the same kernels/selection values matches every rollout log-prob bit-for-bit at 1.03x stock rollout time per token.

Source: https://arxiv.org/abs/2611.10611

### CertiSum

For low-precision training across vendors/layouts, CertiSum certifies when native MXFP4 accumulation is already exact and selectively recomputes uncertified outputs. It demonstrates bitwise-identical training across AMD MI350X and NVIDIA GB200 and changes in parallelism layout. This also occupies much of the selective exact-accumulation angle.

Source: https://arxiv.org/abs/2611.21584

## What RL systems actually do

There are several different goals, and production/research stacks do not always demand bitwise-identical end-to-end training.

1. **Ordinary high-throughput training:** accept numerical rollout/trainer mismatch, log rollout probabilities, and use PPO/GRPO-style correction or truncated importance sampling. In verl, `calculate_log_probs=True` measures rollout/training differences; values above roughly 0.01 are treated as a possible precision/configuration problem.
2. **Reduce avoidable mismatch:** align temperature, top-k/top-p support, precision, attention backend, normalization, MoE routing, and policy weights. vLLM sampling masks can replay the actual truncated sampling support.
3. **Correct residual mismatch:** use rollout log-probs as the behavior-policy denominator and apply importance weighting, clipping, rejection sampling, or TIS. This controls bias/variance but does not create bitwise equality.
4. **Debugging/exact replay:** enable vLLM batch invariance, SGLang deterministic inference, or serialize requests (`max_num_seqs=1`) when a model/backend is unsupported.
5. **Full RL reproducibility:** verl notes that engine-level determinism is insufficient by itself. Asynchronous rollout completion and collection order can still vary. Its exact actor/rollout comparison also requires aligned kernels and fp32 sampler/log-softmax conventions. Mainline V1 does not guarantee end-to-end reproducibility solely from `full_determinism`.

Sources:
- https://github.com/verl-project/verl/blob/main/docs/advance/determinism.md
- https://verl.readthedocs.io/en/latest/algo/rollout_corr.html
- https://verl.readthedocs.io/en/latest/faq/faq.html
- https://docs.vllm.ai/en/latest/training/rollout_artifacts/sampling_mask/

## What speculative decoding systems do

Speculative decoding is algorithmically distribution-preserving: a target model verifies draft tokens and applies the acceptance rule. In practice, “lossless” is scoped to floating-point precision and a fixed runtime configuration; it is not automatically bitwise invariant across batch shapes.

For normal serving, users generally accept this numerical interpretation. For exact replay or debugging, they combine fixed seeds with batch-invariant target execution. LLM-42 repurposes speculative verification itself to enforce deterministic output. Thus deterministic verification is already a published software design, not an open hardware mechanism by itself.

## Verdict

The local BF16 experiment correctly reproduces a known problem, but it does not identify a thesis gap.

- vLLM and SGLang already provide batch-invariant modes.
- LLM-42 already provides selective verified determinism.
- Detent achieves near-stock batch invariance and exact RL rollout/trainer log-probs at about 1.03x rollout cost.
- CertiSum covers selective certification/recomputation for exact low-precision accumulation across platforms.

A fixed-order accumulator or selective deterministic verifier would now need to beat Detent's ~3% RL overhead, broaden guarantees beyond same-hardware/software execution, and demonstrate a compelling hardware-specific benefit. That is not a realistic master's-thesis opening with the available hardware.

**Recommendation: kill deterministic low-precision arithmetic as the primary thesis direction.** Keep the local experiment as supporting evidence/background, not as a front-runner.
