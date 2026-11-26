# Broad search, 2026-11-26 (6 parallel research subagents)

Areas and full answers: `out_<area>.md`; shared context and rules: `common.md`. Main-session spot checks below.

## Ranked shortlist (nothing came back "strong")

| # | Idea | Area | Verdict | Demand (checked) | One-day kill test |
|---|---|---|---|---|---|
| 1 | Host->GPU KV-load path: copy-engine stalls and per-machine DMA vs SM-copy choice; a measured cost model driving a vLLM per-step scheduler (copy path, slicing, recompute-vs-load) | serving-memory | medium | vllm#65515 (Nov 3, open): 6.6 GB load = 161k descriptors stalls the worker 110 ms; slicing 706 -> 859 tok/s. vllm#57555 (Sep 18, open): H100-tuned thresholds wrong on other GPUs (A30: 9.0 vs 5.4 ms) | A100 + 4090 microbenchmark: big load + small foreground H2D copies; dead if p99 step inflation < 1.3x or fix < 10% |
| 2 | MTP head as a resident PL drafter on the KV260 (Qwen3.5 ships one), verify k+1 tokens per weight pass | wildcard | medium | vLLM forum (Nov 19): Qwen3.5-9B MTP 1,067 vs 565 tok/s on A100, tools don't enable it. HF discussion #15 not readable (unverified) | A100 CPU/eager: MTP acceptance k=1..4 with nested 8-bit weights; KV260 T=4 verify cost; dead if < 1.3x or verify > 1.5x a 1-token step |
| 3 | Idle-aware KV260 LLM engine: keep weights/state in DDR, power the PL down between agent/voice bursts, co-design resume latency | edge-agents | medium | ollama#17958 (Aug 23, open): ~25 W extra idle on Strix Halo with 2 resident models. Tandem 2610.26660 (phone NPU keep-alive energy) | KV260 power rig: 5 states (idle, clock-gated, unloaded, suspend, off) + resume time; dead if < ~1 W saved or > ~300 ms resume |
| 4 | Dual-system VLA on one Versal: VLM on AIE, action head on PL with bounded jitter (narrower M) | robotics | medium | Figure Helix (two GPUs), DASH 2611.11854, Dubot 2611.30177, openpi#393 | 2x4090: action loop as a priority stream next to VLM prefills; dead if software keeps p99 <= 1.2x |
| 5 | Exact MXFP4 decode datapath on FPGA (narrowest exact window, reject inexact blocks) | numerics | weak-medium | vllm#66074 (13/34 checkpoints hit) | scale-histogram survey + fake-quant KL; dead if W <= 8 gives KL < 1e-3 |
| 6 | HLS-synthesis reuse inside agentic hardware-design RL | wildcard | weak-medium | Gradyn Labs, SiliconGym 2611.25658 | dead if synthesis < 30% of trajectory time |

## Dropped after checking
- Multi-LoRA on a low-bit FPGA base (fpga-llm #1): AMD shipped Dynamic LoRA in Ryzen AI 1.9 on 2026-11-02 (Qwen2.5-3B, Qwen3-4B, Qwen3.5-4B); FastFlowLM#976 is a different runtime. Demand largely met by the vendor.
- Everything else the subagents rated weak (NPU determinism, IOMMU-aware DMA layout, KV+recurrent host tier (HiState), rising-precision action head (2611.24234), PL preprocessing, GDN divergence-point prefill (Tandem), dithered low-bit state).
- Newly taken: MXFP4 on FPGA (Cruella 2611.27323), RL train/inference consistency (vLLM blog 2026-10-14), agent sandboxes, power capping, TEE serving.

## Caveats
- #2 is close to killed #1 (nested self-speculation); the difference is the MTP drafter. Kill test must check that GDN's per-token recurrence does not make T=4 verify cost ~4x.
- #3: unknown how much of the KV260's 4.0 W is PL (vs PS, DDR, carrier board); the K26 SOM may not expose a separately switchable PL rail.
- #1 risks turning into vLLM PRs (#57569, #66511, #66527 in flight) rather than a thesis; it is the only one with a clear hardware cost model + many machines story.
