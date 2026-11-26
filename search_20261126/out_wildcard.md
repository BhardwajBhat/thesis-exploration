Scanned MTP/hybrid speculative decoding, agent sandboxes, power capping, TEE, CXL, sparse-attention indexers and agentic-EDA infrastructure. Two ideas survive, and neither is strong.

**1. Native-MTP self-speculation on the KV260 (Qwen3.5-0.8B)**
- **Idea:** use the MTP head that ships inside the checkpoint (1 layer plus a trimmed LM head) as a resident drafter in the PL, and verify T=k+1 tokens per weight pass. Experiment 1 tested a nested 4-bit draft, not this head. Exp1 says 1.5x needs a draft step of ~22 ms or less. My rough estimate for the MTP draft is ~14 ms, assuming the 10–12 ms fixed cost applies. Since static power is 70-81% of energy, a speedup is close to an equal energy saving.
- **Demand:** HF Qwen3.5-0.8B discussion #15 (https://huggingface.co/Qwen/Qwen3.5-0.8B/discussions/15): 0.966 per-position acceptance, yet decode drops 342 to 128 tok/s on GPU, so the small model is overhead-bound. llama.cpp #25187 (https://github.com/ggml-org/llama.cpp/issues/25187): MTP draft LM head profiled as a large share of decode; trimming gives +5–9% end to end. The vLLM forum (https://discuss.vllm.ai/t/qwen3-5-had-a-hidden-1-9x-in-its-checkpoint-gemma-4-had-a-different-one-and-none-of-the-auto-tuners-found-either/3153) calls the Qwen3.5 MTP head a "hidden 1.9x" that tools don't turn on.
- **Prior work:**
  - Tandem (2610.26660): hybrid speculative decoding on mobile NPUs. It finds rollback cost outweighs the gain on short outputs. No FPGA.
  - SpecMamba (2509.19873, ICCAD'25): FPGA speculation for Mamba with a separate draft model. Not GDN, not MTP.
  - HybridSpec (2611.04055) and TreeWY (2608.20961): GDN verification cost and state rollback, GPU only.
  - Leaves open: an edge dataflow where the MTP head and multi-token GDN verify share one weight stream.
- **Fit:** direct reuse of your KV260 decoder, the exp1 trimmed head and the snapshot results (snapshots are already cheap).
- **Kill test (1 day):**
  - Measure MTP acceptance at k=1..4 with your nested 8-bit weights on the A100 (as a CPU/eager check, not a GPU job).
  - Time one T=4 verify step on the KV260; if the decoder can't batch yet, model it from DSP slack.
  - Dead if projected speedup is under 1.3x, or if T=4 verify costs more than 1.5x a 1-token step.
- **Verdict:** medium. Biggest risk: a reviewer sees it as killed idea #1 with a different drafter, and GDN's sequential per-token recurrence may make T=4 verify nearly 4x.

**2. Exact-reward HLS tool execution for agentic hardware-design RL**
- **Idea:** memoize Vitis HLS results and reuse the front end across pragma-only edits, then schedule synthesis against rollout generation. The hardware angle is the FPGA toolchain inside the RL loop.
- **Demand:** Gradyn Labs (https://www.gradynlabs.com/) sells chip-design RL environments with ~100 tool calls and ~10 full synthesis runs per 20–40 min episode. SiliconGym (2611.25658, NVIDIA authors) overlaps tool checks with generation for a 21.9% faster RL batch.
- **Prior work:**
  - HLS-Seek (2605.13536, ICCAD'26) avoids real synthesis with a proxy reward. That shows synthesis is the bottleneck, but it gives up exact rewards.
  - SiliconGym only overlaps work; it does no reuse or incremental synthesis.
  - HLS-Eval agentic flow (2609.09526) names tool-runtime overhead as future work.
- **Fit:** you're in an FPGA/HLS lab, A100s can generate the rollouts, Vitis is available, and it reuses STRIDE-style look-ahead.
- **Kill test (1 day):** run HLS-Eval agentic traces (a local 7–30B model in vLLM 0.19.1), log time per stage, and hash the C/pragma inputs of each synthesis call. Dead if synthesis is under 30% of trajectory time, or if exact/near-duplicate plus pragma-only calls are under 40%.
- **Verdict:** weak to medium. Biggest risk: it's more EDA systems engineering than architecture, and proxy rewards may already be "good enough."

**Closest misses (dead or crowded):**
- **Agent sandboxes:** KREX 2609.30057, TideSand 2611.31118, Readie 2611.22171, DeltaBox 2605.22781, TimeTravelContainer 2611.32255.
- **GDN spec-verify on GPU:** HybridSpec, TreeWY, Windowed-MTP 2607.21535. The MTP crashes on hybrids (vLLM #40756) are software bugs, not a research gap.
- **Power capping:** 2605.11999, 2611.05148.
- **TEE serving:** HiPerfCC-KVCache 2611.28702; you also lack CC-capable GPUs.
- **DSA indexers:** Multi-layer Indexer 2611.03768, LiteTopK 2607.11976.

Every link above was opened, except the vLLM issue numbers in the closest-misses list.
