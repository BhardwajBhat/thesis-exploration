# Looped MoE revisited: Inkling 2 Mini (2026-11-26)

The Ouro kill test (`README.md`) only covered a small dense looped model, where looping adds compute and
nothing else. Thinking Machines' **Inkling 2 Mini** (released 2026-11-20, Apache 2.0) is a looped
**MoE** model aimed at local hardware, which raises memory-system questions the Ouro test could not.
**Update 2026-11-26: question 1 run on yellowstone, dead** (mean 14.1 unique experts over 3 passes vs kill line ~12; also already measured on 14.5M tokens by noamshh.github.io/blog/inkling-loop-routing). See `inkling/README.md`.

## Facts (official blog + HF card unless marked)

- 24 layers in four blocks of 6. Layers 0-5 run once, **6-17 run three times**, 18-23 once: 48 layer
  applications from 24 stored layers. Passes 2/3 read the previous pass's output through a learned gate.
- MoE: 26.4B total, 3.4B active, **8 of 128 experts** per layer. Each extra pass adds ~1.0B active, so
  5.4B "weight activations" per token at 3 passes. Shared experts: not stated.
- Whether the (shared) router picks different experts on each pass: **not stated**.
- Attention: hybrid local/global, context up to 1M. Secondary source only (systemdesign.academy, from
  config.json): global layers 5/11/17/23, 1,024-token window elsewhere, and in the HF reference code
  **each pass keeps its own KV copy** for the looped layers (8 growing caches; ~1.2 GB at 32K, ~35 GB at 1M).
- Checkpoint ~20 GB, mixed NVFP4 / MXFP8. Runs on DGX Spark, Macs >= 32 GB, vLLM, SGLang, llama.cpp
  (GGUF from Unsloth, 7.8-22.9 GB), Ollama. The model card says looped support in local runtimes "is new,
  and speed varies". The blog's decode ceiling counts "repeated reads across loops", so engines re-read
  the looped weights every pass.
- Fixed at 3 passes. The research checkpoint `thinkingmachines/tml-d24-varloop-research` takes 1-5
  passes chosen at inference (trained mostly at 3; may misbehave at other counts). **No adaptive exit.**

## Probably dead (back-of-envelope)

- **Cross-pass on-chip weight reuse at batch 1.** A layer is reused only after a full pass over the 12
  looped layers (~1B active params, ~0.55 GB at NVFP4). That fits in no on-chip cache (GB10 L2 is tens of
  MB). At batch 1, looping costs ~1.6x the bandwidth of a non-looped model with the same active size.
  Looping saves capacity, not bandwidth.
- **Fewer passes as the speculative draft.** A 1-pass draft costs 3.4/5.4 = 63% of a full step: the
  same reason direction #1 died on the KV260.

## Open questions (cheap to measure)

1. **Expert overlap across passes.** LOOM (2610.01153) finds shared routers give nearly invariant expert
   choices across loops; 2610.23036 finds that *forcing* pass-1 experts on later passes hurts accuracy.
   If Inkling mostly re-picks the same 8 experts, unique weights per token fall from ~5.4B toward 3.4B, and
   pass 1's routing predicts passes 2-3 exactly. That makes loop-aware expert offload/prefetch attractive
   on 8-16 GB devices (real demand: the small GGUFs). **The only one of the three that leads to a
   hardware / memory-hierarchy thesis.**
2. **KV sharing across passes.** If each pass keeps its own KV and two of the four global layers sit in
   the loop, long-context KV roughly doubles. How much quality does sharing (e.g. reusing the last pass's
   KV) cost? Ouro's paper may already report this for a dense model: check before claiming novelty.
3. **Per-token pass count.** Using the variable-loop checkpoint, measure the oracle number of passes
   needed to match the 3-pass output on agentic and reasoning tasks. With MoE, every skipped pass also
   skips its expert reads.

## Proposed first experiment

One yellowstone A100 (GPU 2 or 3; BF16 ~53 GB fits). Hook the router and attention in HF transformers;
prompts: chat, code, BFCL tool calls, math.

Kill criteria:

| Question | Dead if |
|---|---|
| 1. Expert overlap | mean unique experts per token per looped layer over 3 passes > ~12 (floor 8, disjoint 24) |
| 2. KV sharing | sharing KV across passes costs > ~1 pt on the task set |
| 3. Pass count | a cheap gate cannot reach <= 2.6 mean passes at >= 99% agreement with 3 passes |

Also re-run a novelty search first: the model is days old and papers on it will appear quickly.

## Sources

- https://thinkingmachines.ai/news/inkling-2-mini/
- https://huggingface.co/thinkingmachines/Inkling-2-Mini
- https://thinkingmachines.ai/model-card/tml-d24-varloop-research/
- https://www.systemdesign.academy/explained/inkling-2-mini (KV-per-pass claim, secondary)
- LOOM 2610.01153; Towards Looped Models Done Right 2610.23036; Scaling Laws for Looped MoE 2609.40316;
  Allocating Recurrent Compute in Looped LMs (KV260) 2608.18230; LoopFold 2610.35556
