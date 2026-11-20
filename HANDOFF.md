# Thesis exploration: handoff from the cloud session (2026-11-19)

Source: "Master's thesis topic exploration" cloud session (session_01EcQFdq7KLqJZcGc3DkVRKG), handoff part 1/2.
Part 2 (analytical-model code) goes in `pdmodel/`.

## User & goal
Master's student (UC Irvine, Sitao Huang's group; RaReCache arXiv 2610.11358 is the group's cross-model KV reuse paper). Wants a master's thesis topic at the intersection of (1) computer architecture / FPGA / heterogeneous computing and (2) AI systems (LLM serving, agents). Wants STRONG real motivation (demand), not a solution looking for a problem. Distrusts arXiv volume as a demand signal; preprints from well-known groups/authors are OK. Prefers blunt, short answers.

## User's existing assets
- **KV260 FPGA** (reactor, bare metal, JTAG): Qwen3.5-0.8B (hybrid Gated DeltaNet) decoder in ~/qwen-quant, nested-precision weights: ONE stored code read at 8/6, 6/6, 4/4, 3/3 bits via register write per token. Measured with INA260 (5.03 V rail, idle 4.0 W): 0.388 / 0.331 / 0.247 / 0.216 J/token; 58.0 / 49.9 / 38.6 / 35.1 ms/token. Agent-precision results: prefilling long tool outputs (HotpotQA) at 4-bit is free when generating at 6-8 bit; on BFCL, reading the whole prompt (schemas) at 4-bit hurts the 2B model (AST 0.497 vs 0.695 static 8/6; read4/gen8 = 0.537); switching to 8-bit only inside tool-call spans does nothing; DeltaNet state written at 4-bit is read fine at 6/8.
- **ASIC design** (asic project): two-tier dequant/encode unit. Tier 1 fixed descriptor-configured multi-format dequant (GPTQ/AWQ, MX/NVFP4, NF4, GGUF K-quants...); tier 2 programmable ISA incl. KV encoders (FP8, NVFP4, MX, KIVI, TurboQuant), bit-exact with vLLM. Tier 1 alone not novel (StreamDQ, BitMoD, etc.).
- **STRIDE** (paper written): lossless look-ahead agent-turn execution on A100s (vLLM): speculate tool result, start next turn early; hidden-state probes (Intent Gate, Readiness Gate), KV-aware fork depth. Up to 1.69x single agent, 1.56x at capacity; FAILED on SWE-bench at high load (0.37x, 1,154 vs 66 preemptions). SPORK (arXiv 2607.03333, Tsinghua) replication: batch-invariant mode gave 30/30 identical tasks at ~1.014x.
- **VCK5000** (Versal VC1902, 400 AIE, 145 INT8 TOPS, ~40-50 GB/s usable DDR4; platforms frozen at Vitis 2022.1/2022.2). Lab's HeteroVLM (arXiv 2611.23878, Sitao Huang's group; user knows the authors well) already does AIE prefill + PL GEMV decode concurrently with an on-chip scheduler on VC1902, INT8 fixed.
- **Machines**:
  - yellowstone (yellowstone.isalab.eecs.uci.edu): 4x A100 80GB; GPUs 0/1 belong to others, use 2/3, release after runs; vLLM 0.19.1, torch 2.10; ~/clod has ETA-dev STRIDE code, tau2-bench, gorilla/BFCL, BCP. Session "A100 GPU allocation" / clod-b2: session_01EcQFdojxhBTjzoVtkuRBkY.
  - reactor: RTX 3090 Ti + KV260; Vivado/Vitis 2019.2-2025.1 under /tools/Xilinx, recent builds 2025.1; colleague may hold board (on loan since 2026-11-02, ask first). Session "KV260 board connection": session_01EcQFdnqRKfhKkb88yj5StD.
  - ISALab-Joshua-Tree (this machine, hub session): 2x RTX 4090; no torch/vllm in system python.
- **Pi agent** ("GPT 6.2 astra") reachable via a relay on the user's Fedora box; used as an adversarial reviewer (useful critique; fabricated one citation, "Adaptive Guard").

## Ideas killed / taken (with reason)
- **Tool-wait KV management**: taken (InferCept ICML'24, TokenCake, Continuum ICLR'26, MORI by Ion Stoica's group, vLLM RFC #57103; Microsoft M365 production study arXiv 2610.22863 confirms demand).
- **Cross-model KV for agents**: crowded (CacheBridge 2609.00891, CacheLink, Heo et al. NVIDIA 2608.03893). Hybrid-state cross-model (LatentPort, HST, 2611.12062) are from unknown/small teams -> possibly still open for RaReCache-on-hybrid.
- **KV transform engine on the offload path** (asic tier 2 as engine): pi's objection killed it — BF16 already crossed PCIe before device compresses; a fused GPU kernel may do the same. Offload-path HW precedents: CXL-SpecKV (weak, unknown authors), ShadowServe (Minlan Yu/Junchen Jiang), TRACE.
- **Nested precision on GPU**: taken (Any-Precision LLM, MatGPTQ, NestedFP, AutoEMatic NVIDIA+Tushar Krishna). Bitplane eviction under KV pressure: taken (DPS, PagedWeight UIUC).
- **Phone role-based precision**: weakened — QNN compiles per-precision graphs; weight sharing only across same weights/different shapes. DroidPrefill 2611.20772, Tandem 2610.26660, mzCache MobiCom'26 exist.
- **Power side channel of adaptive precision**: initially liked (Lejla Batina's group has an EM edge-LLM attack 2610.14215) but demand evidence thin.
- **"Phase-different quantization" alone**: now popular -> not a contribution:
  - Mix-Quant (NUS Xinchao Wang, 2605.20315): whole-prefill NVFP4 W4A4, BF16 decode, 2 weight copies, no per-segment, no tool-call breakdown, no KV error-propagation study; 8B-31B, RTX 5090/B200.
  - Disaggregated Quantization (Alistarh IST + NVIDIA, 2609.26333): NVFP4 prefill, weight-only 2.5-4b decode, best results need separate prefill weights; NO agentic/tool eval, removed tool-call traces; prefill-only quant hurt RULER more than decode-only on 7/8 models.
  - DynaCore (Duke): W8A8 prefill / W4A16 decode.
  - DuoRate 2610.13328: prefix-readable MoE weight stream, unknown group.
  - STILL OPEN: per-segment precision within prefill (schemas high, tool outputs low); KV/state error propagation; one nested copy across phases; concurrent heterogeneous P/D in different formats.

## Front-runner A (systems): deterministic, gated speculation for agentic RL rollouts
- ObsSpec (UCSC+ByteDance, 2610.15246) = STRIDE loop inside verl RL with a 0.6B world model, exact-match, 66% match, -9..14% agent latency, NO gain on math (tool 21% of latency); no gate (selective speculation listed as THEIR future work), no KV/memory, "lossless" = observation match only.
- BunshinRL (Yiying Zhang UCSD, 2611.10453): speculates tool execution only (not next LLM turn), signature match, 1.1-1.2x, no GPU KV, predictor drifts by epoch 5; abstract numbers don't match body.
- Heddle (Xin Jin PKU, 2603.28101): rollout scheduling with 0.6B length predictor, KV migration in tool gaps, up to 2.5x, no speculation/determinism/rewards.
- LLM-42 (MSR India, 2601.17768): determinism via decode-verify-rollback, no RL/multi-turn/prefix caching, global verify pause.
- **Gap**: gated speculation + KV cost of forks + bitwise determinism with prefix caching across turns. Tension: determinism delays token release -> delays tool dispatch -> eats speculation gain.
- **Kill test**: profile tool share of rollout latency in target RL tasks; 2x2 speculation x batch-invariance.

## Front-runner B (hardware, user's current favourite): DynaCore-style array, prefill and decode CONCURRENT on spatial partitions
- DynaCore (Duke, Cong Guo/Chiyue Wei/Helen Li/Yiran Chen, arXiv 2610.07443, sim + 28nm synthesis only): splittable/reshapable systolic array (32x32 cores -> up to 1x1024), reshapes per iteration (time-multiplexed, NOT spatial co-location), W8A8 prefill / W4A16 decode, single device would need both formats stored, reconfig cost unmeasured, serving eval uses STATIC device PD splits (4+4, 6+2) with 16 GB/s KV links.
- Other prior: Hybrid Systolic Array (Cornell, ISLPED'25), SPECTRA (Carloni, ICCAD'26), SISA, DUET (DAC'26, separate packages), NpuSim (SJTU IPADS 2510.05632, core-level static PD split), FlexNPU (Huawei 2606.04415, dynamic but temporal/software), NeuPIMs/Duplex/DuoPD (heterogeneous units), GPU dynamic SM splits (Bullet, DuetServe).
- **Open**: intra-array spatial partitions re-split every iteration; nested-bitplane decode; agentic/look-ahead (STRIDE) traces as the source of concurrency on edge; measured FPGA prototype. User wants dynamic switching beyond HeteroVLM's static AIE/PL split.

## Analytical model results (code in `pdmodel/`)
Roofline model. Schemes: TM (DynaCore-style time-multiplex, Trainium-style separate batches), FUSED (chunked prefill, one format; TPU/RPA style), SPATIAL (decode on partition f at W4A16, prefill on 1-f at W8A8). Metric: prefill tok/s at decode TPOT <= slack x decode-alone time. Machine balance beta = MAC/s / DRAM B/s.
1. Spatial helps only on compute-rich chips (beta >= ~30); never on DynaCore's bandwidth-rich motivating example (beta ~2: decode alone saturates the array).
2. Capacity: spatial wins 1.5-1.9x ONLY at small batch (1-8 per device) AND tight TPOT (slack <= ~2). Loose targets -> FUSED wins by 10-30%. Large batch -> FUSED wins.
3. Partition granularity G >= 16 needed (G=4 loses ~20%).
4. Energy (KV260 calibrated: model 0.386 vs measured 0.388 J/token): spatial uses 7-26% less J/token at small batch for ANY slack on edge-class designs (edge NPU -19..-26%, compute-rich ASIC up to -18%, KV260 -4..-14%); reason: decode stays W4 while prefill uses big W8 chunks; fused must pick one format; TM re-reads W8 with small chunks. VCK5000 gain small because assumed 40 W static (unmeasured guess).
5. Nested weight broadcast (one 8-bit stream feeds both partitions): only +3-5% capacity and costs energy -> dropped.
6. Multi-device (8 devices, static PD split with oracle best k vs co-location): fixed static split serves only 17-60% of oracle-static under agentic swings; spatial beats oracle-static 1.0-5.7x at tight targets, but FUSED co-location beats spatial except at ~4/device batch with slack <= 1.5 (1.65-1.82x). Conclusion: datacenter high-batch serving does NOT motivate spatial partitions; framing = "disaggregation's isolation inside one chip" for edge/interactive agents.
7. Mixed-batch penalty eta (fused efficiency vs specialised kernels): RPA paper (Google, arXiv 2604.15464, TPU7x only) gives NO mixed-batch numbers (padding to static max tokens, mixed attention relies on max KV length, block sizes tuned per length distribution). AWS Neuron NEVER mixes prefill+decode (separate batches; DI beta 1P1D). Sweep: break-even eta ~0.71-0.87 at loose slack on edge/compute-rich platforms; if fused mixed batches lose >=13-20%, spatial wins broadly. **eta is THE number the thesis hinges on.**

## Platform recommendation
ASIC-level model + synthesis at an edge-NPU balance (beta ~100-200, G >= 16) for the architecture claim; KV260 prototype for measured usefulness (single-user look-ahead agent traces, TPOT + J/task, spatial vs fused vs TM on same RTL); VCK5000 optional scale-up with HeteroVLM authors (AIE static graphs -> coarse partitions via RTP mode switching, ~10us+).

## Immediate next steps (proposed, NOT yet approved by the user)
1. **Measure eta on yellowstone GPU 2** (via clod-b2): per-layer microbenchmark of a Qwen3-8B layer, mixed batch (B decode + C-token prefill chunk) vs same tokens as separate decode-only and prefill-only steps; sweep B in {1,4,16,64}, ctx {2k,8k,16k}, C {256..2048}; split attention vs dense GEMM; ~2-3 h total, ~15 min GPU; optional e2e vLLM chunked prefill on/off (+30 min GPU). A100 result is a LOWER bound for systolic penalty (GPUs handle ragged shapes well). Better: TPU via TPU Research Cloud (vLLM-TPU RPA mixed vs specialised); best: own FPGA array.
2. **Kill test on real traces**: from STRIDE look-ahead traces, how often prefill overlaps a latency-critical decode and at what batch/slack.
3. **Write both tracks up as a doc** for a meeting with Sitao Huang / HeteroVLM authors.

Also cite in FPGA work: BitNest (Yanzhi Wang, 2610.02800). "Precision on Demand" 2611.14418 is unknown authors, sim only.

## Update 2026-11-19 (hub session)
- **B killed.** eta on GPUs: GEMMs mix for free; mixed-batch loss is only in attention and is fixable in software (4090 layer eta 0.88-0.98 at B<=4; GB10 flex mixing even faster than separate at B<=4). Energy leg fails on KV260: static power is 70-81% of window energy; task-level saving 0.2-11%; and the user's own accuracy data inverts the W4-decode/W8-prefill premise (generation needs 6-8 bit). See bench/results, pdmodel/kill_b.py.
- **D (cross-model state reuse for hybrid models): core taken.** LatentPort (2609.25053, single author), HST (2611.08133, SCUT), 2611.12062 (Hostinger Research, strongest), plus KV-Lingo (2609.32610, Apple) for transformers. Open: engine integration, Mamba-2 hybrids, transfer-vs-reprefill policy on unified memory, selective recompute with recurrent state. Demand caveat: same-model prefix reuse is already 95-98% in Spark agent serving reports.
- **C (role-aware KV precision for agents): core taken** by TriAxialKV (2605.17170, Cambridge/Imperial, NeurIPS 2026): per-role (inst/obs/tool_call/...) INT2/INT4 KV, BFCL Memory + OSWorld, B200/H100. Open: batch-1 bandwidth-bound edge decode (GB10/Mac), FP8/NVFP4 formats, small models, weight x KV precision co-design per segment, calibration-free policy, hardware encoder. Kill test: is UNIFORM NVFP4 KV already near-lossless on BFCL/tau-bench? (NVIDIA/LMSYS report <1-1.6 pt loss, no tool-calling eval.)

## Update 2026-11-19: zoom-out novelty checks (G, H, I, L, M, N)
- **G (quantized self-draft + exact verify for on-policy RL rollouts): core done** by EfficientRollout (2606.18967, FuriosaAI+Berkeley, NeurIPS'26; W4 RTN drafter in veRL GRPO, up to 24% faster rollout). Gaps: verifier is vLLM BF16 (trainer mismatch remains); no comparison vs FP8+TIS/Jet-RL/QuRL; speculation off at large batch.
- **I (determinism as a hardware property): hardware side open, motivation weakened.** No arch-venue paper on batch-invariant/exact-accumulation HW. But Detent (2611.10611, Nov 2026) gets SW batch-invariance to 1.03-1.17x. Needs cross-vendor/cross-TP or MX/NVFP4 exactness argument (CertiSum 2611.21584, TBIK ICML'26, 2609.25624 Hyesoon Kim). Note: agent says SPORK (2607.03333) is self-speculative tool-call forking, not determinism -> user's 30/30 result was SPORK run in batch-invariant mode (to confirm).
- **H (dLLMs on edge): partially done, filling fast.** DART (2601.20706), DynaTE (2610.11284), block-diffusion edge HW (2609.01084), llada.cpp NPU (2606.13740), SoloQ; "compute-bound" claim already shown (2510.04146; NVIDIA DiffusionGemma blog: 150 tok/s on Spark). Counter: open dLLMs unreliable at tool calling (2601.12979).
- **L (SDC in LLMs): mostly done** for training/single-turn inference (AEGIS OSDI'26, LLM-PRISM, TrainSDC, LLMFI ICS'26, CSNB 2610.18557, Flash-ABFT, ReaLM). Gap: SDC in multi-turn agent/tool-calling loops.
- **M (VLA on edge under deadlines): partially done, clearest opening, best lab fit.** EdgeDAE (2610.00311, ASP-DAC'27: action head on FPGA), Deltoris (2608.04428), Cornell SmolVLA on AIE (FCCM'26, ~5 s, not real-time), EagleVLA, RTC, VLASH, FirmVLA/Armory (server). Open: VLA on Versal using AIE+PL with deadline-aware 3-stage scheduling and miss-rate guarantees; HeteroVLM is the starting point. Kill test: per-stage SmolVLA/pi0 on VCK5000 vs Orin; drop if >~100 ms/chunk or no latency/W win.
- **N (full-duplex voice agents): GPU multi-session serving taken** (Metronome 2607.02640, VoxServe 2602.00269, vLLM-Omni 2610.09307). Edge/HW open (no Moshi-class on Thor/Spark/FPGA under 80 ms frames). Kill test: Moshi per-frame step on Spark, int8 vs bf16.
