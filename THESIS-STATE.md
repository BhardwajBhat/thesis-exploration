# Thesis exploration: full state (2026-11-25)

This is the single document to read before continuing. It consolidates `HANDOFF.md` (the chronological log, kept
for history), every experiment and every literature check up to 2026-11-25. Paper IDs like `2611.12062` are arXiv IDs;
literature claims came from research agents that opened each page, but re-check anything before citing it.

---

## 1. The user and the goal

- Master's student at UC Irvine, advisor **Sitao Huang** (FPGA / heterogeneous computing). Group papers: RaReCache
  (2610.11358, cross-model KV reuse), HeteroVLM (2611.23878, VLMs on Versal VC1902), Chimera (2611.21677, HBM multi-die FPGA
  LLM decode), TeLLMe v1/v2 (2504.16266, 2510.15926: ternary LLM on KV260, 25 tok/s at 5 W), Hummingbird (2507.03308, ICCAD'25).
- Wants a master's thesis at the intersection of **computer architecture / FPGA / heterogeneous computing** and **AI systems
  (LLM serving, agents)**.
- Requirements: **strong real demand** (not a solution looking for a problem); distrusts arXiv volume as a demand signal;
  preprints from well-known groups are fine. Prefers **blunt, short answers**. Is happy to use any machine (Spark, A100, 4090,
  KV260, VCK5000) and to work outside their current experience.
- Mood as of 2026-11-24: frustrated that many ideas got killed ("this sucks"); wants ideas that will survive.

## 2. Machines, sessions, rules

| Machine | Hardware | Notes | Session |
|---|---|---|---|
| ISALab-Joshua-Tree (hub) | 2x RTX 4090 (GPU0 free; GPU1 has another user's ~4-5 GB idle process, leave it alone) | Driver 550 / CUDA 12.4. Disk nearly full (~30 GB free of 3.6 TB). Repo at `~/clod/thesis-exploration`, venv `.venv` (torch **2.6.0+cu124**, flash-attn 2.7.4, transformers 5.20, datasets) | this hub, name `clod-2d` / `isalab-joshua-tree-inherited-swing` |
| yellowstone.isalab.eecs.uci.edu | 4x A100 80GB; **use only GPUs 2/3** (0/1 belong to others); release after runs | vLLM 0.19.1, torch 2.10; `~/clod` has ETA-dev STRIDE code, tau2-bench, gorilla/BFCL, BCP | "A100 GPU allocation" = clod-b2, `session_01EcQFdojxhBTjzoVtkuRBkY` |
| reactor | RTX 3090 Ti + **KV260** (JTAG, bare metal) | Vivado/Vitis 2019.2-2025.1 in /tools/Xilinx. **A colleague may hold the KV260 (on loan since 2026-11-02): never touch the board without asking.** User's KV260 project: `~/qwen-quant` | "KV260 board connection", `session_01EcQFdnqRKfhKkb88yj5StD` |
| spark-6b58 (DGX Spark) | GB10, 128 GB unified LPDDR5X, aarch64, driver 580 / CUDA 13, sm_121 | `bhat` not in docker group (no sudo used). Python from **mise** 3.12.16 (system python has no Python.h). eta venv: torch 2.14.1+cu130; `kv_killtest/.venv`: vLLM 0.34.0 + FlashInfer | `spark-6b58-async-summit`, `session_01EcQFdrHmRxfcAQvrMmxr9j` |
| VCK5000 | Versal VC1902 (400 AIE), ~40-50 GB/s DDR4, platforms frozen at Vitis 2022.1/2022.2 | availability unknown | none |
| Thesis cloud session | "Master's thesis topic exploration", `session_01EcQFdq7KLqJZcGc3DkVRKG` | **cannot receive** messages (trusted-device only); it reads the hub transcript | |

Messaging: reach the machine sessions with `SendMessage` by name or with the remote `send_message` by session id. They reply
asynchronously. Sending to clod-b2 was once blocked by the auto-mode permission classifier until the user explicitly allowed it.

Rules the user set for machine sessions: read-only unless asked; don't start/kill other people's jobs; no sudo / no
docker-group changes / no auth changes; commit only code + small results (no weights, datasets, bitstreams, secrets).

## 3. Repo: github.com/BhardwajBhat/thesis-exploration (private)

| Branch | Machine | Contents |
|---|---|---|
| `main` | hub | first commit: handoff notes + analytical model |
| `joshua-tree` | hub (current) | `pdmodel/` analytical P/D model + `kill_b.py`; `bench/eta_bench.py` + 4090 results; `hybrid/` direction-#1 experiments; `HANDOFF.md` log; this file |
| `spark` | Spark | GB10 eta results; `kv_killtest/` (idea C: scripts, vLLM/BFCL/tau2 patches, results, README) |
| `reactor` | reactor | scaffold + .gitignore only |
| `yellowstone` | yellowstone | .gitignore only |

Git identity: repo-local only (BhardwajBhat / bhardwaj.bhat@gmail.com; yellowstone uses "Bhardwaj Bhat").

## 4. User's existing assets (verified details)

- **KV260 Qwen3.5-0.8B decoder** (`~/qwen-quant` on reactor), hybrid model: 18 Gated DeltaNet layers + 6 full-attention layers.
  - Nested-precision weights: one nested-GPTQ 8-bit code (BF16 scales per group of 128, plane-major tiles), mode `k` reads the
    top k bit-planes: `q = c >> (8-k)`, decoded as `(q + 0.5)*2^m*Δ`. Activations INT8. Notation "8/6" = **body weight
    bits / output-head weight bits**; head mode is 6 when body is 6 or 8, otherwise equal to body.
  - lm_head tied with embedding, nested 8-bit store (258 MB), argmax on the PL over the full 248,320 vocab.
  - Measured (INA260, 5.03 V rail, idle 4.0 W), run 21: 8/6, 6/6, 4/4, 3/3 = 58.0 / 49.9 / 38.6 / 35.1 ms/token,
    0.388 / 0.331 / 0.247 / 0.216 J/token. Current all-PL path (run 20c): 54.0 / 45.9 / 34.6 / 31.1 ms/token.
  - Weight-path time (synthetic, run 3): body 33.1 / 25.0 / 17.1 / 14.8 ms, head 16.9 / 12.7 / 8.8 / 7.9 ms at k = 8/6/4/3.
    Non-GEMV fixed cost ~6-12 ms/token (DeltaNet recurrence ~2.8 ms; docs disagree).
  - State: DeltaNet recurrent state stored **FP16** on the board (FP32 math), 9.44 MB read + 9.44 MB write per token; conv
    state ~0.66 MB each way; KV ~12 KB per past position.
  - User already explored: screened nested head (`baseline/docs/SCREENED-HEAD.md`), a 32k-token draft head for speculation
    (`SPECULATIVE.md`), FP8 vs FP16 state (`LONG-STATE.md`, FP8 drifts at 16-32k), state quantization (`STATE-QUANT-CHECK.md`:
    int8_row +5.9% PPL, int4_row +450% PPL, top-1 0.378).
  - Agent-precision results (from the original handoff): prefilling long tool outputs (HotpotQA) at 4-bit weights is free when
    generating at 6-8 bit; on BFCL, reading the whole prompt at 4-bit hurts the 2B model (AST 0.497 vs 0.695 static 8/6;
    read4/gen8 0.537); 8-bit only inside tool-call spans does nothing. **Correction:** "DeltaNet state written at 4-bit is read
    fine at 6/8" meant 4-bit WEIGHTS while the FP32 state is written, not a 4-bit state tensor. The scripts for these agent
    results are not on reactor; probably on yellowstone `~/clod`.
- **ASIC design**: two-tier dequant/encode unit; tier 1 multi-format dequant (GPTQ/AWQ, MX/NVFP4, NF4, GGUF K-quants; not novel
  alone), tier 2 programmable ISA with KV encoders (FP8, NVFP4, MX, KIVI, TurboQuant), bit-exact with vLLM.
- **STRIDE** (paper written): lossless look-ahead agent-turn execution on A100s/vLLM; Intent Gate / Readiness Gate probes,
  KV-aware fork depth; up to 1.69x single agent, 1.56x at capacity; failed on SWE-bench at high load (0.37x, 1,154 vs 66
  preemptions). User replicated SPORK (2607.03333, self-speculative tool-call forking) in batch-invariant mode: 30/30 identical
  tasks at ~1.014x.
- **VCK5000** experience via the lab (HeteroVLM).

## 5. Ideas and their status

### Dead or taken (don't revisit without new evidence)

| Idea | Why dead | Evidence |
|---|---|---|
| **B** spatial prefill/decode partitions on one array (DynaCore-style, concurrent) | Throughput: mixed batches lose little on GPUs (4090 layer eta 0.88-0.98 at B<=4; loss only in attention and software-fixable; GB10 flex mixing even faster than separate at B<=4). Energy: KV260 static power is 70-81% of window energy; task-level saving 0.2-11%; and the precision premise (W4 decode / W8 prefill) is inverted by the user's own data | `bench/`, `pdmodel/kill_b.py` |
| **C** role-aware KV precision for agents | Core taken by TriAxialKV (2605.17170, NeurIPS'26). Spark kill test: uniform NVFP4 KV costs Qwen3-8B ~3-3.5 pt on BFCL multi-turn, but FP8 loses the same on long context, no schema-copy failures (loss is generic, not role-specific); batch-1 decode NVFP4 vs FP8 only 1.01x/1.04x/1.15x at 8k/32k/130k | `origin/spark:kv_killtest/README.md` |
| **D** cross-model state reuse for hybrid models | Core taken: LatentPort (2609.25053), HST (2611.08133), 2611.12062; transformers: KV-Lingo (2609.32610), CacheBridge, CacheLink. Same-model prefix reuse already 95-98% in Spark agent reports | |
| **G** quantized self-draft + exact verify for on-policy RL rollouts | Done by EfficientRollout (2606.18967, NeurIPS'26) | |
| **H** diffusion LLMs on edge | Partially done, fast-moving (DART 2601.20706, DynaTE 2610.11284, 2609.01084); open dLLMs weak at tool calling (2601.12979) | |
| **L** silent data corruption in LLMs | Mostly done (AEGIS OSDI'26, LLM-PRISM, TrainSDC, LLMFI ICS'26, CSNB 2610.18557); only gap: SDC in agent loops | |
| Tool-wait KV management | InferCept, TokenCake, Continuum, MORI, vLLM RFC #57103 | |
| Nested precision on GPU; bitplane eviction | Any-Precision, MatGPTQ, NestedFP, AutoEMatic; DPS, PagedWeight | |
| KV transform engine on the offload path | BF16 already crossed PCIe; fused GPU kernel can do it | |
| Phase-different quantization alone | Mix-Quant (2605.20315), Disaggregated Quantization (2609.26333), DynaCore | |
| Grace CPU as long-context attention engine; huge-MoE NVMe streaming on Spark | DirectKV (OSDI'26), "Who should own the expert cache" (2608.12103), "Ingestion Tax" (2608.12114), SGLang SSD Expert Pack; on GB10 CPU and GPU split the same ~273 GB/s so CPU offload adds nothing | |
| Hybrid prefix caching on servers; EP over PCIe; RL weight sync / energy / MoE balance / drafters | Marconi, SGLang URC, Tail-Replay (2608.30310); ThunderEP; PULSE, EE-RL, ReLibra, GrowMTP | |
| **#1** hybrid-model recurrent-state hardware on KV260 (nested self-speculation + low-bit state snapshots) | Experiment 1 (this repo): acceptance is high, but best KV260 speedup 1.13x (full head) / 1.20x (32k trimmed head) because ~10-12 ms/token is fixed non-GEMV cost and the head is already nested; snapshots are cheap anyway (4.7 MB ~0.4 ms vs 58 ms/token) and 4-bit state hurts (top-1 0.76-0.90) | `hybrid/README.md` |

### Still alive

| Idea | Status | Closest work | Demand | First experiment |
|---|---|---|---|---|
| **#2 Unified-memory capacity governor for agent boxes (Spark)** — **recommended next** | Open | eLLM (2506.15155, elastic KV, treats CPU memory separately), LOCAL (2608.15241, discrete GPU), Continuum (2511.02230); Ripple (2611.01067) is bandwidth/robotics only | NVIDIA forum: Spark freezes instead of CUDA OOM; Docker/systemd MemoryMax bypassed by UMA CUDA allocations; vLLM grows 30->117 GB; page cache competes; ~10 min to swap two vLLM models | On the Spark: run vLLM + a real agent workload (tool processes, builds, page cache), log where the 128 GB goes over time, reproduce the freeze; then a PSI-driven policy that budgets KV + co-resident models + tool RSS + page cache together (shrink/compress idle-agent KV) |
| **A** gated, bit-exact, memory-aware speculation for agentic RL rollouts | Open but crowded | ObsSpec (2610.15246), BunshinRL (2611.10453), Heddle (2603.28101), LLM-42 (2601.17768); async RL (AReaL) hides tool latency except on the straggler tail | RL rollout cost; train-inference mismatch | yellowstone: per-step timelines of a multi-turn tool RL task; compute oracle-speculator speedup on the slowest trajectory; dead if < ~1.15x |
| **M** VLA robot policies on Versal (AIE + PL) under control deadlines | Partial, crowded but specific gap | EdgeDAE (2610.00311), Deltoris (2608.04428), Cornell SmolVLA on AIE (FCCM'26, ~5 s), EagleVLA, RTC, VLASH | Figure Helix, Gemini Robotics On-Device, 10-100 Hz | Profile SmolVLA/pi0 per stage on VCK5000 (HeteroVLM kernels) vs Orin; dead if > ~100 ms/chunk. Needs: advisor buy-in, VCK5000 availability |
| **I** determinism / batch-invariance as a hardware property | Hardware side open; motivation weakened by Detent (2611.10611, SW cost 1.03-1.17x) | CertiSum (2611.21584), TBIK (ICML'26), 2609.25624 | RL on-policy, cross-vendor reproducibility | Model/synthesize a fixed-order wide accumulator PE vs Detent's overhead |
| **N** full-duplex voice agents on edge under 80 ms frames | GPU serving taken (Metronome 2607.02640, VoxServe 2602.00269, vLLM-Omni); edge open | | vLLM-Omni RFC #3745 | Time Moshi's per-frame step on Spark (bf16 vs int8) |
| GPU: KV reuse when chat templates rewrite history | Open (2 searches found nothing) | CacheBlend, EPIC (handle moved chunks, not deleted spans) | Qwen3 #1826, vLLM forum | Measure prefix-hit loss on agent traces; selective recompute + RoPE re-rotation in vLLM (2 A100s) |
| GPU: bit-faithful Blackwell FP4/FP8 MMA emulation on Ampere/Ada + RL mismatch study | Open | MMA-Sim (2511.10909, MICRO'26, models only) | LMSYS NVFP4-RL "quantization contract" | Triton kernel reproducing GB10 FP4 dot products bit-for-bit, validated on the Spark |
| Edge-engine hybrid-state checkpointing (llama.cpp/MLX) | Demand strong, mostly engineering | Marconi, SegLen, SGLang URC on servers | llama.cpp #24055, PR #22929, mlx-lm #980 | — |
| Node-level interference on agentic-RL hosts (cache/MB partitioning) | Partial | CoTail (2609.05425), DeepSeek DSec | Blink (2604.07609) | Needs Intel RDT/AMD QoS on a server CPU |

Watch list (very fresh, re-check monthly): Ripple's authors (Zhang & Ma), CTRL (2611.27119), Prasanna group (GDN on FPGA),
SinkState (2611.14246), Mutable Inference (2611.18523).

## 6. Experiments done (all reproducible from the repo)

1. **Analytical P/D model** (`pdmodel/*.py`, stdlib only, `python3 <file>` from `pdmodel/`): spatial vs fused vs time-multiplexed
   prefill/decode; KV260 energy calibration 0.386 vs 0.388 J/token measured. Findings in HANDOFF.md.
2. **eta microbenchmark** (`bench/eta_bench.py`): one Qwen3-8B layer, mixed vs specialised prefill/decode steps, FA varlen / flex /
   SDPA, fixed vs extra token budget. Results: `bench/results/eta_NVIDIA_GeForce_RTX_4090.csv`, `origin/spark:bench/results/eta_NVIDIA_GB10.csv`.
   GEMMs mix for free; attention loses 20-45% at B>=16 with FA varlen; splitting attention recovers most.
   Bug fixed by the Spark session: flex recompiled per shape and silently fell back to eager after 8 shapes (dynamo limit raised).
   Old invalid flex rows are in `bench/results/invalid_flex_recompile/`.
3. **B kill test** (`pdmodel/kill_b.py`): KV260 static share and task-level dilution.
4. **C kill test** (Spark, `origin/spark:kv_killtest/`): BFCL v4 multi-turn (n=200/category) + tau2 retail/airline (8B) + batch-1 decode speed.
   Found a vLLM 0.34 bug on sm_121: FP8/NVFP4 KV + async scheduling -> CUDA illegal memory access under chat/tool workloads;
   `--no-async-scheduling` avoids it. **Not reported upstream** (user hasn't decided).
5. **#1 experiment** (`hybrid/`): nested self-speculation acceptance, trimmed draft head, state snapshot precision and format.

## 7. Practical gotchas

- On the hub, `uv pip install -U` of anything that depends on torch pulls a CUDA 13 torch the driver can't run: pin
  `torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124` and `triton==3.2.0` afterwards.
- Qwen3.5 in transformers 5.20 falls back to slow PyTorch kernels for Gated DeltaNet (fla / causal_conv1d not installed); fine for 0.8B.
  Recurrent state lives in `cache.layers[i].recurrent_states` (a dict), FP32, shape [1,16,128,128].
- Spark: no sudo, no docker; use mise Python for anything needing headers; quantized KV in vLLM 0.34 needs `--no-async-scheduling`.
- Use `CUDA_VISIBLE_DEVICES=0` on the hub. Research subagents must open every URL they cite (an external reviewer once fabricated a citation).

## 8. Recommended next steps

1. **Start #2 (Spark memory governor)** with the week-1 measurement in section 5. If interference/capacity effects are small or
   NVIDIA's driver already fixes accounting, fall back.
2. In parallel, cheap checks to keep options open: A's oracle-speculator bound on yellowstone; N's Moshi frame-time on the Spark.
3. Ask the advisor: (a) is any labmate on hybrid/DeltaNet state or VLA-on-Versal (overlap with Chimera/HeteroVLM)? (b) VCK5000 availability for M.
4. Before committing to any topic, re-run a verified novelty search (arXiv moves weekly in this area).

## 9. Deep-search follow-up and software kill tests (2026-11-25)

A new broad search proposed grammar-aware FPGA output heads, visual-delta GUI front ends, static-NPU agent re-arrival support, and hardware action monitors. Software tests are in `structured_head/` and `idea_killtests/`.

- **Grammar-aware sparse output head: killed.** Exact XGrammar masks over 15 BFCL/tau2-shaped schemas (500 token boundaries) require 45.45% of rows on average at 64-row granularity; the optimistic KV260 projection is only 1.135x, below the 1.15x threshold, before sparse-DDR overhead. At 50% structured tokens, task speedup is 1.063x.
- **Fused grammar filter + existing full head: killed as a performance thesis.** XGrammar mask generation averages 12.2 us versus 58 ms/token; eliminating it entirely gives 1.00021x.
- **Visual-delta GUI front end: killed.** On 176 transitions from 20 real GUI-CC trajectories, 64.9% of patches change on average, median 84.4%, p90 100%. Headroom is irregular and GUI-KV already covers spatiotemporal reuse.
- **Static-NPU agent re-arrival: killed for novelty.** Escapement already validates bucket alignment, KV survival cliffs, prefill interference, and compile-time configuration selection on Rebellions CA25; reported tuning gain is 9.72%.
- **Grammar-only hardware action monitor: killed.** In 10,000 mutated robot actions, JSON Schema/type/range checks detect 50.98% of unsafe cases but miss environment-dependent hazards; contextual policy is necessary.
- **Spark unified-memory governor remains unresolved but is unavailable in practice.** `idea_killtests/uma_probe.py` is ready, but the user no longer has Spark access, so this cannot be the thesis dependency.

## 10. Hardware-available shortlist kill tests (2026-11-25)

After constraining the search to RTX 4090/A100 experiments, two leading candidates were tested.

- **Batch-invariant/deterministic arithmetic: experimentally real, but killed as a thesis gap after background review.** On Qwen3.5-0.8B BF16 on an RTX 4090, singleton versus batch-4 execution diverged on 3/5 greedy trajectories; FP32 had 0/5 divergences. However, vLLM and SGLang already ship batch-invariant modes; LLM-42 provides selective verify/rollback determinism; Detent reports batch-invariant vLLM/SGLang at 1.04-1.17x stock and exact rollout/trainer log-probs at 1.03x rollout cost; CertiSum covers selective exact MXFP4 accumulation across vendors/layouts. Hardware would have to beat roughly 3% overhead or offer a materially stronger guarantee. See `determinism/BACKGROUND.md`.
- **Gated exact speculation for agentic RL: killed for the public code-RL workload.** RollPacker's 2,001 KodCode evaluator timings have median 30.8 ms, p90 78.4 ms, and a heavy p99 tail of 9.14 s. Even perfect speculation with 20 ms overhead gives only 1.051x/1.042x/1.031x aggregate speedup for a 0.5/1/2 s next LLM step; 90% gate accuracy is lower. Combined with prior math/SPORK results, do not pursue without real target traces showing much longer straggler tool gaps. See `rl_spec/`.

Current recommendation: **none from this shortlist.** Deterministic arithmetic and RL speculation are both removed as front-runners.

## 11. Looped-language-model review and kill test (2026-11-25)

The obvious architecture directions are already crowded: Continuous Depth Batching handles adaptive-depth GPU scheduling; ExactLoop handles exact recurrent-kernel fusion; CHASE handles skipped recurrent-state/cache holes; and MixerLoop implements a complete W8A8 Gated DeltaNet decoder on a **KV260**, retaining mixer weights/state on chip and hiding four mixer passes behind FFN transfer. MixerLoop reports ~152 tok/s for both NoLoop and four-step MixerLoop on the same datapath, so fixed-loop KV260 acceleration directly overlaps the user's assets.

A software kill test in `looped/` evaluated Ouro-1.4B's four recurrent outputs over 3,072 WikiText-2 next-token positions on the RTX 4090. Latency scales almost exactly with loop count (1.00/1.99/2.97/3.96x). There is substantial oracle variability: 49.8/22.1/16.8/11.3% of tokens first agree with depth 4 at depths 1/2/3/4, for a 1.91-step oracle average. But the released learned gate chooses depth 4 for 90.4% of tokens and matches that oracle only 12.2%. A held-out calibrated logit-margin policy preserves 99.0% depth-4 top-1 agreement while using 3.236 steps, only ~1.24x ideal recurrent-compute savings before scheduling and gate overhead; it also requires an impractical intermediate vocabulary projection.

**Verdict:** kill generic looped-model scheduling/fusion/fixed-loop FPGA acceleration. The only conditional survivor is a tiny hidden-state exit gate plus hardware-slack-aware execution. Continue only if a cheap gate reaches <=2.6 mean steps at >=99% depth-4 agreement (~1.5x ideal recurrent-compute reduction) on reasoning tasks. FPGA access cannot be required for the initial test.
