**Verdict: nothing strong survives in numerics/formats.** On GPUs, the software side is moving fast and is mostly covered. The hardware versions I found are either already taken or don't matter on the student's hardware. Two weak-to-medium leftovers, best first. I opened every link below.

**1. Exact decoding of MXFP4/NVFP4 checkpoints on hardware without native support (FPGA or older GPU)**
- **Idea:** design an MX×BF16 datapath for the KV260 or VCK5000 with the narrowest exponent-alignment window that still decodes real public checkpoints exactly, and that rejects any block it can't decode exactly instead of silently rounding it.
- **Demand:**
  - vLLM #66074 (https://github.com/vllm-project/vllm/issues/66074): on Hopper, vLLM's default MXFP4 kernel (Humming) silently rounds or zeroes blocks whose scale is 12 or more powers of two below the tensor's largest. 13 of 34 public checkpoints are hit (about 297M weights).
  - tilelang #2748 (https://github.com/tile-ai/tilelang/issues/2748): kernels disagree on how to read the E8M0 scale byte (2^s vs 2^(s−127)).
  - TensorRT-LLM #12558 (https://github.com/NVIDIA/TensorRT-LLM/issues/12558): MXFP4 with 8-bit activations is unsupported or crashes on Hopper.
  - theservingstack.com/humming-quantized-kernels: an NVFP4 checkpoint with bad stored activation scales served silently and scored 60% vs 95% on GSM8K.
- **Prior work:**
  - Cruella (2611.27323, posted 6 days ago): MXFP4 on FPGA DSPs with per-group pre-alignment, perplexity changes up to 0.7%. Not exact, no checkpoint survey.
  - MFSA (2610.19645): ASIC, folds block scales into the FP32 partial sum.
  - Jack of All Scales (2607.13898): proposed DSP changes on Agilex-5.
  - Conformance-testing work (2609.00363) covers INT8 only.
- **Fit:** extends the student's nested-precision KV260 datapath; A100s for checking logits.
- **Kill test (1 day, CPU plus 1 A100):** read the E8M0 scales of about 30 public MXFP4 checkpoints, histogram how far each block sits below its row or group maximum, and find the window that covers 99.99% of blocks. Then use a fake-quant HF forward pass on an A100 to apply a narrower window W and measure KL against an exact decode. **Dead if** W ≤ 8 already gives KL < 1e-3 on every checkpoint, because then a cheap narrow datapath is good enough and there is nothing to design.
- **Verdict:** weak-medium. Biggest risk: the #66074 author already showed the window error mostly doesn't reach the output (Granite perplexity unchanged, AMD Llama +0.07%). So this is probably a bug-fix plus a survey, not a thesis. Cruella has also just taken "MXFP4 on FPGA".

**2. A hardware rounding unit for low-bit recurrent state in hybrid models (GDN/Mamba) on FPGA**
- **Idea:** build the state-update unit with deterministic dither or stochastic rounding on each state store, so 8-bit or bf16 state doesn't make the model generate longer, looping outputs.
- **Demand:**
  - vLLM #68338 (https://github.com/vllm-project/vllm/issues/68338): bf16 state makes decode drift away from prefill by 13–16×. It also notes there is no stochastic rounding on vLLM's GDN path.
  - llama.cpp PR #32595 (https://github.com/ggml-org/llama.cpp/pull/32595): configurable rounding for an FP16 Mamba-2 state cache.
  - The NVIDIA Nemotron-3-Super FP8 model card on Hugging Face ships with stochastic rounding of the state turned on.
- **Prior work:** all software, and largely complete:
  - Lost in Rounding (2611.12127): with bf16 state, Qwen3.5-9B drops from 85.6% to 70.8%. Stochastic rounding fixes it at bf16 speed.
  - Weyl dither (2609.39185): a deterministic dither beats stochastic rounding at no extra cost.
  - ReplaySSM (2610.13778): keeps two state copies at different precisions.
  - Low-bit GDN state (2610.39931).
- **Fit:** the student's own GDN decoder on the KV260.
- **Kill test:** the student's Experiment 1 data mostly kills it already (state I/O is about 0.4 ms of a 58 ms token). Confirm with an HF run on an A100: Qwen3.5-0.8B with int8 state plus dither vs fp32 state, on 100 math problems. **Dead if** state traffic is under 5% of KV260 token time, or if software dither already matches fp32.
- **Verdict:** weak. Software already gets the fix for almost nothing, so hardware adds nothing on the KV260.

**Closest misses, and why they're dead:**
- **RL train/inference mismatch:** taken in software. vLLM's 2026-10-14 blog post gets bitwise-identical logprobs at 91.3% of rollout throughput (https://vllm.ai/blog/2026-10-14-train-inference-consistency). IsoExec does it with 25% overhead, including GDN (https://vllm-project.github.io/2026/08/21/isoexec.html).
- **Inference-backend reproducibility:** measured by 2605.19537 (up to 16.6 points of benchmark difference from the backend alone).
- **Checking which quantization a provider actually serves:** precision extraction from logprobs (y0mingzhang.github.io/blog/posts/precision-extraction/) and OpenRouter's Exacto routing for tool calls. This is software, not hardware.
- **Accumulator width:** DeepSeek-V3 already raised the FP8 accumulation-precision issue on Hopper, and MFSA and TransDot already cover the hardware side.

My recommendation is to stay with #2 (the Spark memory governor) rather than open a numerics thesis.
