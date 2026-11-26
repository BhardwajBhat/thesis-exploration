**Bottom line:** nothing strong turned up in my area. The real pain is in AMD's own NPU stack, and most of it is engineering.

**1. Multi-LoRA on a low-bit FPGA decoder (keep the adapter separate, never merged)**
1. Keep one shared base weight set in your nested 8/6/4/3-bit format on the KV260, and run small 16-bit LoRA adapters beside it. The adapter is chosen per request and lands in the GEMV output stage, so the low-bit base never has to absorb the adapter's small change.
2. Demand: FastFlowLM #976 (https://github.com/ROCm/FastFlowLM/issues/976). There is no adapter input on the AMD NPU. Each merged adapter becomes its own 1.34 GB model, switching takes 8–16 s, and "requantizing … adds rounding noise about twice the size of a small adapter's own change".
3. Prior work: EpiLoRA (2611.03899) fuses LoRA into FP8/4-bit GEMMs, but only on Hopper GPUs. EdgeLoRA (2507.01438) works at the llama.cpp/CPU system level. Samsung (2604.18655) passes LoRA in as runtime inputs on Qualcomm phones. None of them covers a nested-precision base, the linear-attention projections, or FPGA/AIE.
4. Fit: it uses your nested-precision KV260 decoder and Qwen3.5 Gated DeltaNet model directly. The A100s can run the accuracy study.
5. Kill test (A100, one day): train a rank-32 LoRA on one task. Then compare (a) merge into BF16, then nested-quantize, read at 8/6/4 bits, against (b) the unmerged adapter over the quantized base. Dead if (a) keeps ≥95% of the adapter's task gain at 4 and 6 bits. Also dead if a back-of-envelope KV260 estimate puts the unmerged branch above 10% of the 58 ms/token.
6. Verdict: **medium-weak**. Biggest risk: a search snippet says AMD shipped "Dynamic LoRA on Ryzen AI" for Qwen3.5-4B on 2026-11-02. The page timed out twice, so this is unverified, but if true the vendor already covers the demand.

**2. Bit-reproducible decode on dataflow NPUs/FPGAs (new evidence for your live idea I, not a new idea)**
1. Claim that a fixed reduction order in the PL/AIE datapath gives deterministic greedy decode for free, unlike the closed NPU engines.
2. Demand: FastFlowLM #810 (https://github.com/ROCm/FastFlowLM/issues/810). Identical top_k=1 requests give different outputs, and 5/10 long-prompt runs collapse into the same degenerate output. AMD could not reproduce it on different models and builds. A comment on #767 (https://github.com/ROCm/FastFlowLM/issues/767) also complains about the NPU math and links #810.
3. Prior work: unchanged from your table (Detent, CertiSum, TBIK). I found nothing NPU-specific.
4. Fit: your KV260 decoder is probably deterministic already, which makes it a ready baseline.
5. Kill test: run 1,000 identical greedy requests through FastFlowLM on any Ryzen AI laptop (if you can borrow one), at 5k-token prompts, with prefix cache on and off. Dead if there are 0 divergences on v1.0.12+.
6. Verdict: **weak**. Only one issue, the vendor can't reproduce it, and the cause is likely a buffer-reuse bug, not hardware.

**3. Laying out NPU DMA traffic to minimise IOMMU page translations (prefill)**
1. Lay out prefill GEMM tiles so the AIE shim DMAs touch fewer 4 KB pages (lower IOMMU translation pressure).
2. Demand: xdna-driver #1817 (https://github.com/amd/xdna-driver/issues/1817) and FastFlowLM #1080 (https://github.com/ROCm/FastFlowLM/issues/1080).
   - Windows: 6.0 vs 73.2 GB/s for scattered 128 B rows, against Linux with 2 MB pages.
   - AMD confirms translation cost on both OSes (their Linux buffers are 4 KB too).
   - 2 MB pages give only 7–10% better TTFT, and AMD says they will tile the layouts themselves.
3. Prior work: TileFuse (2606.11357) does XDNA2 layout co-design but doesn't model translation cost.
4. Fit: poor. You have no Ryzen AI box, and the KV260 isn't the same IOMMU setup.
5. Kill test: run `artlite-probe --pages` plus one Llama-1B prefill on Linux with 4 KB vs 2 MB pages. Dead if the end-to-end difference is under 15%; AMD's own numbers (7–10%) already point that way.
6. Verdict: **weak**. Small gain, and the vendor is already fixing it.

**Ideas checked and rejected:**
- **MXFP4 on FPGA:** taken by Cruella (2611.27323, U55C, 20 Nov 2026).
- **Gated DeltaNet decode on FPGA:** done in 2603.05931, and your own experiment #1 killed it.
- **NPU-draft / iGPU-verify speculative decoding on a unified-memory APU:**
  - It measured slower than GPU-only (2640 vs 1313 ms) in hvico's repo (https://github.com/hvico/strix-halo-speculative-decoding-tests).
  - The halo project (https://muchdevsuchcode.github.io/halo/) gets 1.52× with the iGPU alone, while its open NPU backend is ~1000× too slow.
  - Mirror-SD (2510.13161) already covers GPU+NPU speculation, and DFVG (ASPLOS'26, draft on FPGA, verify on GPU) also exists; I couldn't open DFVG (HTTP 403), so I can't cite it.
- **Long context on the NPU** (FastFlowLM #767, an 8k cap from a 10-bit DMA-descriptor limit): one user already lifts it past 100k with a binary patch, so it's engineering.

If you want something in this area anyway, take #1 and run the merge-then-quantize test first: it's cheap and decides the idea quickly.
