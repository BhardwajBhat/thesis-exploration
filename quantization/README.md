# Qwen3.5-0.8B quantization perplexity

Quick comparison using the same first 4,096 next-token predictions from the
PyTorch examples WikiText-2 test text. Eight independent 512-token windows;
logits are converted to FP32 for cross entropy.

## Consolidated results

All PPL values use the identical 4,096-prediction protocol above. `Sim` means
quantize-dequantize numerical simulation followed by BF16 matmul; it measures
quality, not packed-kernel speed. Coverage is the fraction of linear weights in
the common harness; checkpoint rows use their authors' layer selection.

| Format | W/A | Method | GDN projections | Coverage | PPL | Δ vs BF16 | Source |
|---|---|---|---|---:|---:|---:|---|
| **BF16** | 16/16 | — | BF16 | 0% | **21.168** | — | HF |
| FP8 W8A8 | 8/8 | dynamic RTN | **BF16** | 41.0% | **21.182** | **+0.07%** | Sim |
| AWQ INT8 | 8/16 | calibrated AWQ | quantized | checkpoint | 21.208 | +0.19% | HF checkpoint |
| FP8 W8A8, static A | 8/8 | per-tensor A scale, 16k-token calib | quantized | 66.2% | 21.261 | +0.44% | Sim |
| INT8 W8A16 | 8/16 | per-row RTN | quantized | 66.2% | 21.278 | +0.52% | Sim |
| FP8 dynamic checkpoint | 8/8 | dynamic | checkpoint-defined | checkpoint | 21.279 | +0.52% | HF checkpoint |
| FP8 W8A8 | 8/8 | dynamic RTN | quantized | 66.2% | 21.291 | +0.58% | Sim |
| Activation-only INT8 | 16/8 | per-token RTN | quantized | 66.2% | 21.388 | +1.04% | Sim |
| INT8 W8A8 | 8/8 | dynamic RTN | quantized | 66.2% | 21.477 | +1.46% | Sim |
| SmoothQuant INT8 W8A8 | 8/8 | α=0.5 fold + dynamic RTN | quantized | 66.2% | 21.511 | +1.62% | Sim |
| MXFP8 W8A8 | 8/8 | E4M3 + E8M0 block 32 | quantized | 66.2% | 21.645 | +2.26% | Sim |
| **GPTQ INT4** | 4/16 | AutoRound/GPTQ g128 | **BF16** | checkpoint | **21.612** | **+2.10%** | HF checkpoint |
| MR-GPTQ NVFP4 | 4/4 | GPTQ, **no rotation**, 128x2048 calib | quantized | 66.2% | 22.649 | +7.00% | Sim |
| NF4 g64 | 4/16 | RTN | **BF16** | 41.0% | 22.609 | +6.81% | Sim |
| AWQ INT4 | 4/16 | calibrated AWQ g32 | quantized | checkpoint | 22.782 | +7.62% | HF checkpoint |
| NF4 g64 | 4/16 | RTN | quantized | 66.2% | 23.441 | +10.74% | Sim |
| MR-GPTQ NVFP4 | 4/4 | Hadamard-16 + GPTQ, 128x2048 | quantized | 66.2% | 23.358 | +10.35% | Sim |
| NVFP4 W4A4 | 4/4 | RTN | **BF16** | 41.0% | 24.081 | +13.76% | Sim |
| Mixed NVFP4/FP8 | 4/4, 8/8 GDN | RTN | FP8 | 66.2% | 24.308 | +14.84% | Sim |
| M²XFP | 4.5/4.25 eff. | official MXES/MXEM | **BF16** | 41.0% | 24.641 | +16.41% | Sim |
| NVFP4 W4A4 | 4/4 | RTN | quantized | 66.2% | 24.655 | +16.48% | Sim |
| **MR-GPTQ MXFP4** | 4/4 | Hadamard-128 + GPTQ, 128x2048 | quantized | 66.2% | **25.004** | **+18.12%** | Sim |
| MXFP4 W4A4 | 4/4 | RTN | **BF16** | 41.0% | 26.412 | +24.77% | Sim |
| M²XFP | 4.5/4.25 eff. | official MXES/MXEM | quantized | 66.2% | 27.060 | +27.84% | Sim |
| MXFP4 W4A4 | 4/4 | RTN | quantized | 66.2% | 29.259 | +38.22% | Sim |
| INT4 symmetric | 4/16 | per-row RTN | quantized | checkpoint harness | 42.303 | +99.85% | Sim |

**Bottom line:** FP8 W8A8 is effectively lossless only when GDN projections
remain BF16. For weight-only 4-bit, calibrated GPTQ is the clear winner; the
best uncalibrated codebook is the refitted BOF4 LUT at +3.62%. None of the
tested W4A4 formats is close to lossless: MR-GPTQ improves MXFP4 strongly,
but its PPL is still 18.1% above BF16. On NVFP4, Hadamard rotation hurts
(23.358 versus 22.649 without rotation), consistent across smoke and full runs.

The RTN variants quantize every `torch.nn.Linear` per output channel and
immediately dequantize back to BF16, so they measure numerical quality rather
than packed-kernel speed or memory use. AWQ/FP8 use axteam compressed-tensors
checkpoints. GPTQ uses `JW-Hubs/Qwen3.5-0.8B-W4A16-GPTQ`; its GDN and LM head
remain BF16 according to that checkpoint's dynamic configuration.

An `axteam/Qwen3.5-0.8B-NVFP4` reference attempt was invalid on the RTX 4090:
the loader reported missing/unexpected parameters and produced NaN logits.
NVFP4 requires a supported checkpoint/runtime, preferably Blackwell, before its
PPL can be compared. MXFP4 currently surfaced only as an MLX-specific build.

Raw values: `results/qwen35_08b_wikitext2_4096.json`.

## Advanced fake-quant harness (`advanced_ppl.py`)

Same protocol as above (4,096 predictions, 8 x 512 windows). Selected
`nn.Linear`s are replaced by `QLinear`, which fake-quantizes the weight once
and the **input activation on every call**. Modes: `bf16`, `a8_int8`,
`w8a8_int8`, `w8a8_fp8`, `w4a4_mxfp4`, `w4a4_nvfp4`, `w4_nf4_g64`,
`mixed_nvfp4_fp8gdn` (NVFP4 everywhere except GDN `linear_attn.*` = FP8), `w4a4_m2xfp` (M²XFP: MXES weights / MXEM activations, group 32, subgroup 8, 4.5 bit).
Activation-quant modes (`a8_fp8`, `a8_fp8_static`, `w8a8_fp8_static`, `a8_mxfp8`, `w8a8_mxfp8`,
`a4_int4`, `w8a8_int8_sq05`, `sq05_bf16`) are described below (`--modes act`).
Weight-only LUT/codebook modes (`w3_nf3_g64`, `w4_fp4_g32/g64`, `w4_af4_g64`, `w4_bof4_g64`,
`w4_bof4s_g64`, `w4_apot4_g64`, `w4_lut4_*`) are described below (`--modes lut`).
Options: `--exclude-gdn`, `--include-lm-head`, `--compute-dtype {bf16,fp32}`,
`--self-test` (unit tests for the INT8/INT4/E4M3/E2M1/E8M0/MXFP8/NF4/NF3/AF4/BOF4/APoT grids, the
nearest-code rules, Lloyd-Max, static FP8 calibration and the SmoothQuant fold),
`--calib-samples/--calib-seqlen` (calibration set for static scales and SmoothQuant).

    .venv/bin/python quantization/advanced_ppl.py --self-test
    .venv/bin/python quantization/advanced_ppl.py --modes bf16 w4a4_nvfp4 --output /tmp/adv.json

Default coverage: 186 linears = 497.6M weights (66.2% of linear weights incl. the
tied 254M lm_head, which stays BF16 by default). `--exclude-gdn` drops the 90
GDN projections (189.3M), leaving 308.3M (41.0%).

Caveats: numerical simulation only (no packed kernels); NVFP4 activation tensor
scale is dynamic per call (optimistic versus calibrated static scales); MXFP4
uses the OCP floor-based shared exponent; FP8/INT8 activation scales are dynamic
per token; RTN only (no GPTQ/AWQ/rotation); GDN recurrent state, conv1d,
attention softmax and KV cache are not quantized.

### M²XFP (`w4a4_m2xfp`)

Port of the official pseudo-quant functions `get_quant_mxes` (weights) and
`get_quant_mxem` (activations) from `mxq/quantize/quant_func.py` of
[SJTU-ReArch-Group/M2XFP_ASPLOS26](https://github.com/SJTU-ReArch-Group/M2XFP_ASPLOS26)
@ `2ed16eb6`, the W4 `mxes` / A4 `mxem` / group 32 configuration from its `llama3_run.sh`
(paper arXiv:2601.19213: Sg-EM-2bit weights, Elem-EM-top1 activations). A verbatim copy of the
official file (MIT, `third_party/`) is used only as a test oracle: on CUDA the port is
bit-identical (`torch.equal`) on random, wide-range, heavy-tailed, zero, tie-heavy and
bf16/fp16 inputs.

Differences from the official repo: grids are device-agnostic; MXEM picks the lowest index on
top-1 ties explicitly (the paper's Alg. 1; official `torch.topk` does this on CUDA but not on
CPU); K must be a multiple of 32, which holds for every Qwen3.5-0.8B linear (1024/2048/3584);
the official harness runs FP16 with matmul in FP16 while this one uses BF16, and the
dequantized values are exact in either. Official quantizes only linears inside decoder blocks;
here lm_head is likewise BF16 by default, but GDN linears are quantized unless `--exclude-gdn`.
RTN plus scale search only: no GPTQ or rotation. Full and GDN-excluded results are in
`results/m2xfp_4096.json` and `results/m2xfp_exclude_gdn_4096.json`.

### Weight-only LUT / codebook quantizers (activations BF16)

Run these with `--modes lut` (expands to `bf16`, `w4_nf4_g64` and all modes below).
The 4,096-token results are in `results/lut_codebook_4096.json`.

| Mode | Effective bpw | PPL | Δ vs BF16 | Weight relative MSE |
|---|---:|---:|---:|---:|
| `w4_lut4_bof4fit_g64` | 4.5 | **21.933** | **+3.62%** | 0.00800 |
| `w4_bof4_g64` | 4.5 | 22.129 | +4.54% | 0.00804 |
| `w4_lut4_lloyd_g64` | **8.0** | 22.124 | +4.52% | **0.00495** |
| `w4_iq4nl_g32` | 4.5 | 22.609 | +6.81% | 0.0078 |
| `w4_bof4s_g64` | 4.5 + sign metadata | 22.720 | +7.34% | 0.00774 |
| `w4_apot4_g64` | 4.5 | 22.849 | +7.94% | 0.01063 |
| `w4_fp4_g32` | 5.0 | 22.907 | +8.22% | 0.01022 |
| `w4_fp4_g64` | 4.5 | 23.441 | +10.74% | 0.01131 |
| `w4_nf4_g64` | 4.5 | 23.441 | +10.74% | 0.00872 |
| `w4_lut4_lloyd_row` | 4.21 | 24.096 | +13.83% | 0.01092 |
| `w4_af4_g64` | 4.5 | 24.422 | +15.38% | 0.00938 |
| `w3_nf3_g64` | 3.5 | 36.285 | +71.42% | 0.04184 |
| `w3_bcq3_g128` | 3.5 | 245.798 | +1061.19% | 0.0659 |

The low-MSE per-group Lloyd row is not competitive once its 8 bpw table overhead is counted.
Greedy BCQ is a failed baseline, not a bound on trained/refined LUT-GEMM.

| Mode | Codebook / method | Scale | Nominal bpw | Source |
|---|---|---|---:|---|
| `w3_nf3_g64` | NF3, 8 levels (3 neg, 0, 4 pos) | FP32 absmax / 64 | 3.5 | LoftQ `create_normal_map(num_bits=3)` ([yxli2123/LoftQ](https://github.com/yxli2123/LoftQ) @ae33fd4 `glue/utils_qaunt.py`); with `num_bits=4` it reproduces NF4 bit-exactly |
| `w4_nf4_g64` | NF4 (existing) | FP32 absmax / 64 | 4.5 | QLoRA / bitsandbytes |
| `w4_fp4_g32`, `w4_fp4_g64` | OCP E2M1 grid {0,.5,…,6}, RNE | FP32 absmax/6 per group | 5.0 / 4.5 | OCP MX spec element grid; fixed-codebook baseline, **not** MXFP4/NVFP4 (no E8M0/E4M3 scale rounding) |
| `w4_af4_g64` | AF4-64 (L1-optimal, fixed −1/0/1) | FP32 absmax / 64 | 4.5 | Yoshida, arXiv:2306.06965; values from `ifnspaml/bof4` `codebooks/af4/af4_64` |
| `w4_bof4_g64` | BOF4 (MSE, B=64) | FP32 absmax / 64 | 4.5 | Blumenberg et al., arXiv:2505.06653; [ifnspaml/bof4](https://github.com/ifnspaml/bof4) @a447ef1 `codebooks/bof4/bof4_mse_64` |
| `w4_bof4s_g64` | BOF4-S (MSE, B=64), fixed 0/+1 | signed absmax / 64 | 4.5 (+1 sign bit) | same repo, `bof4-s_mse_64` |
| `w4_apot4_g64` | APoT 4-bit weight: sign × `build_power_value(B=3)` = {0,1,2,3,4,6,8,10}/10 | FP32 absmax / 64 | 4.5 | Li et al. ICLR'20, [yhhhli/APoT_Quantization](https://github.com/yhhhli/APoT_Quantization) @13204f1 `quant_layer.py` |
| `w4_lut4_bof4fit_g64` | one 16-entry LUT per tensor, fitted with BOF4's empirical EM (Lloyd, centroid weighted by block-absmax², Eq. 64), initialised at BOF4 | FP32 absmax / 64 | 4.5 + 16 FP32 per tensor | BOF4 Sec. 3.2 / App. B.3 algorithm, applied to the actual weights |
| `w4_lut4_lloyd_g64` | separate 16-entry MSE Lloyd-Max LUT **per 64-weight group**, BF16 entries | none (values in LUT) | **8.0** | plain Lloyd-Max / k-means; best-case reference for "per-group LUT" |
| `w4_lut4_lloyd_row` | 16-entry Lloyd-Max LUT per output row (SqueezeLLM-style with no Fisher weighting) | none | 4 + 256/K | plain Lloyd-Max |

All Lloyd fits use one batched, deterministic 1-D solver (`lloyd_1d`): it sorts each group,
computes centroids from float64 prefix sums, sends exact midpoint ties to the lower level,
leaves empty cells unchanged, and stops at the fixed point or after 100 iterations. Per-group
and per-row LUTs start from NF4×absmax, so they are never worse than NF4 in MSE. A
3584×1024 matrix takes under 0.1 s on GPU. Each run's `coverage` now also includes
`w_bpw_replaced` (nominal bits per weight over the replaced linears),
`w_rel_err_by_format` (‖W−Q(W)‖²/‖W‖²) and `modules_by_format`.

Caveats: (1) APoT is a QAT method in its paper, with a learned per-layer clip α on
standardised weights. Here it is RTN with α = group absmax. Only the level set is taken from
the official code. The definition is unambiguous (the `B=bit-1` weight grid from the code), so
no substitute was needed, and AF4 is included as well. (2) The BOF4/AF4 codebooks were
optimised for Gaussian weights at B=64. `lut4_bof4fit` refits on the real weights but uses
the same assignment rule. OPQ (outlier preservation) is not implemented. (3) BOF4-S
signed-absmax ties use the lowest index (upstream uses `torch.max` indices). All-zero blocks
get scale 1 (upstream would divide by 0). (4) `lut4_lloyd_g64` costs 8 bpw because each group
stores its own 16 BF16 entries, so it is an upper bound and not a 4-bit format. (5) Every mode
is RTN only (no GPTQ/AWQ), fake-quantized with a BF16 matmul. GDN linears are quantized
unless `--exclude-gdn` is set, and lm_head stays BF16.

### ASIC-verified weight-only formats (`--modes asic`)

These two formats are defined in the local ASIC ISA simulator `/home/bhat/asic/v2/isa_sim`
(commit `2039157`): its quantizers are in `gen.py` and its dequantizers in `ref.py`. The
self-tests import those files and use them as oracles. Each small tensor is passed through
`gen.gen_iq4nl` / `gen.gen_bcq` in place of their random weights, unpacked with
`ref.ref_iq4nl` / `ref.ref_bcq`, and compared with `torch.equal`. The comparison covers
the codes, bits, fp16 scales and offsets, and the dequantized FP32 values. The inputs are
heavy-row, wide-range, heavy-tailed and bf16 data, an all-zero block, and an IQ4 block
placed exactly on code midpoints. Mutation checks were run to confirm that the tests
fail when the tie rule or the fp16 rounding of α is changed.

| Mode | Definition | bpw | PPL (4,096 tok) | ‖W−Q‖²/‖W‖² |
|---|---|---:|---:|---:|
| `w4_iq4nl_g32` | GGUF IQ4_NL: `kvalues_iq4nl` (verbatim from llama.cpp via `ggml_tables.py`). Per 32 weights, `d = fp16(absmax/127)` and `code = nearest(x/d)`, with exact midpoint ties going to the lower code. Dequant is `f32(d)·kvalues[code]` | 4.5 | **22.609** | 0.0078 |
| `w3_bcq3_g128` | BCQ / LUT-GEMM, per output row × 128 input columns: `z = fp16(mean)`, then for 3 planes `b = (res ≥ 0)`, `α = fp16(mean\|res\|)`, `res -= α(2b−1)`. Dequant is the FP32 sum of ±α in plane order, then `+ z` | 3.5 | 245.80 | 0.0659 |

The BF16 control in the same run gives 21.168, and GDN linears are quantized in both modes.
Raw results are in `results/asic_iq4nl_bcq_4096.json`. For reference, NF4 g64 with the same
coverage gives 23.441 PPL and 0.0087 relative error.

Caveats:
1. IQ4_NL follows the simulator's generator rule, not llama.cpp's `quantize_row_iq4_nl`.
   llama.cpp uses a signed max (`d = max/−127`), a ±7-step scale search, and an optional
   imatrix weighting, so real GGUF IQ4_NL is probably a little better. The table, block
   layout, fp16 scale and dequant are the same as in llama.cpp.
2. BCQ uses the plain greedy residual fit that the simulator defines. It has no
   alternating refinement of α and the bits (Xu et al. 2018; LUT-GEMM/ShiftAddLLM use better
   fits or calibration). Greedy α is the mean residual magnitude, which is not least-squares
   optimal once the planes interact. Its weight relative error (0.066) is worse than NF3 g64
   (0.042, PPL 36.3) and 3-bit min-max INT3 g128 (0.051, weight error only, no PPL run).
   So the 245.8 is a fair result for *greedy* RTN BCQ at 3 bits on this model, not a bug.
   It should not be read as a bound on what BCQ can achieve. Padded tail columns are left
   out of the means, which never happens for Qwen3.5 since K is a multiple of 128.

### Activation quantization (`--modes act`)

Same 4,096-prediction protocol and harness. Coverage is the fraction of linear weights whose
**input activation** is quantized: 186 linears / 66.2% by default and 96 linears / 41.0% with
`--exclude-gdn`. lm_head stays BF16. All rows are fake-quant simulation (quantize, dequantize,
then a BF16 matmul). **None of these numbers say anything about native FP8/INT8/MX kernel speed.**
Raw results are in `results/activation_4096.json` and `results/activation_exclude_gdn_4096.json`.

| Mode | W | A | Calib. | PPL, GDN quantized (66.2%) | Δ | PPL, GDN BF16 (41.0%) | Δ |
|---|---|---|---|---:|---:|---:|---:|
| `bf16` | BF16 | BF16 | — | 21.168 | — | 21.168 | — |
| `a8_int8` | BF16 | INT8 per-token dynamic | — | 21.388 | +1.04% | 21.324 | +0.74% |
| `a8_fp8` | BF16 | FP8 E4M3 per-token dynamic | — | 21.413 | +1.16% | 21.279 | +0.53% |
| `a8_fp8_static` | BF16 | FP8 E4M3 per-tensor **static** | 32×512 | **21.367** | **+0.94%** | 21.285 | +0.56% |
| `a8_mxfp8` | BF16 | MXFP8 E4M3, block 32, E8M0 | — | 21.443 | +1.30% | 21.353 | +0.87% |
| `a4_int4` | BF16 | INT4 per-token dynamic | — | 1582.8 | +7377% | 326.2 | +1441% |
| `w8a8_int8` | INT8 per-channel | INT8 per-token | — | 21.477 | +1.46% | 21.398 | +1.09% |
| `sq05_bf16` (fold only) | BF16 (W·s) | BF16 | 32×512 | 21.189 | +0.10% | 21.173 | +0.02% |
| `w8a8_int8_sq05` | INT8 per-channel, SmoothQuant α=0.5 | INT8 per-token | 32×512 | 21.511 | +1.62% | 21.358 | +0.90% |
| `w8a8_fp8` | FP8 per-channel | FP8 per-token dynamic | — | 21.291 | +0.58% | **21.182** | **+0.07%** |
| `w8a8_fp8_static` | FP8 per-channel | FP8 per-tensor **static** | 32×512 | **21.261** | **+0.44%** | 21.303 | +0.64% |
| `w8a8_mxfp8` | **MXFP8 E4M3 (FP8 weights)**, block 32 | MXFP8 E4M3, block 32 | — | 21.645 | +2.26% | 21.688 | +2.46% |

Takeaways:
- Every 8-bit activation format costs about 0.5–1.3% PPL by itself. GDN projections cause
  roughly half of that: excluding them halves the A8 loss for every format.
- MXFP8 (power-of-two E8M0 scale per 32) is *worse* than per-token FP8 here, both
  activation-only and as W8A8. Floor-based E8M0 leaves the block max between 256 and 512, so a
  block uses between 1.75× and 0.875× of E4M3's range. That costs up to about 1 bit of mantissa
  headroom per block, compared with an exact FP32 per-token scale. Qwen3.5-0.8B activations
  have no per-block outliers large enough to make up for that loss.
- Static per-tensor FP8 calibrated on 16k tokens is as good as dynamic per-token, and slightly
  better on the full-coverage run. Only about 2–3×10⁻⁸ of evaluation activations exceeded the
  calibrated range and saturated. The static scale is also the *deployable* one (no per-call
  amax reduction).
- SmoothQuant α=0.5 does **not** help INT8 W8A8 on this model when the GDN projections are
  quantized (+1.62% vs +1.46% without it). With GDN in BF16 it helps a little
  (+0.90% vs +1.09%). Per-token dynamic INT8 already absorbs most of the per-token range, and
  migrating channel scale into W raises the INT8 weight error by 19% (relerr 8.35e-5 → 9.97e-5).
- INT4 per-token activations without rotation or outlier handling are unusable (PPL > 300).
  This mode is a floor reference, not a candidate.

Definitions and sources:
- **FP8 E4M3 per-token dynamic** (`a8_fp8`): s = amax_token/448, RNE to E4M3FN (via
  `torch.float8_e4m3fn`), saturate at ±448. These are the compressed-tensors `FP8_DYNAMIC`
  activation args (`FLOAT`, 8 bit, `TOKEN`, dynamic, symmetric). `w8a8_fp8` is the existing
  mode with per-channel FP8 weights.
- **MXFP8** (`q_mxfp8`): OCP MX v1.0 §6.3 with E4M3 elements, k=32 along K, and an E8M0 scale
  X = 2^(⌊log₂ amax⌋ − 8) (emax_elem = 8 because 448 = 2⁸·1.75), clamped to [−127, 127].
  Elements are RNE with clamp-to-max-normal (SAT). `w8a8_mxfp8` uses the same format for the
  weights, i.e. **FP8 weights**, matching the compressed-tensors `MXFP8` preset. It is not an
  INT8-weight variant. The 8.25 bpw counts the E8M0 byte per 32. The test oracle is a verbatim
  copy of `microsoft/microxcaling` `mx/mx_ops.py::_quantize_mx(..., "fp8_e4m3", block_size=32,
  round="even")` @ `7bc4195` (MIT, `third_party/microxcaling_mx/`). It is bit-identical
  (`torch.equal`) on CPU and CUDA on random, ±6-decade, heavy-tailed, zero, bf16, saturating
  (block max in 448–512·2ᵏ) and RNE-tie inputs. Mutating emax to 7 fails both MXFP8 tests.
- **INT4 per-token** (`a4_int4`): s = amax/7, round-half-even, clamp to [−7, 7]. This is
  SmoothQuant's `quantize_activation_per_token_absmax(n_bits=4)` and is tested equal.
- **Static per-tensor FP8** (`*_fp8_static`): s = max|x| over the calibration set / 448 per
  linear (compressed-tensors `FP8` preset activations: `TENSOR`, `dynamic=False`,
  `observer="static_minmax"`). The scale is fixed at evaluation time and values above 448·s
  saturate; the saturated fraction is reported in `coverage.static_saturated_frac`.
  Statistics come from the unquantized BF16 model (one shared pass for all modes), not
  sequentially from the quantized model.
- **Calibration set** (static FP8 and SmoothQuant): the first 32×512 = 16,384 tokens of
  WikiText-2 **train** (`pytorch/examples`, the same repo as the test text, so it is disjoint
  from evaluation), cut into 32 consecutive windows. Settings: `--calib-samples`,
  `--calib-seqlen`. Upstream SmoothQuant uses 512 Pile samples; this set is smaller and from
  the same domain as the test text.
- **SmoothQuant** (`w8a8_int8_sq05`): Xiao et al., ICML'23,
  [mit-han-lab/smoothquant](https://github.com/mit-han-lab/smoothquant) @ `c61476d`.
  Calibration follows `get_act_scales` (per-input-channel max|x|). Scales follow
  `smooth_ln_fcs_llama_like`: s_j = max|X_j|^α / max_fc max_i|W_fc[i,j]|^(1−α), with both
  clamps at 1e-5. A test checks this against the verbatim upstream `smooth.py`
  (`third_party/smoothquant_smooth.py`) on a `LlamaRMSNorm`. Quantization follows
  `fake_quant.W8A8Linear` with `weight_quant="per_channel"` and `act_quant="per_token"`
  (dynamic absmax/127), without BMM/KV quantization. Weights are quantized from the FP32
  product W·s.

SmoothQuant on Qwen3.5 module boundaries: where it was applied and where it was skipped.

| Boundary | Consumers | Smoothed? | Why |
|---|---|---|---|
| full-attn `input_layernorm` | `q_proj` (q+gate), `k_proj`, `v_proj` | yes | norm output feeds only these |
| GDN `input_layernorm` | `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a` | yes (not with `--exclude-gdn`) | only `apply_mask_to_padding_states` (a per-token multiply, which commutes with per-channel scaling) sits in between |
| `post_attention_layernorm` | `gate_proj`, `up_proj` | yes | as Llama |
| `o_proj` input | attn·sigmoid(gate) | **no** | no preceding per-channel linear op to fold into (upstream skips it too) |
| `down_proj` input | SiLU(gate)·up | **no** | no exact fold (upstream skips it too) |
| GDN `out_proj` input | `RMSNormGated(core, z)` | **no** | its weight is per head_dim (128), shared by all 16 heads, so a per-channel 2048-wide scale cannot be folded exactly |
| `lm_head` | final norm | **no** | lm_head is tied to the embedding and stays BF16 |

Qwen3.5's RMSNorm is zero-centred, `rmsnorm(x)·(1+w)`, so the fold sets (1+w) ← (1+w)/s.
This is exact algebra; `_check_zero_centered_rmsnorm` checks it at runtime and refuses any
other norm type. Three checks back this up. The smoothed linears and the norm weight are kept
in FP32. On a tiny random Qwen3.5 (GDN + full attention) the fold changes logits by less than
1e-4. On the real model in FP32 the fold changes PPL by 2×10⁻⁷ relative (21.15658 →
21.15658; one-off check, not in the JSON). The +0.10% of `sq05_bf16` in BF16 therefore comes
only from re-rounding the unquantized W·s back to BF16, a cost the quantized modes do not
pay. In total, 48 norms and 138 linears (359M weights, 72% of the replaced linears) are
smoothed, with s ∈ [2.96, 25.1]. With `--exclude-gdn` the counts are 30 norms and 66 linears.

Caveats: (1) This is simulation only. The FP8/MX rows assume perfect FP32 accumulation and
dequantization, and no native kernel was run or timed. (2) Activation scales for dynamic modes
are computed per call over a 512-token window. "Per-token" is exact per token, and MXFP8 is
per 32 channels of each token. (3) GDN recurrent state, conv1d, the attention softmax/BMM and
the KV cache stay BF16 in every mode. (4) Static scales and SmoothQuant use one fixed 16k-token
in-domain calibration set, with no seed or sample-size sweep. α was not tuned (the task fixed
it at 0.5). (5) The `w8a8_int8` and `w8a8_fp8` rows reproduce the earlier
`advanced_4096.json` values exactly (21.477 / 21.291).

## MR-GPTQ (`mr_gptq/`)

Port of IST-DASLab/FP-Quant MR-GPTQ (Hadamard + GPTQ on MXFP4/NVFP4 grids) to the Qwen3.5
hybrid, evaluated on this protocol. Full 128x2048 calibration gives MXFP4 Had128 PPL 25.004,
NVFP4 Had16 PPL 23.358, and NVFP4 without rotation PPL 22.649. See `mr_gptq/README.md`.
