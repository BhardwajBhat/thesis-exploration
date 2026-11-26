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
| INT8 W8A16 | 8/16 | per-row RTN | quantized | 66.2% | 21.278 | +0.52% | Sim |
| FP8 dynamic checkpoint | 8/8 | dynamic | checkpoint-defined | checkpoint | 21.279 | +0.52% | HF checkpoint |
| FP8 W8A8 | 8/8 | dynamic RTN | quantized | 66.2% | 21.291 | +0.58% | Sim |
| Activation-only INT8 | 16/8 | per-token RTN | quantized | 66.2% | 21.388 | +1.04% | Sim |
| INT8 W8A8 | 8/8 | dynamic RTN | quantized | 66.2% | 21.477 | +1.46% | Sim |
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
remain BF16. For weight-only 4-bit, calibrated GPTQ is the clear winner. None of
the tested W4A4 formats is close to lossless: MR-GPTQ improves MXFP4 strongly,
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
Weight-only LUT/codebook modes (`w3_nf3_g64`, `w4_fp4_g32/g64`, `w4_af4_g64`, `w4_bof4_g64`,
`w4_bof4s_g64`, `w4_apot4_g64`, `w4_lut4_*`) are described below (`--modes lut`).
Options: `--exclude-gdn`, `--include-lm-head`, `--compute-dtype {bf16,fp32}`,
`--self-test` (unit tests for the INT8/E4M3/E2M1/E8M0/NF4/NF3/AF4/BOF4/APoT grids, the
nearest-code rules and Lloyd-Max).

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

No PPL has been measured for these yet (only self-tests have been run). Run them with
`--modes lut` (expands to `bf16`, `w4_nf4_g64` and all modes below). The default `--modes`
list is still the old 9 modes, and nothing in `results/` was changed.

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

## MR-GPTQ (`mr_gptq/`)

Port of IST-DASLab/FP-Quant MR-GPTQ (Hadamard + GPTQ on MXFP4/NVFP4 grids) to the Qwen3.5
hybrid, evaluated on this protocol. Full 128x2048 calibration gives MXFP4 Had128 PPL 25.004,
NVFP4 Had16 PPL 23.358, and NVFP4 without rotation PPL 22.649. See `mr_gptq/README.md`.
