# Qwen3.5-0.8B quantization perplexity

Quick comparison using the same first 4,096 next-token predictions from the
PyTorch examples WikiText-2 test text. Eight independent 512-token windows;
logits are converted to FP32 for cross entropy.

| Format | PPL | Increase over BF16 |
|---|---:|---:|
| BF16 | 21.168 | — |
| AWQ INT8 | 21.208 | 0.19% |
| INT8 symmetric RTN | 21.278 | 0.52% |
| FP8 dynamic | 21.279 | 0.52% |
| GPTQ INT4 | 21.612 | 2.10% |
| AWQ INT4 | 22.782 | 7.62% |
| INT4 symmetric RTN | 42.303 | 99.85% |

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

## Advanced fake-quant harness (`advanced_ppl.py`, not yet run on GPU)

Same protocol as above (4,096 predictions, 8 x 512 windows). Selected
`nn.Linear`s are replaced by `QLinear`, which fake-quantizes the weight once
and the **input activation on every call**. Modes: `bf16`, `a8_int8`,
`w8a8_int8`, `w8a8_fp8`, `w4a4_mxfp4`, `w4a4_nvfp4`, `w4_nf4_g64`,
`mixed_nvfp4_fp8gdn` (NVFP4 everywhere except GDN `linear_attn.*` = FP8), `w4a4_m2xfp` (M²XFP: MXES weights / MXEM activations, group 32, subgroup 8, 4.5 bit).
Options: `--exclude-gdn`, `--include-lm-head`, `--compute-dtype {bf16,fp32}`,
`--self-test` (CPU unit tests for the INT8/E4M3/E2M1/E8M0/NF4 grids).

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
RTN plus scale search only: no GPTQ or rotation. Not run on the model yet.

## MR-GPTQ (`mr_gptq/`)

Port of IST-DASLab/FP-Quant MR-GPTQ (Hadamard + GPTQ on MXFP4/NVFP4 grids) to the Qwen3.5
hybrid, evaluated on this protocol. Smoke (8x512 calib): W4A4 MXFP4 30.33 -> 25.23,
W4A4 NVFP4 24.81 -> 23.65 (GPTQ, no rotation; Had16 gives 24.66). See `mr_gptq/README.md`.
