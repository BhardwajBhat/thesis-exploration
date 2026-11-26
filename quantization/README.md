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
`mixed_nvfp4_fp8gdn` (NVFP4 everywhere except GDN `linear_attn.*` = FP8).
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
