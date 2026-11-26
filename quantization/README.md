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
