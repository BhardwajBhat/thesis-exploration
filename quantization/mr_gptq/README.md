# MR-GPTQ for Qwen3.5-0.8B (MXFP4 / NVFP4 + block Hadamard)

Port of the **official IST-DASLab/FP-Quant** MR-GPTQ path (`gptq_quantization()`, commit
`d2e3092f968262c4de5fb050e1aef568a280dadd`, arXiv:2509.23202) to the Qwen3.5 hybrid
(Gated-DeltaNet + gated attention) architecture. Results go through the same raw
WikiText-2 protocol as `../advanced_ppl.py` (first 4,097 tokens of pytorch/examples
`test.txt`, 8 independent 512-token windows, 4,096 predictions, FP32 CE); `eval_ppl` is
imported directly from that file.

Files
- `fpquant_port.py`: Quantizer (absmax/MSE observer; NVFP E4M3 block scales plus FP32 global
  scale; MXFP E8M0 with upstream's 4/3 rescale), GPTQ (fixed grid, static act-order, damp
  1e-2, block 128), block Hadamard and FP4 packing. The structure follows upstream closely.
- `qwen35_mr_gptq.py`: the Qwen3.5 traversal, sequential layer-wise calibration and eval.
- `test_mr_gptq.py`: unit tests. With `FPQUANT_SRC=<clone>` it also checks **parity against upstream**:
  quantizer scales and outputs are bit-exact. GPTQ outputs differ in <0.1% of elements; the
  difference comes only from upstream's TF32 Triton Hessian kernel.

## Qwen3.5 port

Upstream supports only Llama/Qwen3, and it builds rotated attention/MLP classes for each
architecture. Here each target `nn.Linear` is wrapped instead. The forward becomes
`y = Qa(H·x) · (W·H)ᵀ`: activations are rotated online and the rotation is fused into the
weights. That is exact at full precision because H is symmetric and orthogonal. Linears
that share an input share one transform and one activation quantizer, as upstream does
for qkv and gate_up:

| group | modules | K |
|---|---|---:|
| gdn_in | `linear_attn.in_proj_{qkv,z,b,a}` | 1024 |
| gdn_out | `linear_attn.out_proj` | 2048 |
| qkv | `self_attn.{q (incl. output-gate rows),k,v}_proj` | 1024 |
| o | `self_attn.o_proj` | 2048 |
| gate_up | `mlp.{gate,up}_proj` | 1024 |
| down | `mlp.down_proj` | 3584 = 28·128 |

Any unmapped linear inside a decoder layer raises an error. The script quantizes 186
linears (497.6M weights), the same coverage as `advanced_ppl.py`'s defaults. These parts
stay BF16: embeddings, the tied lm_head, conv1d, norms, A_log/dt_bias, the GDN recurrence,
attention softmax and the KV cache. `--exclude-gdn` also leaves the GDN linears in BF16.

The per-layer pipeline follows upstream steps 1–8:

1. Build the transforms and wrap the linears.
2. Freeze the NVFP weight global scale from the rotated weight.
3. Run the BF16-weight forward over the calibration inputs. Activation quantization is
   live, and the Hessians are collected on the rotated inputs. The activation global scale
   is tracked during this pass and then frozen (static).
4. Rotate the weights.
5. Run GPTQ on the fixed grid.
6. Re-run the quantized block to produce the next layer's inputs.

Deviations from upstream:
- No fp16 `--amp` autocast. Everything runs in bf16, and activations are quantized in fp32.
- The Hadamard is an explicit fp32 Sylvester matmul. It has the same ordering and scale as
  `fast_hadamard_transform`, so no CUDA extension build is needed.
- The Hessian uses an fp32 `addmm` instead of the TF32 Triton kernel.
- The export (`--save-dir`) is a flat safetensors file: packed FP4 codes, scales and global
  scales per linear. It is **not** a loadable FP-Quant/vLLM checkpoint. Writing that
  requires adapting `export_quantized_model` plus `FPQuantConfig` to Qwen3.5.

Defaults are the paper's MR-GPTQ choices: MSE observer, static act-order, Had16 for NVFP4
and Had128 for MXFP4 (paper App. I), and FineWeb-Edu calibration (FP-Quant `get_fineweb_edu`).
The README script uses 128 × 2048 sequences; the paper uses 1024 FineWeb sequences.

## Smoke results (8 × 512 FineWeb-Edu calibration only; **not** a converged sweep)

The model is Qwen/Qwen3.5-0.8B; BF16 PPL is 21.168. All W4A4 rows use the FP-Quant
activation quantizer (RTN, static global scale for NVFP). Raw JSON is in `results/smoke_*.json`.

| format | A | method | transform | observer / order | PPL |
|---|---|---|---|---|---:|
| NVFP4 | 4 | RTN | none | absmax / – | 24.814 |
| NVFP4 | 4 | RTN | Had16 | MSE | 25.131 |
| NVFP4 | 4 | GPTQ | none | MSE / act | **23.654** |
| NVFP4 | 4 | MR-GPTQ | Had16 | MSE / act | 24.661 |
| MXFP4 | 4 | RTN | none | absmax / – | 30.326 |
| MXFP4 | 4 | RTN | Had128 | MSE | 29.027 |
| MXFP4 | 4 | MR-GPTQ | Had128 | MSE / act | **25.227** |
| NVFP4 | 16 | RTN | none | absmax | 23.300 |
| NVFP4 | 16 | MR-GPTQ | Had16 | MSE / act | **21.971** |
| MXFP4 | 16 | RTN | none | absmax | 23.475 |
| MXFP4 | 16 | MR-GPTQ | Had128 | MSE / act | **22.656** |

Reference points from `advanced_ppl.py`, which uses dynamic activation scales:
W4A4 NVFP4 RTN 24.655 and MXFP4 RTN 29.259.

Early read:
- MR-GPTQ closes most of the MXFP4 gap (30.3 → 25.2 W4A4).
- For W4A4 NVFP4 on this model, Had16 *hurts*: plain GPTQ (23.65) beats MR-GPTQ (24.66).
  This is consistent with the paper's point that NVFP4's 16-element groups already
  neutralize outliers. It needs confirmation at full calibration size before any claim.
- Runtime is about 2.5 s per layer on an RTX 4090 at 8 × 512, and about 2–4 min per config
  in total. Fetching the calibration data from FineWeb streaming costs about 75 s.

## Commands

    .venv/bin/python quantization/mr_gptq/test_mr_gptq.py
    FPQUANT_SRC=/tmp/mrgptq_src/FP-Quant .venv/bin/python quantization/mr_gptq/test_mr_gptq.py   # + upstream parity
    # paper-like configs (not run yet):
    .venv/bin/python quantization/mr_gptq/qwen35_mr_gptq.py --format nvfp --nsamples 128 --seqlen 2048 --output quantization/mr_gptq/results/nvfp_w4a4_mrgptq.json
    .venv/bin/python quantization/mr_gptq/qwen35_mr_gptq.py --format mxfp --nsamples 128 --seqlen 2048 --output quantization/mr_gptq/results/mxfp_w4a4_mrgptq.json
    # ablations: --transform identity | --hadamard 32/128 | --method rtn | --w-observer minmax | --order default | --a-bits 16 | --exclude-gdn | --fuse-global-scale

No new packages are needed; the repo `.venv` is used unchanged. The upstream clone is
needed only for the parity test: `git clone https://github.com/IST-DASLab/FP-Quant /tmp/mrgptq_src/FP-Quant`.
At the pinned commit, upstream has no LICENSE file. The port is a re-implementation kept
close to the upstream code, with attribution. Check this before any redistribution.
