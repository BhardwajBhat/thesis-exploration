#!/usr/bin/env python3
"""Fake-quant W/A perplexity harness for Qwen/Qwen3.5-0.8B.

Same raw-WikiText protocol as quick_ppl.py / results/qwen35_08b_wikitext2_4096.json:
first 4097 tokenizer tokens of pytorch/examples wikitext-2 test.txt, 8 independent
512-token windows, 4096 next-token predictions, FP32 cross entropy.

Every selected nn.Linear is replaced by QLinear, which holds a fake-quantized
(quantize -> dequantize) copy of the weight and fake-quantizes its *input* on every
forward call, so activation quantization is really applied. The original weight
tensor is never modified, so tied embeddings stay BF16 even if lm_head is included.

Modes (W = weight format, A = activation format; blocks run along the K / input dim):
  bf16                 reference (no replacement)
  a8_int8              A: INT8 symmetric per-token; W: BF16
  w8a8_int8            W: INT8 symmetric per-output-channel; A: INT8 per-token
  w8a8_fp8             W: FP8 E4M3 per-output-channel scale; A: FP8 E4M3 per-token scale
  w4a4_mxfp4           W/A: OCP MXFP4, E2M1 elements, block 32, E8M0 power-of-two scale
  w4a4_nvfp4           W/A: NVFP4, E2M1 elements, block 16, E4M3 block scale x FP32 tensor scale
  w4_nf4_g64           W: NF4 (QLoRA codebook), absmax group 64; A: BF16
  Weight-only LUT/codebook family (A: BF16; see "LUT / codebook weight quantizers" below):
  w3_nf3_g64           NF3 (LoftQ create_normal_map(num_bits=3)), absmax group 64
  w4_fp4_g32/_g64      OCP E2M1 grid, absmax->6 FP32 scale per group (fixed-codebook FP4)
  w4_af4_g64           AF4-64 (Yoshida 2023), absmax group 64
  w4_bof4_g64          BOF4 (MSE, B=64), absmax group 64 (Blumenberg et al. 2025)
  w4_bof4s_g64         BOF4-S (MSE, B=64), signed-absmax group 64
  w4_apot4_g64         APoT 4-bit weight levels (Li et al. ICLR'20), absmax group 64 (PTQ scaling)
  w4_lut4_bof4fit_g64  per-tensor 16-entry LUT fitted by BOF4's empirical EM (weighted Lloyd)
                       on this tensor's absmax-normalised g64 blocks, init = BOF4 codebook
  w4_lut4_lloyd_g64    per-group (64) own 16-entry Lloyd-Max LUT, BF16 entries (8 bpw!)
  w4_lut4_lloyd_row    per-output-row 16-entry Lloyd-Max LUT (SqueezeLLM-style, unweighted)
  mixed_nvfp4_fp8gdn   NVFP4 W4A4 on ordinary linears, FP8 W8A8 on linear_attn.* (GDN)
  w4a4_m2xfp           M2XFP (ASPLOS'26), port of official SJTU-ReArch-Group/M2XFP_ASPLOS26
                       quant_func.py: W = get_quant_mxes (Sg-EM-2bit adaptive scale),
                       A = get_quant_mxem (Elem-EM-top1 FP6), group 32 / subgroup 8, 4.5 bit

Usage:
  python quantization/advanced_ppl.py --self-test
  python quantization/advanced_ppl.py --modes bf16 w8a8_int8 w4a4_nvfp4 --output out.json
  python quantization/advanced_ppl.py --modes w4a4_mxfp4 --exclude-gdn
"""
import argparse, json, math, re, sys, time, urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL = "Qwen/Qwen3.5-0.8B"
URL = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/test.txt"
GDN_RE = re.compile(r"(^|\.)linear_attn\.")

FP8_MAX = 448.0          # E4M3FN max finite
FP4_MAX = 6.0            # E2M1 max
FP4_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
# QLoRA NF4 codebook (bitsandbytes create_normal_map, 16 values incl. exact 0)
NF4_GRID = torch.tensor([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0])

# ----------------------------------------------------------------------------- element grids

def round_fp8_e4m3(x):
    """Round FP32 to nearest E4M3FN value (RNE via torch cast), saturating at +-448."""
    return x.clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).to(torch.float32)

def round_fp4_e2m1(x):
    """Round to nearest E2M1 value, ties-to-even mantissa, saturating at +-6."""
    a = x.abs()
    g = FP4_GRID.to(x.device)
    # midpoints between grid values; boundary i separates grid[i] (even i -> even code)
    # and grid[i+1]. Tie goes to the even code: down for even i (strict >), up for odd i (>=).
    idx = torch.zeros_like(a, dtype=torch.long)
    for i, m in enumerate((0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)):
        idx += (a > m) if i % 2 == 0 else (a >= m)
    return torch.sign(x) * g[idx]

def round_codebook(x, grid):
    """Nearest-value rounding to an arbitrary sorted codebook (ties go to lower value)."""
    g = grid.to(x.device)
    mids = (g[1:] + g[:-1]) / 2
    return g[torch.bucketize(x, mids)]

# ----------------------------------------------------------------------------- blocking helpers

def _blocks(x, bs):
    """Reshape [..., K] -> [..., K/bs, bs] padding K with zeros. Returns (view, K)."""
    K = x.shape[-1]
    pad = (-K) % bs
    if pad:
        x = F.pad(x, (0, pad))
    return x.reshape(*x.shape[:-1], -1, bs), K

def _unblock(xb, K):
    return xb.reshape(*xb.shape[:-2], -1)[..., :K]

# ----------------------------------------------------------------------------- quantizers
# All take/return FP32 and reduce along the last dim (= K for both W [N,K] and A [..,K]).
# For weights "per-row" means per output channel; for activations it means per token.

def q_int8_row(x):
    s = x.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 127.0
    return (x / s).round().clamp(-127, 127) * s

def q_fp8_row(x):
    s = x.abs().amax(-1, keepdim=True).clamp_min(1e-12) / FP8_MAX
    return round_fp8_e4m3(x / s) * s

def mx_scale_exp(amax):
    """OCP MX scale exponent: floor(log2(amax)) - emax_elem (2 for E2M1), clamped to E8M0."""
    e = torch.floor(torch.log2(amax.clamp_min(2.0 ** -127))) - 2
    return e.clamp(-127, 127)

def q_mxfp4(x, bs=32):
    xb, K = _blocks(x, bs)
    amax = xb.abs().amax(-1, keepdim=True)
    s = torch.exp2(mx_scale_exp(amax))
    s = torch.where(amax > 0, s, torch.ones_like(s))
    return _unblock(round_fp4_e2m1(xb / s) * s, K)

def nvfp4_scales(xb, tensor_amax):
    """Per-block E4M3 scale (decoded, FP32) and FP32 per-tensor scale, per NVFP4 recipe:
    global = amax_tensor / (448*6); block_scale = e4m3(amax_block / 6 / global)."""
    g = (tensor_amax / (FP8_MAX * FP4_MAX)).clamp_min(1e-12)
    amax = xb.abs().amax(-1, keepdim=True)
    sb = round_fp8_e4m3(amax / FP4_MAX / g)
    return sb, g

def q_nvfp4(x, bs=16):
    """Dynamic NVFP4. Tensor scale from the current tensor's amax (weights: whole matrix;
    activations: whole call). Real deployments usually calibrate a static activation
    global scale; dynamic is the optimistic case."""
    xb, K = _blocks(x, bs)
    sb, g = nvfp4_scales(xb, x.abs().amax())
    s = sb * g
    safe = torch.where(s > 0, s, torch.ones_like(s))
    q = torch.where(s > 0, round_fp4_e2m1(xb / safe), torch.zeros_like(xb))
    return _unblock(q * s, K)

def q_nf4(x, bs=64):
    xb, K = _blocks(x, bs)
    s = xb.abs().amax(-1, keepdim=True).clamp_min(1e-12)
    return _unblock(round_codebook(xb / s, NF4_GRID) * s, K)

# ----------------------------------------------------------------------------- LUT / codebook weight quantizers
# All grids below are copied from primary sources (no invented values):
#
# NF3: LoftQ (Li et al., ICLR'24) github.com/yxli2123/LoftQ @ae33fd4 glue/utils_qaunt.py
#   create_normal_map(offset=0.9677083, symmetric=False, num_bits=3). The same function with
#   num_bits=4 reproduces NF4_GRID bit-exactly (see _create_normal_map / tests). bitsandbytes has
#   no NF3; this is the k-bit generalisation of QLoRA's NF construction used by LoftQ.
#   Asymmetric like NF4: 3 negative levels, exact 0, 4 positive levels.
NF3_GRID = torch.tensor([-1.0, -0.4786292016506195, -0.217141792178154, 0.0,
                         0.16093020141124725, 0.33791524171829224, 0.5626170039176941, 1.0])
# AF4-64 (Yoshida 2023, arXiv:2306.06965 "NF4 isn't information theoretically optimal"): codes
# from generate_af4.py (github.com/davisyoshida/abnormal-floats), as distributed float32 in
# github.com/ifnspaml/bof4 @a447ef1 codebooks/af4/af4_64 (Apache-2.0). Fixed -1, 0, +1.
AF4_64_GRID = torch.tensor([
    -1.0, -0.6944100856781006, -0.5124374032020569, -0.37369510531425476,
    -0.25607553124427795, -0.14982476830482483, -0.049348123371601105, 0.0,
    0.0427316389977932, 0.12934483587741852, 0.2196127325296402, 0.3167566657066345,
    0.4256388247013092, 0.5549623370170593, 0.7242486476898193, 1.0])
# BOF4 (Blumenberg, Graave, Fingscheidt 2025, arXiv:2505.06653), official codebooks from
# github.com/ifnspaml/bof4 @a447ef1 (Apache-2.0):
#   codebooks/bof4/bof4_mse_64    normalization_method: absmax         (no level fixed)
#   codebooks/bof4/bof4-s_mse_64  normalization_method: signed_absmax  (0 and +1 fixed)
BOF4_MSE_64_GRID = torch.tensor([
    -0.9532026648521423, -0.7398636341094971, -0.5820896029472351, -0.4524894952774048,
    -0.33935657143592834, -0.23635068535804749, -0.13954706490039825, -0.046129997819662094,
    0.04622155427932739, 0.13964194059371948, 0.2364330291748047, 0.3394077718257904,
    0.45256364345550537, 0.5821195244789124, 0.7398874759674072, 0.9532160758972168])
BOF4S_MSE_64_GRID = torch.tensor([
    -0.8568463921546936, -0.6692874431610107, -0.5235266089439392, -0.4004882574081421,
    -0.2910638153553009, -0.19000929594039917, -0.09385295957326889, 0.0,
    0.0887671709060669, 0.17948026955127716, 0.27430960536003113, 0.37601974606513977,
    0.4886530041694641, 0.6188603639602661, 0.7791395783424377, 1.0])

def _create_normal_map(offset=0.9677083, num_bits=4):
    """Torch-only port of LoftQ/bitsandbytes create_normal_map (symmetric=False); scipy
    norm.ppf replaced by sqrt(2)*erfinv(2p-1) in float64 (bit-identical after float32 cast)."""
    n = 2 ** num_bits
    ppf = lambda p: (2 ** 0.5 * torch.erfinv(2 * p.double() - 1)).tolist()
    v1 = ppf(torch.linspace(offset, 0.5, n // 2 + 1)[:-1])
    v3 = [-v for v in ppf(torch.linspace(offset, 0.5, n // 2)[:-1])]
    v = torch.Tensor(v1 + [0] + v3).sort().values
    return v / v.max()

def _apot_build_power_value(B=3, additive=True):
    """Port of build_power_value() from github.com/yhhhli/APoT_Quantization @13204f1
    (ImageNet/models/quant_layer.py; identical in CIFAR10/). Non-negative levels with clip
    threshold 1. A b-bit *weight* uses B = b-1 (QuantConv2d: build_power_value(B=bit-1)) plus a
    sign: data.clamp(-1,1) -> nearest level of |x| -> * sign. Returned sorted (official returns
    an unordered torch.Tensor(list(set(values))); the set of values is identical)."""
    a, b, c = [0.], [0.], [0.]
    if not additive:
        a += [2 ** (-i - 1) for i in range(2 ** B - 1)]
    elif B == 2:
        a += [2 ** (-i - 1) for i in range(3)]
    elif B == 4:
        for i in range(3):
            a.append(2 ** (-2 * i - 1)); b.append(2 ** (-2 * i - 2))
    elif B == 3:
        for i in range(3):
            if i < 2:
                a.append(2 ** (-i - 1))
            else:
                b.append(2 ** (-i - 1)); a.append(2 ** (-i - 2))
    else:
        raise NotImplementedError(B)  # B=5/6 exist upstream; not needed here
    vals = torch.Tensor(list({x + y + z for x in a for y in b for z in c}))
    return (vals * (1.0 / torch.max(vals))).sort().values

APOT4_W_MAG = _apot_build_power_value(3)   # {0, .1, .2, .3, .4, .6, .8, 1} (8 magnitudes)

def _absmax_codebook(x, grid, bs):
    xb, K = _blocks(x, bs)
    s = xb.abs().amax(-1, keepdim=True).clamp_min(1e-12)
    return _unblock(round_codebook(xb / s, grid) * s, K)

def _signed_absmax(xb):
    """BOF4-S normalisation constant: the block element with largest |x| (keeps its sign), so
    that element maps to +1. Ties: lowest index (official uses torch .max(...).indices, whose
    tie order is unspecified). All-zero blocks get constant 1 (official would divide by 0)."""
    a = xb.abs()
    idx = (a == a.amax(-1, keepdim=True)).to(torch.uint8).argmax(-1, keepdim=True)
    s = xb.gather(-1, idx)
    return torch.where(s == 0, torch.ones_like(s), s)

def q_nf3(x, bs=64):   return _absmax_codebook(x, NF3_GRID, bs)
def q_af4(x, bs=64):   return _absmax_codebook(x, AF4_64_GRID, bs)
def q_bof4(x, bs=64):  return _absmax_codebook(x, BOF4_MSE_64_GRID, bs)

def q_bof4s(x, bs=64):
    xb, K = _blocks(x, bs)
    s = _signed_absmax(xb)
    return _unblock(round_codebook(xb / s, BOF4S_MSE_64_GRID) * s, K)

def q_fp4_absmax(x, bs=32):
    """Fixed-codebook FP4: OCP E2M1 grid {0,.5,1,1.5,2,3,4,6}, RNE, scale = group absmax / 6 in
    FP32 (no E8M0/E4M3 scale rounding, unlike MXFP4/NVFP4). Weight-only baseline."""
    xb, K = _blocks(x, bs)
    s = (xb.abs().amax(-1, keepdim=True) / FP4_MAX).clamp_min(1e-12)
    return _unblock(round_fp4_e2m1(xb / s) * s, K)

def q_apot4(x, bs=64):
    """APoT 4-bit weight levels (sign x build_power_value(B=3)) with per-group absmax as the
    clipping threshold alpha. Deviation: APoT is QAT with a learned per-layer alpha on
    standardised weights; here it is RTN PTQ with alpha = group absmax (no clipping).
    Nearest magnitude, ties -> lower magnitude (official: first index of unordered set)."""
    xb, K = _blocks(x, bs)
    s = xb.abs().amax(-1, keepdim=True).clamp_min(1e-12)
    return _unblock(torch.sign(xb) * round_codebook((xb / s).abs(), APOT4_W_MAG) * s, K)

# ---- batched, deterministic 1-D Lloyd-Max (weighted k-means) --------------------------------

def lloyd_1d(v, c, w=None, iters=100):
    """Weighted 1-D Lloyd-Max, batched over leading dim. v [G,n] values, c [G,L] sorted initial
    levels, w [G,n] non-negative weights (None = 1). Minimises sum w*(v - c[assign])^2.
    Assignment = nearest level, exact midpoint ties -> lower level (same rule as
    round_codebook / _assign). Centroids from float64 prefix sums over each group's sorted values
    (no atomics -> bit-deterministic on a given device). Empty cells keep their level. Stops
    when levels stop changing (1-D Lloyd converges in finitely many steps) or after iters."""
    G, n = v.shape
    vs, order = torch.sort(v.double(), dim=-1, stable=True)
    ws = torch.ones_like(vs) if w is None else w.double().gather(-1, order)
    zero = vs.new_zeros(G, 1)
    cw = torch.cat([zero, ws.cumsum(-1)], -1)
    cx = torch.cat([zero, (ws * vs).cumsum(-1)], -1)
    end = torch.full((G, 1), n, dtype=torch.long, device=v.device)
    c = c.double().contiguous()
    for _ in range(iters):
        mids = ((c[:, 1:] + c[:, :-1]) / 2).contiguous()
        b = torch.searchsorted(vs, mids, right=True)            # #values <= mid -> lower cell
        b = torch.cat([torch.zeros_like(end), b, end], -1)
        W = cw.gather(-1, b[:, 1:]) - cw.gather(-1, b[:, :-1])
        X = cx.gather(-1, b[:, 1:]) - cx.gather(-1, b[:, :-1])
        new = torch.where(W > 0, X / W.clamp_min(1e-300), c)
        new = new.sort(-1).values
        if torch.equal(new, c):
            break
        c = new
    return c.float()

def _assign(x, c):
    """Batched nearest-level rounding: x [G,n], c [G,L] sorted -> values. Ties -> lower level."""
    mids = ((c[:, 1:] + c[:, :-1]) / 2).contiguous()
    return c.gather(-1, torch.searchsorted(mids, x.contiguous(), right=False))

def _lut_groups(xg, mask):
    """Own 16-entry LUT per row of xg, Lloyd-Max from NF4*absmax init (so per-group MSE is never
    above NF4's), entries rounded to BF16 (the stored LUT format) before final assignment."""
    s = xg.abs().amax(-1, keepdim=True).clamp_min(1e-12)
    c = lloyd_1d(xg, NF4_GRID.to(xg.device) * s, mask)
    c = c.bfloat16().float().sort(-1).values
    return _assign(xg, c)

def q_lut4_lloyd(x, bs=64):
    """Per-group (bs) 16-entry MSE Lloyd-Max LUT in value space (no scale). Padding has weight 0.
    Storage: 4 bit/weight + 16 BF16 entries per group = 4 + 256/bs bits (8.0 at bs=64)."""
    xb, K = _blocks(x, bs)
    shp = xb.shape
    mask, _ = _blocks(torch.ones_like(x), bs)
    q = _lut_groups(xb.reshape(-1, bs), mask.reshape(-1, bs))
    return _unblock(q.reshape(shp), K)

def q_lut4_lloyd_row(x):
    """Per-row (output channel for W) 16-entry Lloyd-Max LUT: SqueezeLLM-style non-uniform
    per-channel LUT without its Fisher sensitivity weighting (plain MSE k-means)."""
    xr = x.reshape(-1, x.shape[-1])
    return _lut_groups(xr, None).reshape(x.shape)

def q_lut4_bof4fit(x, bs=64):
    """BOF4 empirical EM (arXiv:2505.06653 Sec. 3.2 / App. B.3, Eq. 64) on *this tensor*:
    absmax-normalise g64 blocks, fit one 16-entry codebook to the normalised values with
    centroid = sum m_b^2 x / sum m_b^2 (m_b = block absmax), i.e. minimise the MSE of the
    *de-normalised* weights. Init = official BOF4 (MSE, B=64) codebook; no level fixed (as in
    bof4_mse_64). Assignment is plain nearest-neighbour in normalised space, as in BOF4.
    Storage: 4.5 bpw (FP32 absmax per 64) + one 16-entry FP32 LUT per tensor."""
    xb, K = _blocks(x, bs)
    s = xb.abs().amax(-1, keepdim=True).clamp_min(1e-12)
    mask, _ = _blocks(torch.ones_like(x), bs)
    xn = (xb / s)
    w = (s * s).expand_as(xn) * mask
    c = lloyd_1d(xn.reshape(1, -1), BOF4_MSE_64_GRID.to(x.device).view(1, -1), w.reshape(1, -1))
    return _unblock(round_codebook(xn, c[0]) * s, K)

# ----------------------------------------------------------------------------- M2XFP (exact port)
# Port of mxq/quantize/quant_func.py from github.com/SJTU-ReArch-Group/M2XFP_ASPLOS26
# @ 2ed16eb6 (MIT). Arithmetic is kept op-for-op identical to the official code (same FP32
# ops, shapes, reduction dims, argmin tie rules) so results are bit-identical on the same
# device; see QuantizerTests.test_m2xfp_*_matches_official. The only intentional deviations:
#   * device-agnostic grids (official builds them on "cuda" at import time);
#   * MXEM top-1 tie break is explicit lowest-index (paper Alg. 1). Official uses torch.topk,
#     which gives lowest index on CUDA but NOT on CPU, so official-on-CPU can differ on ties.
#   * official flattens the whole tensor to (-1, 32); we group along the last dim, which is
#     identical whenever K % 32 == 0 (asserted, like the official code).

def _float_grid(exp_bit, man_bit):
    """Official float_value(): non-negative grid, bias 0, subnormals in the first binade."""
    vals, first, sub = [], True, True
    for i in range(2 ** exp_bit):
        for j in range(2 ** man_bit):
            if first:
                vals.append(0.0); first = False
            elif sub:
                vals.append((2 ** i) * (j * 2 ** (-man_bit)))
            else:
                vals.append((2 ** (i - 1)) * (1 + j * 2 ** (-man_bit)))
        sub = False
    return torch.tensor(vals)

M2X_FP4_GRID = _float_grid(2, 1)   # E2M1: 0 .5 1 1.5 2 3 4 6
M2X_FP6_GRID = _float_grid(2, 3)   # E2M3: 0 .. 7.5 step 1/8 in first binade (32 values)
M2X_SUBGROUP = 8

def _m2x_quantize_to_grid(x, levels):
    """Official quantize_to_grid: nearest level; exact ties on odd boundaries nudged up
    (+5e-7) so ties round to the even code; returns (values, indices)."""
    levels = levels.to(x.device)
    boundaries = (levels[:-1] + levels[1:]) / 2.0
    mask = torch.isin(x, boundaries[1::2])
    x = x + 0.0000005 * mask
    idx = torch.bucketize(x, boundaries).clamp_(0, len(levels) - 1)
    return levels[idx], idx

def _m2x_cast_fp4(x):
    v, _ = _m2x_quantize_to_grid(torch.abs(x), M2X_FP4_GRID)
    return v * torch.sign(x)

def _m2x_cast_fp4_em(x):
    """Official cast_to_fp4_em: FP4 value plus FP6(E2M3) value whose index is clamped to
    [4*i4-1, 4*i4+2] (2-bit metadata = FP6 low bits after +1 bias, clamped)."""
    sign, a = torch.sign(x), torch.abs(x)
    fp4, i4 = _m2x_quantize_to_grid(a, M2X_FP4_GRID)
    _, i6 = _m2x_quantize_to_grid(a, M2X_FP6_GRID)
    i6.clamp_(min=i4 * 4 - 1, max=i4 * 4 + 2)
    return fp4 * sign, M2X_FP6_GRID.to(x.device)[i6] * sign

def _m2x_groups(x, group_size):
    assert group_size % M2X_SUBGROUP == 0
    assert x.shape[-1] % group_size == 0, "M2XFP (official) requires K % group_size == 0"
    return x.float().reshape(-1, group_size)

def q_m2xfp_mxes(x, group_size=32):
    """M2XFP weight format (official get_quant_mxes). Per group of 32: shared exponent
    E = floor(log2 amax) - 2 plus a group bias b in {-1,0,1} (absorbed into E8M0, free);
    per subgroup of 8 a 2-bit mantissa ratio r in {1,1.25,1.5,1.75}. r chosen per subgroup
    by min MSE given b, then b chosen by min mean-subgroup MSE (first index on ties)."""
    shape, sg = x.shape, M2X_SUBGROUP
    t = _m2x_groups(x, group_size)
    max_val = t.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    t = t.reshape(-1, sg)
    exp = torch.floor(torch.log2(max_val)) - torch.floor(torch.log2(torch.tensor(FP4_MAX, device=t.device)))
    ratios = torch.tensor([1.0, 1.25, 1.5, 1.75], dtype=t.dtype, device=t.device)
    nsub = group_size // sg
    row = torch.arange(t.size(0), device=t.device)
    deqs, mses = [], []
    for bias in (-1, 0, 1):
        scales = torch.pow(2, exp + bias).expand(-1, nsub).reshape(-1, 1)
        xe = t.unsqueeze(2)
        cand_s = scales.unsqueeze(2) * ratios.view(1, 1, -1)
        cand_q = _m2x_cast_fp4(xe / cand_s) * cand_s
        mse = (cand_q - xe).pow(2).mean(dim=1)
        best = mse.argmin(dim=1)
        deqs.append(cand_q[row, :, best].reshape(-1, group_size))
        mses.append(mse[row, best].view(-1, nsub).mean(dim=1, keepdim=True))
    best_b = torch.cat(mses, dim=1).argmin(dim=1)
    all_deq = torch.stack(deqs, 0)
    out = torch.gather(all_deq, 0, best_b.view(1, -1, 1).expand(1, -1, group_size)).squeeze(0)
    return out.reshape(shape)

def q_m2xfp_mxem(x, group_size=32):
    """M2XFP activation format (official get_quant_mxem). OCP-floor shared scale
    2^(floor(log2 amax) - 2) per group of 32, all elements E2M1, except the top-1 |FP4|
    element of each subgroup of 8 (lowest index on ties) which gets the clamped FP6 value."""
    shape, sg = x.shape, M2X_SUBGROUP
    t = _m2x_groups(x, group_size)
    max_val = t.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    exp = torch.floor(torch.log2(max_val)) - torch.floor(torch.log2(torch.tensor(FP4_MAX, device=t.device)))
    scales = torch.pow(2, exp)
    fp4, fp6 = _m2x_cast_fp4_em(t / scales)
    a = fp4.reshape(-1, sg).abs()
    # lowest index among maxima (== official torch.topk(.,1) on CUDA; paper Alg. 1)
    top = (a == a.amax(dim=1, keepdim=True)).to(torch.uint8).argmax(dim=1, keepdim=True)
    mask = torch.zeros_like(a).scatter_(1, top, 1).reshape(-1, group_size)
    return ((fp4 * (1 - mask) + fp6 * mask) * scales).reshape(shape)

QUANT = {"int8": q_int8_row, "fp8": q_fp8_row, "mxfp4": q_mxfp4, "nvfp4": q_nvfp4, "nf4": q_nf4,
         "m2xfp_mxes": q_m2xfp_mxes, "m2xfp_mxem": q_m2xfp_mxem,
         "nf3": q_nf3, "fp4_g32": lambda x: q_fp4_absmax(x, 32), "fp4_g64": lambda x: q_fp4_absmax(x, 64),
         "af4": q_af4, "bof4": q_bof4, "bof4s": q_bof4s, "apot4": q_apot4,
         "lut4_bof4fit": q_lut4_bof4fit, "lut4_lloyd_g64": q_lut4_lloyd, "lut4_lloyd_row": q_lut4_lloyd_row}
# Nominal bits/weight incl. scale overhead: int8/fp8 ~8, mxfp4 4.25, nvfp4 4.5, nf4 g64 4.5 (fp32 absmax),
# m2xfp 4.5 (E8M0 scale + 8 bits metadata per 32: 4 x 2-bit subgroup codes)
# Weight bits/element for coverage reporting (scale overhead counted as FP32 per group, LUT
# entries as 16 bit; per-row LUT depends on K so it is computed per module in nominal_bpw).
W_BPW = {"int8": 8.0, "fp8": 8.0, "mxfp4": 4.25, "nvfp4": 4.5, "nf4": 4.5, "m2xfp_mxes": 4.5,
         "nf3": 3.5, "fp4_g32": 5.0, "fp4_g64": 4.5, "af4": 4.5, "bof4": 4.5,
         "bof4s": 4.5,   # + 1 sign bit/block if the scale format has no sign (paper Sec. 6)
         "apot4": 4.5, "lut4_bof4fit": 4.5, "lut4_lloyd_g64": 8.0, "lut4_lloyd_row": None}

def nominal_bpw(fmt, K):
    if fmt is None:
        return 16.0
    if fmt == "lut4_lloyd_row":
        return 4.0 + 16 * 16 / K
    return W_BPW[fmt]

# (weight_fmt, act_fmt) for ordinary linears and for GDN linears
MODES = {
    "bf16":               {"default": (None, None),       "gdn": (None, None)},
    "a8_int8":            {"default": (None, "int8"),     "gdn": (None, "int8")},
    "w8a8_int8":          {"default": ("int8", "int8"),   "gdn": ("int8", "int8")},
    "w8a8_fp8":           {"default": ("fp8", "fp8"),     "gdn": ("fp8", "fp8")},
    "w4a4_mxfp4":         {"default": ("mxfp4", "mxfp4"), "gdn": ("mxfp4", "mxfp4")},
    "w4a4_nvfp4":         {"default": ("nvfp4", "nvfp4"), "gdn": ("nvfp4", "nvfp4")},
    "w4_nf4_g64":         {"default": ("nf4", None),      "gdn": ("nf4", None)},
    "mixed_nvfp4_fp8gdn": {"default": ("nvfp4", "nvfp4"), "gdn": ("fp8", "fp8")},
    "w4a4_m2xfp":         {"default": ("m2xfp_mxes", "m2xfp_mxem"), "gdn": ("m2xfp_mxes", "m2xfp_mxem")},
}
LEGACY_MODES = list(MODES)   # default --modes (unchanged so old invocations reproduce)
# weight-only LUT / codebook family (activations BF16)
for _m, _f in [("w3_nf3_g64", "nf3"), ("w4_fp4_g32", "fp4_g32"), ("w4_fp4_g64", "fp4_g64"),
               ("w4_af4_g64", "af4"), ("w4_bof4_g64", "bof4"), ("w4_bof4s_g64", "bof4s"),
               ("w4_apot4_g64", "apot4"), ("w4_lut4_bof4fit_g64", "lut4_bof4fit"),
               ("w4_lut4_lloyd_g64", "lut4_lloyd_g64"), ("w4_lut4_lloyd_row", "lut4_lloyd_row")]:
    MODES[_m] = {"default": (_f, None), "gdn": (_f, None)}
LUT_MODES = [m for m in MODES if m not in LEGACY_MODES]

# ----------------------------------------------------------------------------- module replacement

class QLinear(nn.Module):
    """Drop-in nn.Linear replacement: fake-quant weight once, fake-quant input per call."""
    def __init__(self, lin: nn.Linear, wfmt, afmt, compute_dtype=None):
        super().__init__()
        self.in_features, self.out_features = lin.in_features, lin.out_features
        self.wfmt, self.afmt = wfmt, afmt
        self.compute_dtype = compute_dtype or lin.weight.dtype
        with torch.no_grad():
            w = lin.weight.detach().float()
            self.w_sq = w.double().pow(2).sum().item()
            if wfmt:
                wq = QUANT[wfmt](w)
                self.err_sq = (wq.double() - w.double()).pow(2).sum().item()
                w = wq
            else:
                self.err_sq = 0.0
            # own copy: never aliases (possibly tied) original weight
            self.register_buffer("weight", w.to(self.compute_dtype).clone(), persistent=False)
            b = lin.bias
            self.register_buffer("bias", None if b is None else b.detach().to(self.compute_dtype).clone(),
                                 persistent=False)
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        dt = x.dtype
        if self.afmt:
            x = QUANT[self.afmt](x.float())
        y = F.linear(x.to(self.compute_dtype), self.weight, self.bias)
        return y.to(dt)

    def extra_repr(self):
        return f"{self.in_features}->{self.out_features}, W={self.wfmt}, A={self.afmt}"

def is_gdn(name):
    return bool(GDN_RE.search(name))

def apply_mode(model, mode, exclude_gdn=False, include_lm_head=False, compute_dtype=None):
    """Replace linears in-place. Returns coverage dict."""
    spec = MODES[mode]
    total_params = sum(p.numel() for p in {id(p): p for p in model.parameters()}.values())
    linears = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    cov = {"total_model_params": total_params,
           "linear_weight_params": sum(m.weight.numel() for _, m in linears),
           "w_quant_params": 0, "a_quant_params": 0, "w_quant_modules": 0, "a_quant_modules": 0,
           "by_format": {}, "skipped": {}, "modules_by_format": {},
           "w_err_sq_by_format": {}, "w_sq_by_format": {}, "w_bits": 0.0}
    if spec["default"] == (None, None) and spec["gdn"] == (None, None):
        cov["skipped"]["all"] = len(linears)
        return _finish_cov(cov)
    for name, lin in linears:
        if name.endswith("lm_head") and not include_lm_head:
            cov["skipped"]["lm_head"] = cov["skipped"].get("lm_head", 0) + 1; continue
        gdn = is_gdn(name)
        if gdn and exclude_gdn:
            cov["skipped"]["gdn"] = cov["skipped"].get("gdn", 0) + 1; continue
        wfmt, afmt = spec["gdn" if gdn else "default"]
        if wfmt is None and afmt is None:
            continue
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        ql = QLinear(lin, wfmt, afmt, compute_dtype)
        setattr(parent, child, ql)
        n = lin.weight.numel()
        key = f"W={wfmt or 'bf16'},A={afmt or 'bf16'}"
        cov["by_format"][key] = cov["by_format"].get(key, 0) + n
        cov["modules_by_format"][key] = cov["modules_by_format"].get(key, 0) + 1
        cov["w_err_sq_by_format"][key] = cov["w_err_sq_by_format"].get(key, 0.0) + ql.err_sq
        cov["w_sq_by_format"][key] = cov["w_sq_by_format"].get(key, 0.0) + ql.w_sq
        cov["w_bits"] += n * nominal_bpw(wfmt, lin.in_features)
        if wfmt: cov["w_quant_params"] += n; cov["w_quant_modules"] += 1
        if afmt: cov["a_quant_params"] += n; cov["a_quant_modules"] += 1
    return _finish_cov(cov)

def _finish_cov(c):
    c["w_quant_frac_of_linear"] = c["w_quant_params"] / max(c["linear_weight_params"], 1)
    c["w_quant_frac_of_model"] = c["w_quant_params"] / max(c["total_model_params"], 1)
    c["a_quant_frac_of_linear"] = c["a_quant_params"] / max(c["linear_weight_params"], 1)
    if "w_bits" in c:
        # nominal bits/weight over the *replaced* linears (BF16 = 16 for A-only replacements)
        nrep = sum(c["by_format"].values())
        c["w_bpw_replaced"] = c.pop("w_bits") / nrep if nrep else None
        # relative weight MSE ||W - Q(W)||^2 / ||W||^2 per format
        c["w_rel_err_by_format"] = {k: c["w_err_sq_by_format"][k] / max(c["w_sq_by_format"][k], 1e-30)
                                    for k in c["w_err_sq_by_format"]}
        del c["w_err_sq_by_format"], c["w_sq_by_format"]
    return c

# ----------------------------------------------------------------------------- evaluation

_TEXT = None
def eval_ppl(model, tok, tokens=4096, seqlen=512, device="cuda"):
    global _TEXT
    if _TEXT is None:
        _TEXT = urllib.request.urlopen(URL).read().decode()
    ids = tok(_TEXT, return_tensors="pt", add_special_tokens=False).input_ids[0][:tokens + 1]
    total, n = 0.0, 0
    with torch.inference_mode():
        for s in range(0, len(ids) - 1, seqlen):
            x = ids[s:min(s + seqlen, len(ids) - 1)].unsqueeze(0).to(device)
            y = ids[s + 1:min(s + seqlen + 1, len(ids))].to(device)
            out = model(x, use_cache=False, logits_to_keep=0).logits[0].float()
            if not torch.isfinite(out).all():
                raise RuntimeError("non-finite logits")
            total += F.cross_entropy(out, y, reduction="sum").item(); n += y.numel()
    return total / n, math.exp(total / n), n

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modes", nargs="+", default=LEGACY_MODES, choices=list(MODES) + ["lut"],
                    help="'lut' expands to the weight-only LUT/codebook family incl. bf16 and w4_nf4_g64")
    ap.add_argument("--tokens", type=int, default=4096)
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--exclude-gdn", action="store_true", help="leave linear_attn.* (GDN) linears BF16")
    ap.add_argument("--include-lm-head", action="store_true",
                    help="also quantize lm_head (own copy; tied embedding stays BF16)")
    ap.add_argument("--compute-dtype", choices=["bf16", "fp32"], default="bf16",
                    help="dtype of dequantized W/A fed to the matmul in QLinear")
    ap.add_argument("--output", default="advanced_ppl_results.json")
    ap.add_argument("--self-test", action="store_true", help="run quantizer unit tests (CPU) and exit")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(run_self_tests())
    modes = []
    for m in a.modes:
        for x in (["bf16", "w4_nf4_g64"] + LUT_MODES if m == "lut" else [m]):
            if x not in modes:
                modes.append(x)
    a.modes = modes

    from transformers import AutoModelForCausalLM, AutoTokenizer
    cdt = {"bf16": torch.bfloat16, "fp32": torch.float32}[a.compute_dtype]
    tok = AutoTokenizer.from_pretrained(MODEL)
    res = {"model": MODEL, "dataset": URL,
           "protocol": f"first {a.tokens+1} tokenizer tokens, independent {a.seqlen}-token windows, "
                       f"{a.tokens} next-token predictions, FP32 cross entropy",
           "options": {"exclude_gdn": a.exclude_gdn, "include_lm_head": a.include_lm_head,
                       "compute_dtype": a.compute_dtype},
           "results": {}}
    for mode in a.modes:
        torch.cuda.empty_cache(); t = time.time()
        m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
        cov = apply_mode(m, mode, a.exclude_gdn, a.include_lm_head, cdt)
        try:
            nll, ppl, n = eval_ppl(m, tok, a.tokens, a.seqlen)
            r = {"nll": nll, "ppl": ppl, "tokens": n}
        except RuntimeError as e:
            r = {"nll": None, "ppl": None, "error": str(e)}
        calls = [q.calls for q in m.modules() if isinstance(q, QLinear)]
        r.update(spec={k: list(v) for k, v in MODES[mode].items()}, coverage=cov,
                 qlinear_modules=len(calls), qlinear_min_calls=min(calls) if calls else 0,
                 seconds=time.time() - t)
        res["results"][mode] = r
        print(f"{mode:20s} ppl={r['ppl']}  Wq={cov['w_quant_frac_of_linear']:.1%} of linear "
              f"({cov['w_quant_frac_of_model']:.1%} of model)  Aq={cov['a_quant_frac_of_linear']:.1%}  "
              f"bpw={cov.get('w_bpw_replaced')}  relerr={cov.get('w_rel_err_by_format')}", flush=True)
        del m
        open(a.output, "w").write(json.dumps(res, indent=2) + "\n")

# ----------------------------------------------------------------------------- self tests

def run_self_tests():
    import unittest
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(QuantizerTests)
    return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1

import unittest
class QuantizerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_fp4_grid_and_ties(self):
        x = torch.tensor([0.0, 0.2, 0.25, 0.26, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 5.1, 7.0, 100.0, -0.75, -5.0, -9.0])
        exp = torch.tensor([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 6.0, 6.0, 6.0, -1.0, -4.0, -6.0])
        self.assertTrue(torch.equal(round_fp4_e2m1(x), exp))
        r = round_fp4_e2m1(torch.randn(10000) * 4).abs()
        self.assertTrue(torch.isin(r, FP4_GRID).all())
        # grid points are fixed points
        self.assertTrue(torch.equal(round_fp4_e2m1(FP4_GRID), FP4_GRID))

    def test_fp8_grid(self):
        x = torch.randn(10000) * 100
        r = round_fp8_e4m3(x)
        self.assertTrue(torch.equal(r.to(torch.float8_e4m3fn).float(), r))
        self.assertEqual(round_fp8_e4m3(torch.tensor([1e4])).item(), 448.0)
        self.assertEqual(round_fp8_e4m3(torch.tensor([17.0])).item(), 16.0)  # 3-bit mantissa, tie->even
        q = q_fp8_row(torch.randn(8, 64))
        self.assertTrue(torch.isfinite(q).all())

    def test_int8_row(self):
        x = torch.randn(16, 128)
        q = q_int8_row(x)
        s = x.abs().amax(-1, keepdim=True) / 127
        codes = q / s
        self.assertTrue(torch.allclose(codes, codes.round(), atol=1e-4))
        self.assertLessEqual(codes.abs().max().item(), 127 + 1e-4)
        self.assertTrue(((q - x).abs() <= s / 2 + 1e-6).all())

    def test_mxfp4(self):
        x = torch.randn(4, 96) * torch.logspace(-3, 3, 96)
        q = q_mxfp4(x, 32)
        xb, _ = _blocks(x, 32); qb, _ = _blocks(q, 32)
        s = torch.exp2(mx_scale_exp(xb.abs().amax(-1, keepdim=True)))
        self.assertTrue(torch.equal(torch.log2(s), torch.log2(s).round()))     # power of two
        self.assertTrue(torch.isin((qb / s).abs(), FP4_GRID).all())
        # block max lands in [4, 8) before rounding -> saturates to 6 at most
        self.assertLessEqual((qb / s).abs().max().item(), 6.0)
        r = (xb.abs().amax(-1) / s.squeeze(-1))
        self.assertTrue(((r >= 4) & (r < 8)).all())
        self.assertTrue(torch.equal(q_mxfp4(torch.zeros(2, 32)), torch.zeros(2, 32)))

    def test_nvfp4(self):
        x = torch.randn(4, 64) * torch.logspace(-2, 2, 64)
        q = q_nvfp4(x, 16)
        xb, _ = _blocks(x, 16); qb, _ = _blocks(q, 16)
        sb, g = nvfp4_scales(xb, x.abs().amax())
        self.assertTrue(torch.equal(sb.to(torch.float8_e4m3fn).float(), sb))  # E4M3 block scales
        self.assertLessEqual(sb.max().item(), 448.0)
        codes = torch.where(sb > 0, qb / (sb * g), torch.zeros_like(qb))
        self.assertTrue(torch.allclose(codes.abs(), round_fp4_e2m1(codes).abs(), atol=1e-5))
        self.assertLess(((q - x).norm() / x.norm()).item(), 0.15)
        self.assertTrue(torch.equal(q_nvfp4(torch.zeros(1, 16)), torch.zeros(1, 16)))

    def test_nf4(self):
        self.assertEqual(len(NF4_GRID), 16)
        x = torch.randn(8, 128)
        q = q_nf4(x, 64)
        xb, _ = _blocks(x, 64); qb, _ = _blocks(q, 64)
        s = xb.abs().amax(-1, keepdim=True)
        codes = qb / s
        nearest = (codes.unsqueeze(-1) - NF4_GRID).abs().min(-1).values
        self.assertLess(nearest.max().item(), 1e-6)  # every code is an NF4 grid value
        self.assertTrue(torch.equal(round_codebook(NF4_GRID, NF4_GRID), NF4_GRID))
        # absmax element maps to +-1 exactly
        self.assertTrue(torch.allclose((qb / s).abs().amax(-1), torch.ones(8, 2)))

    def test_padding(self):
        x = torch.randn(3, 40)
        for f in (q_mxfp4, q_nvfp4, q_nf4):
            self.assertEqual(f(x).shape, x.shape)

    def test_qlinear_applies_activation_quant(self):
        lin = nn.Linear(64, 8, bias=True)
        x = torch.randn(5, 64)
        for wf, af in [(None, "int8"), ("fp8", "fp8"), ("mxfp4", "mxfp4"), ("nvfp4", "nvfp4"), ("nf4", None)]:
            ql = QLinear(lin, wf, af, torch.float32)
            w = QUANT[wf](lin.weight.float()) if wf else lin.weight.float()
            xa = QUANT[af](x) if af else x
            self.assertTrue(torch.allclose(ql(x), F.linear(xa, w, lin.bias), atol=1e-5), (wf, af))
            if af:
                self.assertFalse(torch.allclose(ql(x), F.linear(x, w, lin.bias), atol=1e-6))
        self.assertEqual(ql.calls, 1)
        self.assertFalse(ql.weight.data_ptr() == lin.weight.data_ptr())

    def test_apply_mode_coverage_and_gdn(self):
        class Blk(nn.Module):
            def __init__(s):
                super().__init__()
                s.linear_attn = nn.Module(); s.linear_attn.in_proj_qkv = nn.Linear(32, 96, bias=False)
                s.mlp = nn.Module(); s.mlp.up_proj = nn.Linear(32, 64, bias=False)
        class M(nn.Module):
            def __init__(s):
                super().__init__()
                s.layers = nn.ModuleList([Blk(), Blk()]); s.lm_head = nn.Linear(32, 100, bias=False)
        m = M(); cov = apply_mode(m, "mixed_nvfp4_fp8gdn")
        self.assertEqual(m.layers[0].linear_attn.in_proj_qkv.wfmt, "fp8")
        self.assertEqual(m.layers[0].mlp.up_proj.wfmt, "nvfp4")
        self.assertIsInstance(m.lm_head, nn.Linear)
        self.assertEqual(cov["w_quant_params"], 2 * (32 * 96 + 32 * 64))
        m = M(); cov = apply_mode(m, "w4a4_mxfp4", exclude_gdn=True)
        self.assertIsInstance(m.layers[1].linear_attn.in_proj_qkv, nn.Linear)
        self.assertEqual(cov["w_quant_params"], 2 * 32 * 64)
        self.assertEqual(cov["skipped"]["gdn"], 2)
        m = M(); cov = apply_mode(m, "w4_nf4_g64", include_lm_head=True)
        self.assertEqual(cov["a_quant_params"], 0)
        self.assertEqual(cov["w_quant_frac_of_linear"], 1.0)

    # ---------------------------------------------------------------- M2XFP vs official oracle
    @staticmethod
    def _official():
        """Load the verbatim official quant_func.py (third_party/). It allocates its grids on
        'cuda' at import, so it needs a CUDA build+device."""
        if not torch.cuda.is_available():
            raise unittest.SkipTest("official quant_func.py builds grids on cuda at import")
        import importlib.util, pathlib
        p = pathlib.Path(__file__).resolve().parent / "third_party" / "m2xfp_quant_func.py"
        spec = importlib.util.spec_from_file_location("m2xfp_official", p)
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        return mod

    @staticmethod
    def _m2x_inputs(dev):
        g = torch.Generator(device="cpu").manual_seed(1)
        xs = [torch.randn(4, 64, generator=g),
              torch.randn(3, 96, generator=g) * torch.logspace(-4, 4, 96),   # wide dynamic range
              torch.randn(2, 5, 128, generator=g) * 0.02,                     # 3-D activation shape
              torch.randn(6, 32, generator=g).pow(3),                         # heavy tails
              torch.zeros(2, 32),                                             # all-zero groups
              # exact FP4 grid values / boundaries x powers of two: many ties
              (M2X_FP4_GRID[torch.randint(0, 8, (8, 64), generator=g)]
               * torch.randint(0, 2, (8, 64), generator=g).mul(2).sub(1)
               * 2.0 ** torch.randint(-3, 4, (8, 1), generator=g)),
              torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0] * 4).repeat(2, 1)]
        xs.append(torch.randn(4, 64, generator=g).to(torch.bfloat16).float())
        return [x.to(dev) for x in xs]

    def test_m2xfp_grids_match_official_definition(self):
        self.assertTrue(torch.equal(M2X_FP4_GRID, FP4_GRID))
        self.assertEqual(len(M2X_FP6_GRID), 32)
        self.assertTrue(torch.equal(M2X_FP6_GRID[:8], torch.arange(8) / 8))
        self.assertEqual(M2X_FP6_GRID[-1].item(), 7.5)
        # E2M1 rounding of official helper == this harness's RNE E2M1 rounding
        x = torch.randn(20000) * 4
        x = torch.cat([x, torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.25, -2.5, -5.0])])
        self.assertTrue(torch.equal(_m2x_cast_fp4(x), round_fp4_e2m1(x)))

    def test_m2xfp_mxes_matches_official(self):
        off = self._official()
        for x in self._m2x_inputs("cuda"):
            ref = off.get_quant_mxes(x, 32)
            self.assertTrue(torch.equal(q_m2xfp_mxes(x, 32), ref), tuple(x.shape))
        # half-precision input path used by the official harness (fp16 model)
        w = torch.randn(16, 256, device="cuda").half()
        self.assertTrue(torch.equal(q_m2xfp_mxes(w, 32).half(), off.get_quant_mxes(w, 32)))

    def test_m2xfp_mxem_matches_official(self):
        off = self._official()
        for x in self._m2x_inputs("cuda"):
            ref = off.get_quant_mxem(x, 32)
            self.assertTrue(torch.equal(q_m2xfp_mxem(x, 32), ref), tuple(x.shape))
        a = torch.randn(7, 1024, device="cuda").bfloat16()
        self.assertTrue(torch.equal(q_m2xfp_mxem(a, 32).bfloat16(), off.get_quant_mxem(a, 32)))

    def test_m2xfp_cpu_matches_official_on_cpu(self):
        """Run the official functions on CPU (grids moved) - MXES must be identical; MXEM may
        differ only where CPU torch.topk breaks FP4 ties at a non-lowest index."""
        off = self._official()
        off.FP4_E2M1_GRID, off.FP6_E2M3_GRID = off.FP4_E2M1_GRID.cpu(), off.FP6_E2M3_GRID.cpu()
        for x in self._m2x_inputs("cpu"):
            self.assertTrue(torch.equal(q_m2xfp_mxes(x, 32), off.get_quant_mxes(x, 32)))
            ours, ref = q_m2xfp_mxem(x, 32), off.get_quant_mxem(x, 32)
            diff = (ours != ref).reshape(-1, 8).any(1)
            fp4 = _m2x_cast_fp4(_blocks(x, 32)[0] / torch.exp2(mx_scale_exp(
                _blocks(x, 32)[0].abs().amax(-1, keepdim=True).clamp(min=1e-8)))).reshape(-1, 8).abs()
            tied = ((fp4 == fp4.amax(1, keepdim=True)).sum(1) > 1)
            self.assertFalse((diff & ~tied).any(), tuple(x.shape))

    def test_m2xfp_handworked(self):
        # MXEM: group amax 5.2 -> E = floor(log2 5.2) - 2 = 0, scale 1. Subgroup 0 top-1 is
        # 5.2 (> 5 -> FP4 6, index 7), so its FP6 index is clamped to [27, 30]:
        # nearest FP6 to 5.2 is 5.0 (index 26) -> clamped to 27 -> 5.5. Others FP4.
        x = torch.tensor([[5.2, 0.3, -1.1, 2.6, 0.0, 0.1, 0.7, -3.4] + [0.9] * 8 + [0.0] * 16])
        q = q_m2xfp_mxem(x, 32)
        self.assertEqual(q[0, :8].tolist(), [5.5, 0.5, -1.0, 3.0, 0.0, 0.0, 0.5, -3.0])
        # subgroup 1: all FP4 = 1.0, lowest index gets FP6(0.9)=0.875 (idx 7, clamp [7,10])
        self.assertEqual(q[0, 8:16].tolist(), [0.875] + [1.0] * 7)
        # all-zero subgroup: FP6 of 0 is 0
        self.assertEqual(q[0, 16:].abs().sum().item(), 0.0)
        # MXES on a group whose values are exactly 1.25 x FP4 grid: recovered losslessly
        w = (FP4_GRID.repeat(4) * 1.25 * 0.5).view(1, 32)
        self.assertTrue(torch.equal(q_m2xfp_mxes(w, 32), w))
        # never worse than MXFP4 (b=0, r=1 is in the search space) per group in MSE
        w = torch.randn(64, 256)
        e_m2 = (q_m2xfp_mxes(w) - w).pow(2).reshape(-1, 32).mean(1)
        e_mx = (q_mxfp4(w) - w).pow(2).reshape(-1, 32).mean(1)
        self.assertTrue((e_m2 <= e_mx + 1e-12).all())
        self.assertRaises(AssertionError, q_m2xfp_mxes, torch.randn(2, 40))

    # ---------------------------------------------------------------- LUT / codebook family
    def test_lut_exact_grids(self):
        # NF: torch port of LoftQ create_normal_map reproduces NF4 (bnb) and NF3 bit-exactly
        self.assertTrue(torch.equal(_create_normal_map(num_bits=4), NF4_GRID))
        self.assertTrue(torch.equal(_create_normal_map(num_bits=3), NF3_GRID))
        self.assertEqual(len(NF3_GRID), 8)
        for g, n in [(AF4_64_GRID, 16), (BOF4_MSE_64_GRID, 16), (BOF4S_MSE_64_GRID, 16)]:
            self.assertEqual(len(g), n)
            self.assertTrue((g[1:] > g[:-1]).all())
        # fixed points per source: AF4 {-1,0,1}; BOF4-S {0,1}; BOF4 (MSE) none, symmetric-ish
        for v in (-1.0, 0.0, 1.0):
            self.assertIn(v, AF4_64_GRID.tolist())
        self.assertEqual(BOF4S_MSE_64_GRID[7].item(), 0.0); self.assertEqual(BOF4S_MSE_64_GRID[-1].item(), 1.0)
        self.assertNotIn(0.0, BOF4_MSE_64_GRID.tolist())
        self.assertLess((BOF4_MSE_64_GRID + BOF4_MSE_64_GRID.flip(0)).abs().max().item(), 1e-4)
        self.assertAlmostEqual(BOF4S_MSE_64_GRID[0].item(), -0.8568463921546936, places=7)
        # APoT 4-bit weight: B=3 -> {0, 2^-1, 2^-2, 2^-3+2^-4 ...} normalised by max (=0.625)
        exp = torch.tensor([0, 1/16, 1/8, 3/16, 1/4, 3/8, 1/2, 5/8]) / (5/8)
        self.assertTrue(torch.allclose(APOT4_W_MAG, exp, atol=1e-7))
        self.assertEqual(len(APOT4_W_MAG), 8)  # 8 magnitudes x sign = 15 distinct signed levels
        self.assertEqual(len(_apot_build_power_value(2)), 4)
        self.assertTrue(torch.equal(_apot_build_power_value(2), torch.tensor([0, .25, .5, 1.])))
        self.assertEqual(len(_apot_build_power_value(4)), 16)

    def test_lut_nearest_code_assignment(self):
        cases = [(q_nf3, NF3_GRID), (q_af4, AF4_64_GRID), (q_bof4, BOF4_MSE_64_GRID)]
        x = torch.randn(16, 256) * torch.logspace(-2, 1, 256)
        for f, g in cases:
            q = f(x, 64)
            xb, _ = _blocks(x, 64); qb, _ = _blocks(q, 64)
            s = xb.abs().amax(-1, keepdim=True)
            d = ((xb / s).unsqueeze(-1) - g).abs()
            ref = g[d.argmin(-1)] * s                         # brute-force nearest
            self.assertTrue(torch.allclose(qb, ref, atol=1e-6), f.__name__)
        # exact midpoint ties go to the lower level
        for g in (NF3_GRID, AF4_64_GRID, BOF4S_MSE_64_GRID):
            mids = (g[1:] + g[:-1]) / 2
            self.assertTrue(torch.equal(round_codebook(mids, g), g[:-1]))
            self.assertTrue(torch.equal(round_codebook(g, g), g))
        # BOF4-S: largest-|x| element maps to exactly +1 * signed max (incl. negative maxima)
        w = torch.tensor([[0.1, -2.0, 0.5, 1.9] + [0.0] * 60])
        q = q_bof4s(w, 64)
        self.assertEqual(q[0, 1].item(), -2.0)
        self.assertAlmostEqual(q[0, 3].item(), -2.0 * BOF4S_MSE_64_GRID[0].item(), places=5)
        self.assertTrue(torch.equal(q_bof4s(torch.zeros(2, 64)), torch.zeros(2, 64)))
        xb, _ = _blocks(x, 64); s = _signed_absmax(xb)
        self.assertTrue(torch.allclose(_blocks(q_bof4s(x), 64)[0].gather(-1, (xb.abs()).argmax(-1, keepdim=True)), s))
        # APoT: sign x nearest magnitude, absmax element exact
        q = q_apot4(x, 64); qb, _ = _blocks(q, 64); sb = xb.abs().amax(-1, keepdim=True)
        mag = (qb / sb).abs()
        self.assertLess((mag.unsqueeze(-1) - APOT4_W_MAG).abs().min(-1).values.max().item(), 1e-6)
        self.assertTrue(torch.equal(torch.sign(qb)[qb != 0], torch.sign(xb)[qb != 0]))
        ref = torch.sign(xb) * APOT4_W_MAG[((xb / sb).abs().unsqueeze(-1) - APOT4_W_MAG).abs().argmin(-1)] * sb
        self.assertTrue(torch.allclose(qb, ref, atol=1e-6))
        # FP4 absmax: codes on E2M1 grid, group max -> exactly 6, scale is NOT power-of-two
        for bs in (32, 64):
            q = q_fp4_absmax(x, bs); xb2, _ = _blocks(x, bs); qb2, _ = _blocks(q, bs)
            s = xb2.abs().amax(-1, keepdim=True) / 6
            self.assertTrue(torch.allclose((qb2 / s).abs(), round_fp4_e2m1(xb2 / s).abs(), atol=1e-5))
            self.assertTrue(torch.allclose((qb2 / s).abs().amax(-1), torch.full((16, 256 // bs), 6.0)))
        # handworked FP4 g32: absmax 3 -> scale 0.5; 0.6/0.5=1.2 -> 1.0 -> 0.5; 1.25/.5=2.5 tie->2 -> 1.0
        w = torch.tensor([[3.0, 0.6, 1.25, -0.125] + [0.0] * 28])
        self.assertEqual(q_fp4_absmax(w, 32)[0, :4].tolist(), [3.0, 0.5, 1.0, -0.0])

    def test_lloyd_1d(self):
        # textbook case: two well-separated clusters, 2 levels -> cluster means, 1 Lloyd step
        v = torch.tensor([[0., 1., 2., 10., 11., 12.]])
        c = lloyd_1d(v, torch.tensor([[0., 12.]]))
        self.assertTrue(torch.equal(c, torch.tensor([[1., 11.]])))
        # weights: centroid is the weighted mean (BOF4 Eq. 64)
        c = lloyd_1d(torch.tensor([[0., 1., 10.]]), torch.tensor([[0., 10.]]), torch.tensor([[1., 3., 1.]]))
        self.assertAlmostEqual(c[0, 0].item(), 0.75, places=6); self.assertEqual(c[0, 1].item(), 10.)
        # midpoint tie (v=5 between levels 0,10) goes to the lower cell, matching _assign
        c = lloyd_1d(torch.tensor([[0., 5., 10.]]), torch.tensor([[0., 10.]]), iters=1)
        self.assertEqual(c.tolist(), [[2.5, 10.]])
        self.assertEqual(_assign(torch.tensor([[5.]]), torch.tensor([[0., 10.]])).item(), 0.)
        # empty cell keeps its level; batched groups are independent
        c = lloyd_1d(torch.tensor([[0., 0.1], [5., 6.]]), torch.tensor([[0., 100.], [5., 6.]]))
        self.assertTrue(torch.allclose(c, torch.tensor([[0.05, 100.], [5., 6.]])))
        # fixed point satisfies both Lloyd conditions; never worse than its init (NF4*absmax)
        x = torch.randn(64, 64)
        s = x.abs().amax(-1, keepdim=True)
        c = lloyd_1d(x, NF4_GRID * s)
        q = _assign(x, c)
        self.assertTrue(((q - x).pow(2).sum(-1) <= (_assign(x, NF4_GRID * s) - x).pow(2).sum(-1) + 1e-6).all())
        a = torch.searchsorted(((c[:, 1:] + c[:, :-1]) / 2).contiguous(), x)
        for g in range(4):
            for l in a[g].unique():
                self.assertAlmostEqual(x[g][a[g] == l].mean().item(), c[g, l].item(), places=5)
        # deterministic
        self.assertTrue(torch.equal(lloyd_1d(x, NF4_GRID * s), c))

    def test_lut4_quantizers(self):
        x = torch.randn(32, 200) * torch.logspace(-1, 1, 200)   # K % 64 != 0 -> padding path
        for f in (q_lut4_lloyd, q_lut4_lloyd_row, q_lut4_bof4fit):
            q = f(x)
            self.assertEqual(q.shape, x.shape)
            self.assertTrue(torch.equal(f(x), q), f.__name__)          # deterministic
        # per-group LUT: <= 16 distinct BF16 values per group, beats NF4 g64 in MSE
        q = q_lut4_lloyd(x, 64); qb, _ = _blocks(q, 64)
        for g in qb.reshape(-1, 64)[:50]:
            self.assertLessEqual(len(g.unique()), 16)
        self.assertTrue(torch.equal(q.bfloat16().float(), q))
        e = lambda f: (f(x) - x).pow(2).sum().item()
        self.assertLess(e(q_lut4_lloyd), e(q_nf4))
        # per-row LUT: <= 16 distinct values per row
        q = q_lut4_lloyd_row(x)
        self.assertTrue(all(len(r.unique()) <= 16 for r in q))
        # BOF4 fit: one shared 16-entry codebook in normalised space, MSE <= official BOF4 init
        q = q_lut4_bof4fit(x); xb, _ = _blocks(x, 64); qb, _ = _blocks(q, 64)
        s = xb.abs().amax(-1, keepdim=True)
        mask = _blocks(torch.ones_like(x), 64)[0].bool()
        self.assertLessEqual(len((qb / s)[mask].round(decimals=5).unique()), 16)
        self.assertLessEqual(e(q_lut4_bof4fit), e(q_bof4) * (1 + 1e-6))
        # on Gaussian data, BOF4 beats NF4 and AF4 in weight MSE at B=64 (paper's claim)
        g = torch.randn(512, 1024)
        eg = lambda f: (f(g) - g).pow(2).sum().item()
        self.assertLess(eg(q_bof4), eg(q_nf4)); self.assertLess(eg(q_bof4), eg(q_af4))
        self.assertLess(eg(q_bof4s), eg(q_bof4))
        self.assertLess(eg(q_nf4), eg(q_nf3))

    def test_lut_modes_coverage(self):
        lin = nn.Linear(128, 16, bias=False)
        for m in LUT_MODES:
            wf, af = MODES[m]["default"]
            self.assertIsNone(af, m)                                  # weight-only, BF16 activations
            ql = QLinear(lin, wf, af, torch.float32)
            x = torch.randn(3, 128)
            self.assertTrue(torch.allclose(ql(x), F.linear(x, QUANT[wf](lin.weight.float())), atol=1e-5), m)
        class M(nn.Module):
            def __init__(s):
                super().__init__()
                s.linear_attn = nn.Module(); s.linear_attn.in_proj_qkv = nn.Linear(128, 64, bias=False)
                s.up = nn.Linear(128, 32, bias=False); s.lm_head = nn.Linear(128, 10, bias=False)
        cov = apply_mode(M(), "w3_nf3_g64")
        self.assertEqual(cov["w_quant_params"], 128 * 96); self.assertEqual(cov["a_quant_params"], 0)
        self.assertEqual(cov["modules_by_format"], {"W=nf3,A=bf16": 2})
        self.assertEqual(cov["w_bpw_replaced"], 3.5)
        self.assertGreater(cov["w_rel_err_by_format"]["W=nf3,A=bf16"], 0)
        cov = apply_mode(M(), "w4_lut4_lloyd_row", exclude_gdn=True)
        self.assertEqual(cov["w_bpw_replaced"], 4 + 256 / 128); self.assertEqual(cov["skipped"]["gdn"], 1)
        cov = apply_mode(M(), "bf16")
        self.assertIsNone(cov["w_bpw_replaced"])
        self.assertNotIn("lut", MODES); self.assertEqual(LEGACY_MODES[-1], "w4a4_m2xfp")

    def test_m2xfp_mode(self):
        lin = nn.Linear(64, 8, bias=True)
        x = torch.randn(5, 64)
        ql = QLinear(lin, *MODES["w4a4_m2xfp"]["default"], compute_dtype=torch.float32)
        ref = F.linear(q_m2xfp_mxem(x), q_m2xfp_mxes(lin.weight.float()), lin.bias)
        self.assertTrue(torch.allclose(ql(x), ref, atol=1e-5))
        # dequantized values are exactly representable in bf16 (<= 7 significant bits)
        w = q_m2xfp_mxes(torch.randn(32, 256)); a = q_m2xfp_mxem(torch.randn(32, 256))
        self.assertTrue(torch.equal(w.bfloat16().float(), w))
        self.assertTrue(torch.equal(a.bfloat16().float(), a))

if __name__ == "__main__":
    main()
