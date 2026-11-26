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
         "m2xfp_mxes": q_m2xfp_mxes, "m2xfp_mxem": q_m2xfp_mxem}
# Nominal bits/weight incl. scale overhead: int8/fp8 ~8, mxfp4 4.25, nvfp4 4.5, nf4 g64 4.5 (fp32 absmax),
# m2xfp 4.5 (E8M0 scale + 8 bits metadata per 32: 4 x 2-bit subgroup codes)

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
            if wfmt:
                w = QUANT[wfmt](w)
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
           "by_format": {}, "skipped": {}}
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
        setattr(parent, child, QLinear(lin, wfmt, afmt, compute_dtype))
        n = lin.weight.numel()
        key = f"W={wfmt or 'bf16'},A={afmt or 'bf16'}"
        cov["by_format"][key] = cov["by_format"].get(key, 0) + n
        if wfmt: cov["w_quant_params"] += n; cov["w_quant_modules"] += 1
        if afmt: cov["a_quant_params"] += n; cov["a_quant_modules"] += 1
    return _finish_cov(cov)

def _finish_cov(c):
    c["w_quant_frac_of_linear"] = c["w_quant_params"] / max(c["linear_weight_params"], 1)
    c["w_quant_frac_of_model"] = c["w_quant_params"] / max(c["total_model_params"], 1)
    c["a_quant_frac_of_linear"] = c["a_quant_params"] / max(c["linear_weight_params"], 1)
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
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=list(MODES))
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
              f"({cov['w_quant_frac_of_model']:.1%} of model)  Aq={cov['a_quant_frac_of_linear']:.1%}", flush=True)
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
