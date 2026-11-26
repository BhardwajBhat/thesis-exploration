"""Faithful port of the IST-DASLab/FP-Quant MR-GPTQ numerics (quantizer + GPTQ + Hadamard).

Upstream: https://github.com/IST-DASLab/FP-Quant  commit d2e3092f968262c4de5fb050e1aef568a280dadd
  src/quantization/quant_ops.py   -> cast_to_fp4, cast_to_eBm0, constants
  src/quantization/quantizer.py   -> Quantizer.get_quantization_params / __call__
  src/quantization/gptq.py        -> GPTQ.update / step / _get_hessian_inverse
  src/transforms/transforms.py    -> HadamardTransform (block-diagonal, Sylvester order)
Paper: Egiazarian et al., "Bridging the Gap Between Promise and Performance for Microscaling
FP4 Quantization", arXiv:2509.23202 (MR-GPTQ).

Deliberate deviations (numerically equivalent or strictly more precise):
  * Hadamard is an explicit Sylvester matrix matmul in fp32 instead of the
    fast_hadamard_transform CUDA extension (same ordering, same 1/sqrt(g) scale).
  * Hessian accumulation uses torch addmm instead of the Triton kernel.
  * Only the 4-bit FP formats (nvfp / mxfp) and int are ported; no wandb / export packing here.
Code is kept structurally close to upstream so diffs are easy to audit; see
test_mr_gptq.py::UpstreamParity for a direct numeric comparison against the clone.
"""
import math
from typing import Optional

import torch
import torch.nn as nn

FP4_E2M1_MAX = 6.0
FP8_E4M3_MAX = 448.0
NVFP_GROUPSIZE = 16
MXFP_GROUPSIZE = 32
FP32_MIN_NORMAL = 2 ** (-127 + 1)
FP4_SCALE = 3 / 4
FP4_GRID = [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
FP4_BITPACKING_PERM = [15, 14, 13, 12, 11, 10, 9, 8, 0, 1, 2, 3, 4, 5, 6, 7]


# ----------------------------------------------------------------------------- quant_ops.py
def cast_to_fp4(x):
    """Upstream cast_to_fp4 (E2M1 RTN, ties as upstream: 0.25->0, 0.75->1, 1.25->1, 1.75->2,
    2.5->2, 3.5->4, 5->4, i.e. ties-to-even mantissa)."""
    sign = torch.sign(x)
    x = torch.abs(x)
    x[(x >= 0.0) & (x <= 0.25)] = 0.0
    x[(x > 0.25) & (x < 0.75)] = 0.5
    x[(x >= 0.75) & (x <= 1.25)] = 1.0
    x[(x > 1.25) & (x < 1.75)] = 1.5
    x[(x >= 1.75) & (x <= 2.5)] = 2.0
    x[(x > 2.5) & (x < 3.5)] = 3.0
    x[(x >= 3.5) & (x <= 5.0)] = 4.0
    x[x > 5.0] = 6.0
    return x * sign


def quantize_fp4(x, scales, zeros, q_min, q_max):
    return cast_to_fp4(x / scales)


def dequantize_fp4(q, scales, zeros):
    return q.mul(scales)


def quantize_dequantize_fp4(x, scales, zeros, q_min, q_max):
    return dequantize_fp4(quantize_fp4(x, scales, zeros, q_min, q_max), scales, zeros)


def quantize_int(x, scales, zeros, q_min, q_max):
    return (x / scales + zeros).round().clamp(q_min, q_max)


def dequantize_int(q, scales, zeros):
    return q.sub(zeros).mul(scales)


def quantize_dequantize_int(x, scales, zeros, q_min, q_max):
    return dequantize_int(quantize_int(x, scales, zeros, q_min, q_max), scales, zeros)


def cast_to_eBm0(x, ebits: int, emax: int):
    qmin = -(2 ** (ebits - 1) - 1)
    qmax = +(2 ** (ebits - 1) - 1)
    return 2 ** (x.clamp(min=FP32_MIN_NORMAL).log2().floor().clamp(qmin, qmax) - emax)


def _reciprocal(x):
    if isinstance(x, torch.Tensor):
        return torch.where(x == 0, torch.zeros((), dtype=x.dtype, device=x.device), 1.0 / x)
    return 0.0 if x == 0 else 1.0 / x


def split_dim(x, num_splits, dim=-1):
    if dim == -1:
        dim = x.ndim - 1
    return x.reshape(*x.shape[:dim], num_splits, x.shape[dim] // num_splits, *x.shape[dim + 1:])


def pack_fp4_to_uint8(x: torch.Tensor) -> torch.Tensor:
    """Upstream packing (codes for QuTLASS / vLLM); x must already lie on the FP4 grid."""
    grid = torch.tensor(FP4_GRID, device=x.device)
    perm = torch.tensor(FP4_BITPACKING_PERM, device=x.device)
    ids = torch.bucketize(x, grid)
    lo, hi = (ids - 1).clamp(0, 15), ids.clamp(0, 15)
    g_lo, g_hi = grid[lo], grid[hi]
    pick_hi = (g_hi - x < x - g_lo) | (g_hi - x == x - g_lo) & (perm[hi] % 2 == 0)
    q = torch.where(pick_hi, perm[hi], perm[lo])
    return (q[:, 1::2] << 4 | q[:, ::2]).to(torch.uint8)


# ----------------------------------------------------------------------------- quantizer.py
class Quantizer:
    """Port of FP-Quant Quantizer. format in {int, fp, nvfp, mxfp}; scale_precision in
    {fp16, e4m3, e8m0}; observer in {minmax, mse}. Group granularity along last dim."""

    def __init__(self, bits=4, symmetric=True, format="nvfp", granularity="group", observer="minmax",
                 dim=-1, group_size=None, scale_precision="e4m3"):
        assert symmetric
        if granularity == "group":
            assert group_size is not None
        self.bits, self.symmetric, self.format = bits, symmetric, format
        self.granularity, self.observer, self.scale_precision = granularity, observer, scale_precision
        self.dim, self.group_size = dim, group_size
        if format in ("fp", "nvfp", "mxfp"):
            assert bits == 4
            self.quant_fn, self.dequant_fn, self.quant_dequant_fn = quantize_fp4, dequantize_fp4, quantize_dequantize_fp4
            self.q_min, self.q_max = -FP4_E2M1_MAX, FP4_E2M1_MAX
        else:
            self.quant_fn, self.dequant_fn, self.quant_dequant_fn = quantize_int, dequantize_int, quantize_dequantize_int
            r = 2 ** bits
            self.q_min, self.q_max = -r // 2, r // 2 - 1
        # Global scale is 3 for MXFP (4/3-rescaled E8M0 scales x 3 = exact power of two on export)
        self.global_scale = torch.tensor([3.0 if format == "mxfp" else float("inf")], dtype=torch.float32)
        self._track_global_scale = scale_precision == "e4m3"

    def _reshape(self, x, scales=None, zeros=None):
        if self.group_size:
            dim = x.ndim - 1 if self.dim == -1 else self.dim
            x = split_dim(x, x.shape[dim] // self.group_size, dim)
            if scales is not None:
                scales = scales.unsqueeze(dim + 1)
            if zeros is not None:
                zeros = zeros.unsqueeze(dim + 1)
        return x, scales, zeros

    @torch.no_grad()
    def get_quantization_params(self, x, scale_search_iters=100, max_scale_shrink_factor=0.80, error_norm=2.4):
        dim = x.ndim - 1 if self.dim == -1 else self.dim
        reduce_dim = dim + 1 if self.granularity == "group" else (dim if self.granularity == "channel" else None)
        x, _, _ = self._reshape(x)
        x_min = x.amin(dim=reduce_dim, keepdim=True)
        x_max = x.amax(dim=reduce_dim, keepdim=True)
        scales = 2 * torch.maximum(-x_min, x_max) / (self.q_max - self.q_min)
        zeros = torch.zeros_like(x_min)

        if self.observer == "mse":
            init_scales = scales.clone()
            best = torch.full(x.shape[:-1], float("inf"), device=x.device, dtype=x.dtype)
            for i in range(scale_search_iters):
                cand = (1 - i * max_scale_shrink_factor / scale_search_iters) * init_scales
                q = self.quant_fn(x, cand, zeros, self.q_min, self.q_max)
                err = (x - self.dequant_fn(q, cand, zeros)).abs_().pow_(error_norm).sum(dim=-1)
                if (err < best).any():
                    ids = torch.where(err < best)
                    best[ids] = err[ids]
                    scales[ids] = cand[ids]

        if self.group_size:
            x = x.flatten(dim, dim + 1)
            scales = scales.squeeze(dim + 1)
            zeros = zeros.squeeze(dim + 1)

        if self.scale_precision == "e4m3":
            if self._track_global_scale:
                cur = FP8_E4M3_MAX * FP4_E2M1_MAX * _reciprocal(x.abs().max().to(torch.float32).view(1))
                if not cur:
                    raise ValueError(f"global scale not finite: {cur}")
                self.global_scale = torch.minimum(self.global_scale.to(x.device), cur)
            gs = self.global_scale.to(x.device)
            scales = (scales * gs).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn) \
                .to(torch.float32).mul(_reciprocal(gs)).to(x.dtype)
        elif self.scale_precision == "e8m0":
            scales = cast_to_eBm0(FP4_E2M1_MAX * scales, ebits=8, emax=2) / FP4_SCALE

        scales[scales == 0] = 1
        if scales.isnan().any():
            raise ValueError("Scales are not finite.")
        return scales, zeros

    def quantize(self, x, scales, zeros=None):
        return self.quant_fn(*self._reshape(x, scales, zeros), self.q_min, self.q_max).reshape(x.shape)

    def dequantize(self, q, scales, zeros=None):
        return self.dequant_fn(*self._reshape(q, scales, zeros)).reshape(q.shape)

    def __call__(self, x, scales, zeros=None):
        return self.quant_dequant_fn(*self._reshape(x, scales, zeros), self.q_min, self.q_max).reshape(x.shape)


def make_quantizer(fmt: str, observer: str = "minmax") -> Quantizer:
    """Format presets exactly as model_quant.py parse_args() forces them."""
    if fmt == "nvfp":
        return Quantizer(4, True, "nvfp", "group", observer, group_size=NVFP_GROUPSIZE, scale_precision="e4m3")
    if fmt == "mxfp":
        return Quantizer(4, True, "mxfp", "group", observer, group_size=MXFP_GROUPSIZE, scale_precision="e8m0")
    raise ValueError(fmt)


# ----------------------------------------------------------------------------- transforms.py
_HAD_CACHE = {}


def hadamard_matrix(g: int, device, dtype=torch.float32):
    """Normalized Sylvester Hadamard (same ordering as fast_hadamard_transform / Dao-AILab)."""
    assert g >= 1 and (g & (g - 1)) == 0, "Hadamard size must be a power of two"
    key = (g, str(device), dtype)
    if key not in _HAD_CACHE:
        h = torch.ones(1, 1, dtype=torch.float64)
        while h.shape[0] < g:
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        _HAD_CACHE[key] = (h / math.sqrt(g)).to(device=device, dtype=dtype)
    return _HAD_CACHE[key]


class HadamardTransform(nn.Module):
    """Block-diagonal Hadamard along the last dim (self-inverse, symmetric).
    Computed in fp32 and cast back to the input dtype."""

    def __init__(self, group_size: int = 128):
        super().__init__()
        self.group_size = group_size

    def forward(self, x, inv_t: bool = False, dim: int = -1):
        assert dim == -1
        g = self.group_size
        shp, dt = x.shape, x.dtype
        assert shp[-1] % g == 0, f"dim {shp[-1]} not divisible by Hadamard size {g}"
        y = x.float().reshape(-1, g) @ hadamard_matrix(g, x.device)
        return y.reshape(shp).to(dt)


class IdentityTransform(nn.Module):
    group_size = None

    def forward(self, x, inv_t: bool = False, dim: int = -1):
        return x


def build_transform(kind: str, group_size: int):
    return HadamardTransform(group_size) if kind == "hadamard" else IdentityTransform()


# ----------------------------------------------------------------------------- gptq.py
def get_relative_mse_error(q, w, H):
    d = q - w
    return d.mm(H).mul(d).mean() / (w.mm(H).mul(w).mean() + 1e-6)


class GPTQ:
    """Port of FP-Quant GPTQ for nn.Linear-like modules exposing .weight [N, K]."""

    def __init__(self, layer, quantizer: Quantizer, quantization_order="default", block_size=128, rel_damp=1e-2):
        self.layer = layer
        self.W = layer.weight
        self.d_row, self.d_col = self.W.shape
        self.quantizer = quantizer
        self.quantization_order = quantization_order
        self.block_size = block_size
        self.rel_damp = rel_damp
        self.W_device, self.W_dtype = self.W.device, self.W.dtype
        self.H = None
        self.num_samples = 0

    @torch.no_grad()
    def update(self, inp: torch.Tensor):
        batch_size = inp.shape[0]
        if self.H is None:
            self.H = torch.zeros((self.d_col, self.d_col), device=inp.device, dtype=torch.float32)
        inp = inp.reshape(-1, inp.shape[-1]).float()
        beta = self.num_samples / (self.num_samples + batch_size)
        alpha = 2.0 / (self.num_samples + batch_size)
        self.H.mul_(beta)
        inp = inp * math.sqrt(alpha)
        self.H.addmm_(inp.t(), inp)
        self.num_samples += batch_size

    @torch.no_grad()
    def _get_hessian_inverse(self):
        w, H = self.W, self.H
        zero_cols = torch.nonzero(w.eq(0).all(dim=0))
        H[zero_cols, :] = 0
        H[:, zero_cols] = 0
        H[zero_cols, zero_cols] = 1
        damp = self.rel_damp * torch.diag(H).mean()
        H[range(self.d_col), range(self.d_col)] += damp
        try:
            Hi = torch.cholesky_inverse(torch.linalg.cholesky(H))
            self.hinv_fallback = False
            return torch.linalg.cholesky(Hi, upper=True)
        except Exception:
            self.hinv_fallback = True
            return torch.eye(self.d_col, device=H.device, dtype=torch.float32)

    @torch.no_grad()
    def quantize(self, export_codes: bool = False):
        """Returns (dequantized weight in W dtype, fp4 codes or None, scales [N, K/g])."""
        assert self.H is not None, "no calibration samples"
        self.W = self.W.clone().float()
        d_col, block_size, device = self.d_col, self.block_size, self.W_device
        qgs = self.quantizer.group_size
        group_size = qgs or d_col
        num_groups = d_col // group_size
        qweight = torch.empty(self.W.shape, device=device, dtype=torch.float32) if export_codes else None
        # grid/scales fixed up-front on the (rotated) weight -> static ordering keeps group structure
        scales, zeros = self.quantizer.get_quantization_params(self.W)
        self.quantizer.group_size = None
        if self.quantization_order == "activation":
            perm = torch.argsort(self.H.diag(), descending=True)
            group_idx = torch.arange(num_groups, device=device).repeat_interleave(group_size)[perm]
        else:
            perm = torch.arange(d_col, device=device)
        perm_inv = torch.argsort(perm)
        self.H = self.H[perm][:, perm]
        w = self.W[:, perm]
        H_inv_cho = self._get_hessian_inverse()
        for c1 in range(0, d_col, block_size):
            c2 = min(c1 + block_size, d_col)
            w_blk = w[:, c1:c2].clone()
            errs = torch.zeros_like(w_blk)
            Hb = H_inv_cho[c1:c2, c1:c2]
            for i in range(c2 - c1):
                w_ci = w_blk[:, i]
                d = Hb[i, i]
                g_idx = group_idx[c1 + i] if self.quantization_order == "activation" else (c1 + i) // group_size
                if export_codes:
                    q = self.quantizer.quantize(w_ci, scales[:, g_idx], zeros[:, g_idx])
                    w_q = self.quantizer.dequantize(q, scales[:, g_idx], zeros[:, g_idx])
                    qweight[:, c1 + i] = q
                else:
                    w_q = self.quantizer(w_ci, scales[:, g_idx], zeros[:, g_idx])
                w[:, c1 + i] = w_q
                err = (w_ci - w_q) / d
                w_blk[:, i:].addr_(err, Hb[i, i:], alpha=-1)
                errs[:, i] = err
            w[:, c2:].addmm_(errs, H_inv_cho[c1:c2, c2:], alpha=-1)
        w = w[:, perm_inv].contiguous()
        if qweight is not None:
            qweight = qweight[:, perm_inv].contiguous()
        self.H = self.H[perm_inv][:, perm_inv]
        self.quantizer.group_size = qgs
        return w.to(self.W_dtype), qweight, scales


@torch.no_grad()
def rtn_quantize(weight: torch.Tensor, quantizer: Quantizer):
    """Same fixed grid as GPTQ but without error feedback (MR-RTN baseline)."""
    w = weight.float()
    s, z = quantizer.get_quantization_params(w)
    return quantizer(w, s, z).to(weight.dtype), quantizer.quantize(w, s, z), s
