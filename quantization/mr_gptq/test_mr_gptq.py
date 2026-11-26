#!/usr/bin/env python3
"""Unit tests for the MR-GPTQ port.

  .venv/bin/python quantization/mr_gptq/test_mr_gptq.py            # CPU/GPU tests
  FPQUANT_SRC=/path/to/FP-Quant .venv/bin/python quantization/mr_gptq/test_mr_gptq.py
      # + bit-exact parity against an upstream clone (fast_hadamard_transform is stubbed with
      #   the same Sylvester matrix, so the parity covers quantizer + GPTQ numerics).
"""
import math, os, sys, types, unittest
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fpquant_port as fq  # noqa: E402
from qwen35_mr_gptq import MRQLinear, group_of  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
FP4 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


class PortTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_hadamard_orthogonal_and_matches_scipy_order(self):
        for g in (16, 32, 128):
            H = fq.hadamard_matrix(g, "cpu")
            self.assertTrue(torch.allclose(H @ H, torch.eye(g), atol=1e-6))
            self.assertTrue(torch.equal(H, H.T))
            try:
                from scipy.linalg import hadamard
                self.assertTrue(torch.allclose(H, torch.tensor(hadamard(g), dtype=torch.float32) / math.sqrt(g)))
            except ImportError:
                pass

    def test_rotation_preserves_linear_output(self):
        lin = nn.Linear(256, 64, bias=False)
        x = torch.randn(3, 7, 256)
        q = MRQLinear(lin, fq.HadamardTransform(32), None, "x", "g")
        self.assertTrue(torch.allclose(q(x), lin(x), atol=1e-4))
        q.weight.data = q.transform(q.weight.data); q.rotated = True
        self.assertTrue(torch.allclose(q(x), lin(x), atol=1e-4))

    def test_nvfp_grid(self):
        qz = fq.make_quantizer("nvfp")
        w = torch.randn(64, 256) * torch.logspace(-2, 1, 256)
        s, z = qz.get_quantization_params(w)
        self.assertEqual(s.shape, (64, 16))
        gs = qz.global_scale
        self.assertTrue(torch.allclose(gs, 448 * 6 / w.abs().max()))
        sb = s * gs
        self.assertTrue(torch.allclose(sb.to(torch.float8_e4m3fn).float(), sb, rtol=1e-6))
        codes = qz.quantize(w, s)
        self.assertTrue(torch.isin(codes.abs(), FP4).all())
        self.assertLess(((qz(w, s) - w).norm() / w.norm()).item(), 0.15)

    def test_mxfp_grid_power_of_two_after_global(self):
        qz = fq.make_quantizer("mxfp")
        w = torch.randn(32, 256)
        s, _ = qz.get_quantization_params(w)
        self.assertEqual(s.shape, (32, 8))
        e = torch.log2(s * 3.0)   # FP-Quant: scales * global(3) == E8M0 power of two
        self.assertTrue(torch.allclose(e, e.round(), atol=1e-5))
        self.assertTrue(torch.isin(qz.quantize(w, s).abs(), FP4).all())

    def test_mse_observer_not_worse(self):
        w = torch.randn(128, 512)
        for f in ("nvfp", "mxfp"):
            a, b = fq.make_quantizer(f, "minmax"), fq.make_quantizer(f, "mse")
            ea = (a(w, *a.get_quantization_params(w)) - w).abs().pow(2.4).sum()
            eb = (b(w, *b.get_quantization_params(w)) - w).abs().pow(2.4).sum()
            self.assertLessEqual(eb.item(), ea.item() * (1 + 1e-6), f)

    def test_gptq_beats_rtn_on_hessian_loss(self):
        torch.manual_seed(1)
        K, N = 256, 64
        X = torch.randn(4096, K, device=DEV) @ torch.randn(K, K, device=DEV) * 0.1  # correlated inputs
        lin = nn.Linear(K, N, bias=False).to(DEV)
        for fmt in ("nvfp", "mxfp"):
            for order in ("default", "activation"):
                h = fq.GPTQ(lin, fq.make_quantizer(fmt), quantization_order=order)
                h.update(X.unsqueeze(0))
                dq, codes, s = h.quantize(export_codes=True)
                rtn, _, _ = fq.rtn_quantize(lin.weight.data, fq.make_quantizer(fmt))
                eg = ((X @ (dq - lin.weight).T) ** 2).mean().item()
                er = ((X @ (rtn - lin.weight).T) ** 2).mean().item()
                self.assertLess(eg, er, (fmt, order))
                # dequantized weight lies on the fixed grid: codes * per-group scale
                g = h.quantizer.group_size
                recon = (codes.view(N, -1, g) * s.unsqueeze(-1)).view(N, K)
                self.assertTrue(torch.allclose(recon, dq.float(), atol=1e-6))
                self.assertTrue(torch.isin(codes.abs().cpu(), FP4).all())

    def test_group_rules_cover_qwen35(self):
        names = ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.in_proj_b",
                 "linear_attn.in_proj_a", "linear_attn.out_proj", "self_attn.q_proj", "self_attn.k_proj",
                 "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
        self.assertEqual([group_of(n) for n in names],
                         ["gdn_in"] * 4 + ["gdn_out", "qkv", "qkv", "qkv", "o", "gate_up", "gate_up", "down"])
        self.assertIsNone(group_of("lm_head"))
        self.assertIsNone(group_of("linear_attn.conv1d"))


@unittest.skipUnless(os.environ.get("FPQUANT_SRC"), "set FPQUANT_SRC to an FP-Quant clone")
class UpstreamParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        stub = types.ModuleType("fast_hadamard_transform")
        stub.hadamard_transform = lambda x, scale=1.0: (x.float() @ (fq.hadamard_matrix(x.shape[-1], x.device)
                                                                    * math.sqrt(x.shape[-1]) * scale)).to(x.dtype)
        sys.modules.setdefault("fast_hadamard_transform", stub)
        sys.path.insert(0, os.environ["FPQUANT_SRC"])
        from src.quantization.quantizer import Quantizer as UQ
        from src.quantization import gptq as ug
        cls.UQ, cls.ug = UQ, ug

    def _uq(self, fmt, obs):
        g, sp = (16, "e4m3") if fmt == "nvfp" else (32, "e8m0")
        return self.UQ(bits=4, symmetric=True, format=fmt, granularity="group", observer=obs, group_size=g,
                       scale_precision=sp)

    def test_quantizer_parity(self):
        w = torch.randn(64, 512, device=DEV)
        for fmt in ("nvfp", "mxfp"):
            for obs in ("minmax", "mse"):
                a, b = self._uq(fmt, obs), fq.make_quantizer(fmt, obs)
                sa, za = a.get_quantization_params(w.clone()); sb, zb = b.get_quantization_params(w.clone())
                self.assertTrue(torch.equal(sa, sb), (fmt, obs))
                self.assertTrue(torch.equal(a(w, sa, za), b(w, sb, zb)), (fmt, obs))

    def test_gptq_parity(self):
        if DEV != "cuda":
            self.skipTest("upstream accumulate_hessian is a Triton kernel")
        torch.manual_seed(2)
        K, N = 512, 96
        X = (torch.randn(2, 1024, K, device=DEV) @ torch.randn(K, K, device=DEV) * 0.05)
        for fmt in ("nvfp", "mxfp"):
            for order in ("default", "activation"):
                lin = nn.Linear(K, N, bias=False).to(DEV)
                hu = self.ug.GPTQ(lin, self._uq(fmt, "mse"), quantization_order=order, rel_damp=1e-2)
                hp = fq.GPTQ(lin, fq.make_quantizer(fmt, "mse"), quantization_order=order, rel_damp=1e-2)
                for xb in X:   # same per-sample Hessian update order
                    hu.update(xb.unsqueeze(0).clone()); hp.update(xb.unsqueeze(0).clone())  # upstream mutates input in place
                # upstream Triton kernel uses TF32 tl.dot; port is full fp32 -> compare in relative norm
                self.assertLess(((hu.H - hp.H).norm() / hp.H.norm()).item(), 2e-3)
                hp.H = hu.H.clone()   # isolate GPTQ loop from accumulation rounding
                du = hu.quantize()[0].float(); dp = hp.quantize()[0].float()
                frac = (du != dp).float().mean().item()
                self.assertLess(frac, 1e-3, (fmt, order, frac))


if __name__ == "__main__":
    unittest.main(verbosity=2)
