"""
eta microbenchmark: how much efficiency does a MIXED prefill+decode step lose vs SPECIALISED steps?

One Qwen3-8B-shaped decoder layer (BF16, random weights). Three ways to do the same work:
  dec : B decode tokens, each attending to its own ctx-token KV cache
  pre : one C-token prefill chunk on top of a ctx-token prefix (causal, bottom-right aligned)
  mix : both in ONE step (chunked-prefill style): GEMMs on B+C packed rows, one ragged attention call

Efficiency of a step = roofline_time / measured_time, with roofline from analytic FLOPs/bytes and
peak MAC rate + DRAM bandwidth calibrated on this device. Then
  eta = eff(mix) / eff(dec+pre)   (eff(dec+pre) = (roof_dec + roof_pre) / (t_dec + t_pre))
eta < 1 means the mixed step runs further from its roofline than the specialised steps do.
Separate steps legitimately pay the weight read twice; that is in their roofline, so it does not bias eta.

Attention variants:
  fa   : FlashAttention-2 varlen for all three steps (what vLLM uses for mixed batches on Ampere/Ada):
         each request gets its own query tiles, so this is the realistic fused kernel
  flex : torch flex_attention, mix = ONE packed sequence with a document mask; decode queries share
         query tiles with prefill rows -> pessimistic upper bound on the ragged-batch cost
  best : SDPA for dec and pre (best specialised kernels); mix uses FA varlen
Token budget modes:
  extra : prefill chunk = C, mixed step = B + C rows (adds B rows on top of an aligned chunk)
  fixed : mixed step = C rows total (prefill chunk = C - B), as vLLM's max_num_batched_tokens does
Components are timed separately (gemm = QKV/O/MLP incl. norms+act; attn) and as a whole layer.
"""
import argparse, csv, itertools, os, statistics, sys, time
import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from torch.nn.attention.bias import causal_lower_right
try:
    from flash_attn import flash_attn_varlen_func
except ImportError:
    flash_attn_varlen_func = None

HID, H, KVH, HD, FFN = 4096, 32, 8, 128, 12288     # Qwen3-8B layer
DT = torch.bfloat16
BYTES = 2
dev = "cuda"
# Every (B, ctx, C) shape is a fresh static compile; the default limit (8) is hit
# early in the sweep and dynamo then silently falls back to eager flex (~100x slower).
torch._dynamo.config.cache_size_limit = 1024
if hasattr(torch._dynamo.config, "recompile_limit"):
    torch._dynamo.config.recompile_limit = 1024
flex = torch.compile(flex_attention, dynamic=False)

# ---------------- timing ----------------
def bench(fn, iters=20, warmup=5):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) / 1e3)
    return statistics.median(ts)

def calibrate():
    a = torch.randn(8192, 8192, device=dev, dtype=DT); b = torch.randn(8192, 8192, device=dev, dtype=DT)
    t = bench(lambda: a @ b, iters=10)
    mac = 8192 ** 3 / t
    x = torch.empty(512 * 2**20, device=dev, dtype=torch.uint8); y = torch.empty_like(x)
    t = bench(lambda: y.copy_(x), iters=10)
    bw = 2 * x.numel() / t
    del a, b, x, y; torch.cuda.empty_cache()
    return mac, bw

# ---------------- layer ----------------
class Layer:
    def __init__(self):
        r = lambda *s: (torch.randn(*s, device=dev, dtype=DT) * 0.02)
        self.wqkv, self.wo = r((H + 2 * KVH) * HD, HID), r(HID, H * HD)
        self.wgu, self.wd = r(2 * FFN, HID), r(HID, FFN)
        self.n1, self.n2 = torch.ones(HID, device=dev, dtype=DT), torch.ones(HID, device=dev, dtype=DT)
    def wbytes(self):
        return sum(w.numel() for w in (self.wqkv, self.wo, self.wgu, self.wd)) * BYTES
    def pre_attn(self, x):
        return F.linear(F.rms_norm(x, (HID,), self.n1), self.wqkv)
    def post_attn(self, x, a):
        h = x + F.linear(a, self.wo)
        gu = F.linear(F.rms_norm(h, (HID,), self.n2), self.wgu)
        g, u = gu.chunk(2, -1)
        return h + F.linear(F.silu(g) * u, self.wd)

def gemm_cost(L, T):
    macs = T * (HID * (H + 2 * KVH) * HD + H * HD * HID + HID * 2 * FFN + FFN * HID)
    act = T * (HID * 6 + (H + 2 * KVH) * HD + 2 * FFN * 2 + FFN) * BYTES   # rough activation traffic
    return macs, L.wbytes() + act

def attn_cost(nq_ctx_pairs):
    """list of (num_query_tokens, kv_len, causal) -> (macs, bytes)"""
    macs = by = 0
    for nq, kv, causal in nq_ctx_pairs:
        eff_kv = (kv - nq / 2) if causal else kv
        macs += 2 * H * HD * nq * eff_kv
        by += 2 * KVH * HD * kv * BYTES + 2 * nq * H * HD * BYTES
    return macs, by

# ---------------- attention impls ----------------
def make_case(B, ctx, C):
    """Tensors for one (B, ctx, C) point. KV caches are separate per request (realistic paged-free layout)."""
    g = lambda *s: torch.randn(*s, device=dev, dtype=DT)
    d = dict(
        qd=g(B, H, 1, HD), kd=g(B, KVH, ctx, HD), vd=g(B, KVH, ctx, HD),
        qp=g(1, H, C, HD), kp=g(1, KVH, ctx + C, HD), vp=g(1, KVH, ctx + C, HD))
    # packed layout for mix: queries [B dec | C pre], kv [B*ctx dec | ctx+C pre]
    d["qm"] = torch.cat([d["qd"].permute(2, 1, 0, 3).reshape(1, H, B, HD).contiguous(), d["qp"]], 2)
    d["km"] = torch.cat([d["kd"].permute(1, 0, 2, 3).reshape(1, KVH, B * ctx, HD), d["kp"]], 2).contiguous()
    d["vm"] = torch.cat([d["vd"].permute(1, 0, 2, 3).reshape(1, KVH, B * ctx, HD), d["vp"]], 2).contiguous()
    Q, K = B + C, B * ctx + ctx + C
    qdoc = torch.cat([torch.arange(B, device=dev), torch.full((C,), B, device=dev)])
    kdoc = torch.cat([torch.arange(B, device=dev).repeat_interleave(ctx), torch.full((ctx + C,), B, device=dev)])
    qpos = torch.cat([torch.full((B,), ctx, device=dev), torch.arange(ctx, ctx + C, device=dev)])
    kpos = torch.cat([torch.arange(ctx, device=dev).repeat(B), torch.arange(ctx + C, device=dev)])
    def mm(b, h, q, k): return (qdoc[q] == kdoc[k]) & (kpos[k] <= qpos[q])
    d["bm_mix"] = create_block_mask(mm, 1, None, Q, K, device=dev)
    # flex masks for specialised steps
    d["bm_pre"] = create_block_mask(lambda b, h, q, k: k <= q + ctx, 1, None, C, ctx + C, device=dev)
    # FA varlen layout: [tokens, heads, hd]; cu_seqlens per request
    i32 = lambda xs: torch.tensor(xs, device=dev, dtype=torch.int32)
    d["fq"] = d["qm"][0].transpose(0, 1).contiguous(); d["fk"] = d["km"][0].transpose(0, 1).contiguous(); d["fv"] = d["vm"][0].transpose(0, 1).contiguous()
    d["cuq_mix"] = i32(list(range(B + 1)) + [B + C]); d["cuk_mix"] = i32([i * ctx for i in range(B + 1)] + [B * ctx + ctx + C])
    d["cuq_dec"] = i32(list(range(B + 1))); d["cuk_dec"] = i32([i * ctx for i in range(B + 1)])
    d["cuq_pre"] = i32([0, C]); d["cuk_pre"] = i32([0, ctx + C])
    d["fq_dec"], d["fk_dec"], d["fv_dec"] = d["fq"][:B], d["fk"][:B * ctx], d["fv"][:B * ctx]
    d["fq_pre"], d["fk_pre"], d["fv_pre"] = d["fq"][B:], d["fk"][B * ctx:], d["fv"][B * ctx:]
    d["dims"] = (B, ctx, C)
    return d

def fa(q, k, v, cuq, cuk, mq, mk):
    return flash_attn_varlen_func(q, k, v, cuq, cuk, mq, mk, causal=True)

def attn_dec(d, kind):
    B, ctx, C = d["dims"]
    if kind == "fa":
        return fa(d["fq_dec"], d["fk_dec"], d["fv_dec"], d["cuq_dec"], d["cuk_dec"], 1, ctx)
    if kind == "best":
        return F.scaled_dot_product_attention(d["qd"], d["kd"], d["vd"], enable_gqa=True)
    return flex(d["qd"], d["kd"], d["vd"], enable_gqa=True)

def attn_pre(d, kind):
    B, ctx, C = d["dims"]
    if kind == "fa":
        return fa(d["fq_pre"], d["fk_pre"], d["fv_pre"], d["cuq_pre"], d["cuk_pre"], C, ctx + C)
    if kind == "best":
        C, kv = d["qp"].shape[2], d["kp"].shape[2]
        return F.scaled_dot_product_attention(d["qp"], d["kp"], d["vp"], attn_mask=causal_lower_right(C, kv), enable_gqa=True)
    return flex(d["qp"], d["kp"], d["vp"], block_mask=d["bm_pre"], enable_gqa=True)

def attn_mix(d, kind):
    B, ctx, C = d["dims"]
    if kind == "flex":
        return flex(d["qm"], d["km"], d["vm"], block_mask=d["bm_mix"], enable_gqa=True)
    return fa(d["fq"], d["fk"], d["fv"], d["cuq_mix"], d["cuk_mix"], C, ctx + C)

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", default="1,4,16,64"); ap.add_argument("--ctx", default="2048,8192,16384")
    ap.add_argument("--C", default="256,512,1024,2048"); ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--out", default=None); ap.add_argument("--max-kv-gb", type=float, default=8.0)
    ap.add_argument("--modes", default="fixed,extra"); ap.add_argument("--kinds", default="fa,flex,best")
    a = ap.parse_args()
    global KINDS
    KINDS = [k for k in a.kinds.split(",") if k != "fa" or flash_attn_varlen_func is not None]
    if flash_attn_varlen_func is None:   # "best" mixes with FA varlen, so it needs FA too
        KINDS = [k for k in KINDS if k != "best"]
        print("# flash_attn not installed: skipping kind=fa and kind=best (flex only)")
    ints = lambda s: [int(x) for x in s.split(",")]
    name = torch.cuda.get_device_name(0).replace(" ", "_")
    out = a.out or os.path.join(os.path.dirname(__file__), "results", f"eta_{name}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.manual_seed(0)
    mac, bw = calibrate()
    print(f"# device {torch.cuda.get_device_name(0)}  torch {torch.__version__}  "
          f"calibrated BF16 {2*mac/1e12:.1f} TFLOPS, copy BW {bw/1e9:.0f} GB/s, beta={mac/bw:.0f} MAC/B", flush=True)
    roof = lambda macs, by: max(macs / mac, by / bw)
    L = Layer()
    fields = ["device", "mode", "B", "ctx", "C", "part", "kind", "t_dec_us", "t_pre_us", "t_mix_us",
              "roof_dec_us", "roof_pre_us", "roof_mix_us", "eff_sep", "eff_mix", "eta", "speedup_mix_vs_sep"]
    f = open(out, "w", newline=""); w = csv.DictWriter(f, fields); w.writeheader()
    print(f"{'mode':>5}{'B':>3}{'ctx':>6}{'C':>5} {'part':>5} {'kind':>4} {'t_dec':>8}{'t_pre':>8}{'t_mix':>8}  {'eff_sep':>7}{'eff_mix':>8}{'eta':>6}{'sep/mix':>8}")
    for mode, B, ctx, Cb in itertools.product(a.modes.split(","), ints(a.B), ints(a.ctx), ints(a.C)):
        C = Cb - B if mode == "fixed" else Cb
        if C < 1: print(f"skip {mode} B={B} C={Cb}: no room for prefill"); continue
        kv_gb = 2 * 2 * KVH * HD * (B * ctx + ctx + C) * BYTES / 1e9   # separate + packed copies
        if kv_gb > a.max_kv_gb: print(f"skip B={B} ctx={ctx} C={C}: {kv_gb:.1f} GB KV"); continue
        d = make_case(B, ctx, C)
        xd, xp = torch.randn(B, HID, device=dev, dtype=DT), torch.randn(C, HID, device=dev, dtype=DT)
        xm = torch.cat([xd, xp])
        ad = torch.randn(B, H * HD, device=dev, dtype=DT); ap_ = torch.randn(C, H * HD, device=dev, dtype=DT)
        am = torch.cat([ad, ap_])
        gd, gp, gm = gemm_cost(L, B), gemm_cost(L, C), gemm_cost(L, B + C)
        attd = attn_cost([(1, ctx, False)] * B); attp = attn_cost([(C, ctx + C, True)]); attm = (attd[0] + attp[0], attd[1] + attp[1])
        r = {}
        r["gemm"] = (lambda: (L.pre_attn(xd), L.post_attn(xd, ad)), lambda: (L.pre_attn(xp), L.post_attn(xp, ap_)),
                     lambda: (L.pre_attn(xm), L.post_attn(xm, am)), gd, gp, gm)
        for kind in KINDS:
            r[f"attn/{kind}"] = (lambda k=kind: attn_dec(d, k), lambda k=kind: attn_pre(d, k), lambda k=kind: attn_mix(d, k), attd, attp, attm)
            r[f"layer/{kind}"] = (lambda k=kind: (L.pre_attn(xd), attn_dec(d, k), L.post_attn(xd, ad)),
                                  lambda k=kind: (L.pre_attn(xp), attn_pre(d, k), L.post_attn(xp, ap_)),
                                  lambda k=kind: (L.pre_attn(xm), attn_mix(d, k), L.post_attn(xm, am)),
                                  (gd[0] + attd[0], gd[1] + attd[1]), (gp[0] + attp[0], gp[1] + attp[1]), (gm[0] + attm[0], gm[1] + attm[1]))
        for key, (fd, fp, fm, cd, cp, cm) in r.items():
            part, _, kind = key.partition("/")
            td, tp, tm = bench(fd, a.iters), bench(fp, a.iters), bench(fm, a.iters)
            rd, rp, rm = roof(*cd), roof(*cp), roof(*cm)
            eff_sep, eff_mix = (rd + rp) / (td + tp), rm / tm
            row = dict(device=name, mode=mode, B=B, ctx=ctx, C=C, part=part, kind=kind or "-",
                       t_dec_us=td * 1e6, t_pre_us=tp * 1e6, t_mix_us=tm * 1e6,
                       roof_dec_us=rd * 1e6, roof_pre_us=rp * 1e6, roof_mix_us=rm * 1e6,
                       eff_sep=eff_sep, eff_mix=eff_mix, eta=eff_mix / eff_sep, speedup_mix_vs_sep=(td + tp) / tm)
            w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in row.items()}); f.flush()
            print(f"{mode:>5}{B:>3}{ctx:>6}{C:>5} {part:>5} {(kind or '-'):>4} {td*1e6:>8.0f}{tp*1e6:>8.0f}{tm*1e6:>8.0f}  "
                  f"{eff_sep:>7.2f}{eff_mix:>8.2f}{eff_mix/eff_sep:>6.2f}{(td+tp)/tm:>8.2f}", flush=True)
        del d; torch.cuda.empty_cache()
    print(f"# wrote {out}")

if __name__ == "__main__":
    main()
