"""
Analytical model: prefill/decode on one reshapable PE array.
Schemes (all must meet decode TPOT <= tau for a decode batch B; metric = prefill tokens/s served):
  TM      : DynaCore-style time multiplexing. Each iteration = decode step (whole array, W4A16)
            + one prefill chunk (whole array, W8A8) + 2 reshape switches.
  FUSED   : chunked prefill. One GEMM with M = B + C rows in ONE format (best of W8A8 / W4A16).
  SPATIAL : decode on a partition f of the array (W4A16), prefill concurrently on 1-f (W8A8),
            sharing DRAM bandwidth (decode has priority). f quantized to 1/G.
All numbers are first-order roofline: time = max(compute, memory). No overlap losses beyond the
efficiency factors below. Every assumption is a parameter.
"""
import math, itertools, json, sys

# ---------------- models ----------------
def model(name):
    if name == "8B":     # Qwen3-8B-like dense
        return dict(name=name, P=6.95e9, L=36, H=32, kvh=8, hd=128, state=0)
    if name == "1.7B":   # Qwen3-1.7B-like dense
        return dict(name=name, P=1.41e9, L=28, H=16, kvh=8, hd=128, state=0)
    if name == "0.8B":   # Qwen3.5-0.8B-like hybrid (APPROXIMATE): few full-attn layers + DeltaNet state
        return dict(name=name, P=0.75e9, L=6, H=8, kvh=2, hd=256, state=2*9.4e6)  # state bytes r+w per req per step
    raise ValueError(name)

KV_BYTES = 1  # int8 KV cache
def kv_per_tok(m):  return 2 * m["L"] * m["kvh"] * m["hd"] * KV_BYTES
def attn_mac_per_tok(m, ctx): return 2 * m["L"] * m["H"] * m["hd"] * ctx

# ---------------- platforms ----------------
# R8: peak INT8 MAC/s of the array; r16: rate of W4A16/FP16-activation MACs relative to INT8 (DynaCore packs 2 INT8 in FP16 mult -> 0.5)
# BW: effective DRAM B/s; e: achieved efficiency of the array; G: partition granularity; sw: reshape/repartition switch cost (s)
PLATFORMS = {
 "ASIC-DynaCore-example": dict(R8=16384*0.5e9, r16=0.5, BW=4.0e12, e=0.9, G=16, sw=1e-6, model="8B",
                               Bs=[1,8,32,128], note="128x128-equivalent @500MHz, 4 TB/s (DynaCore motivating example)"),
 "ASIC-compute-rich":     dict(R8=65536*1.0e9, r16=0.5, BW=1.0e12, e=0.9, G=16, sw=1e-6, model="8B",
                               Bs=[1,8,32,128], note="4x128x128 @1GHz, 1 TB/s HBM (TPU-like balance)"),
 "ASIC-edge-NPU":         dict(R8=16e12, r16=0.5, BW=60e9, e=0.8, G=16, sw=1e-6, model="1.7B",
                               Bs=[1,2,4,8], note="~32 INT8 TOPS, LPDDR5X ~60 GB/s eff (phone-NPU-like)"),
 "VCK5000-AIE":           dict(R8=72.5e12, r16=0.5, BW=45e9, e=0.4, G=8, sw=10e-6, model="1.7B",
                               Bs=[1,2,4,8], note="145 INT8 TOPS AIE, ~45 GB/s usable DDR4 (HeteroVLM measured ~40-50)"),
 "KV260-PL":              dict(R8=1248*2*0.3e9, r16=0.5, BW=12e9, e=0.5, G=4, sw=5e-6, model="0.8B",
                               Bs=[1,2,4,8], note="1248 DSP x2 INT8 @300MHz, ~12 GB/s effective (from measured 4/4 decode)"),
}
E_PART = 0.95   # efficiency loss from running on a sub-array (edge tiles, smaller tiles)
C_SPATIAL = 2048  # prefill chunk size used by the concurrent prefill partition (not TPOT-bound)

def wbytes(m, bits): return m["P"] * bits / 8

def decode_cost(m, B, ctx):
    macs = B * (m["P"] + attn_mac_per_tok(m, ctx))
    bytes_ = wbytes(m, 4) + B * (ctx * kv_per_tok(m) + kv_per_tok(m) + m["state"])
    return macs, bytes_

def prefill_cost(m, C, ctx, wbits=8):
    macs = C * m["P"] + C * 2 * m["L"] * m["H"] * m["hd"] * (ctx + C / 2)
    bytes_ = wbytes(m, wbits) + ctx * kv_per_tok(m) + C * kv_per_tok(m) + m["state"]
    return macs, bytes_

def t_decode_alone(p, m, B, ctx):
    macs, by = decode_cost(m, B, ctx)
    return max(macs / (p["R8"] * p["r16"] * p["e"]), by / p["BW"])

def tm_rate(p, m, B, ctx, tau):
    td = t_decode_alone(p, m, B, ctx)
    def it_time(C):
        macs, by = prefill_cost(m, C, ctx, 8)
        return td + max(macs / (p["R8"] * p["e"]), by / p["BW"]) + 2 * p["sw"]
    if it_time(1) > tau: return 0.0, 0
    lo, hi = 1, 1 << 20
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if it_time(mid) <= tau: lo = mid
        else: hi = mid - 1
    return lo / it_time(lo), lo

def fused_rate(p, m, B, ctx, tau):
    best = (0.0, 0, None)
    for fmt, wb, rate in (("W8A8", 8, p["R8"]), ("W4A16", 4, p["R8"] * p["r16"])):
        def it_time(C):
            dm = B * (m["P"] + attn_mac_per_tok(m, ctx))
            pm = C * m["P"] + C * 2 * m["L"] * m["H"] * m["hd"] * (ctx + C / 2)
            by = wbytes(m, wb) + B * (ctx * kv_per_tok(m) + kv_per_tok(m) + m["state"]) \
                 + (ctx * kv_per_tok(m) + C * kv_per_tok(m) + m["state"] if C > 0 else 0)
            return max((dm + pm) / (rate * p["e"]), by / p["BW"])
        if it_time(1) > tau: continue
        lo, hi = 1, 1 << 20
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if it_time(mid) <= tau: lo = mid
            else: hi = mid - 1
        r = lo / it_time(lo)
        if r > best[0]: best = (r, lo, fmt)
    return best

def spatial_rate(p, m, B, ctx, tau):
    dmacs, dby = decode_cost(m, B, ctx)
    if dby / p["BW"] > tau: return 0.0, None
    pmacs, pby = prefill_cost(m, C_SPATIAL, ctx, 8)
    mac_tok, by_tok = pmacs / C_SPATIAL, pby / C_SPATIAL
    best = (0.0, None)
    for k in range(1, p["G"]):
        f = k / p["G"]
        td = max(dmacs / (f * p["R8"] * p["r16"] * p["e"] * E_PART), dby / p["BW"])
        if td > tau: continue
        bw_left = p["BW"] - dby / tau
        comp = (1 - f) * p["R8"] * p["e"] * E_PART
        x = min(comp / mac_tok, bw_left / by_tok) * (1 - p["sw"] / tau)
        if x > best[0]: best = (x, f)
    return best

def run():
    rows = []
    for pname, p in PLATFORMS.items():
        m = model(p["model"])
        beta = p["R8"] * p["e"] / p["BW"]
        for B, ctx, slack in itertools.product(p["Bs"], [2048, 16384], [1.5, 3.0]):
            td0 = t_decode_alone(p, m, B, ctx)
            tau = slack * td0
            tm, Ctm = tm_rate(p, m, B, ctx, tau)
            fu, Cfu, ffmt = fused_rate(p, m, B, ctx, tau)
            sp, fsp = spatial_rate(p, m, B, ctx, tau)
            other = max(tm, fu)
            rows.append(dict(platform=pname, model=m["name"], beta=beta, B=B, ctx=ctx, slack=slack,
                             td0_ms=td0*1e3, tau_ms=tau*1e3, tm=tm, C_tm=Ctm, fused=fu, C_fu=Cfu, fused_fmt=ffmt,
                             spatial=sp, f_dec=fsp, gain=(sp/other if other > 0 else float('inf'))))
    return rows

def pdd_experiment():
    """Two devices: static PD disaggregation (1 prefill dev + 1 decode dev, KV over a link)
       vs dynamic spatial partition on both devices vs TM on both. Fixed absolute TPOT SLO."""
    out = []
    for pname in ["ASIC-DynaCore-example", "ASIC-compute-rich"]:
        p = PLATFORMS[pname]; m = model(p["model"])
        for Btot, ctx, link in itertools.product([16, 64, 128], [2048, 16384], [16e9, 64e9]):
            td_static = t_decode_alone(p, m, Btot, ctx)
            tau = 2.0 * td_static       # SLO set so the static decode device is feasible
            pm, pb = prefill_cost(m, C_SPATIAL, ctx, 8)
            x_dev = C_SPATIAL / max(pm / (p["R8"] * p["e"]), pb / p["BW"])
            x_static = min(x_dev, link / kv_per_tok(m))
            x_dyn = 2 * spatial_rate(p, m, Btot // 2, ctx, tau)[0]
            x_tm = 2 * tm_rate(p, m, Btot // 2, ctx, tau)[0]
            out.append(dict(platform=pname, Btot=Btot, ctx=ctx, link_GBs=link/1e9, tau_ms=tau*1e3,
                            static=x_static, dyn_spatial=x_dyn, tm_both=x_tm,
                            tpot_noprefill_static_ms=td_static*1e3,
                            tpot_noprefill_dyn_ms=t_decode_alone(p, m, Btot//2, ctx)*1e3))
    return out

if __name__ == "__main__":
    rows = run(); pdd = pdd_experiment()
    json.dump(dict(single=rows, pdd=pdd), open("results.json", "w"), indent=1)
    print("platform betas (effective MAC/byte):")
    for pname, p in PLATFORMS.items():
        print(f"  {pname:24s} beta={p['R8']*p['e']/p['BW']:8.1f}  model={p['model']}  {p['note']}")
    print("\nSINGLE DEVICE: prefill tokens/s under TPOT <= slack x decode-alone time")
    hdr = f"{'platform':24s}{'B':>4}{'ctx':>6}{'slk':>5}{'td0ms':>8}{'TM':>10}{'FUSED':>10}{'fmt':>7}{'SPATIAL':>10}{'f_dec':>6}{'gain':>7}"
    print(hdr)
    for r in rows:
        print(f"{r['platform']:24s}{r['B']:>4}{r['ctx']:>6}{r['slack']:>5}{r['td0_ms']:>8.2f}{r['tm']:>10.0f}{r['fused']:>10.0f}"
              f"{(r['fused_fmt'] or '-'):>7}{r['spatial']:>10.0f}{(r['f_dec'] or 0):>6.2f}{r['gain']:>7.2f}")
    print("\nTWO DEVICES: static PD disaggregation vs dynamic spatial on both vs TM on both (prefill tok/s)")
    print(f"{'platform':24s}{'Btot':>5}{'ctx':>6}{'link':>5}{'static':>10}{'dynSP':>10}{'TMboth':>10}{'TPOT0 st/dyn ms':>18}")
    for r in pdd:
        print(f"{r['platform']:24s}{r['Btot']:>5}{r['ctx']:>6}{r['link_GBs']:>5.0f}{r['static']:>10.0f}{r['dyn_spatial']:>10.0f}{r['tm_both']:>10.0f}"
              f"{r['tpot_noprefill_static_ms']:>9.2f}/{r['tpot_noprefill_dyn_ms']:<8.2f}")
