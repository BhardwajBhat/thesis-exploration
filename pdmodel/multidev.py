"""
N-device comparison under a TIGHT per-token (TPOT) target.
  STATIC(k): k prefill-only devices + (N-k) decode-only devices; decode batch spread over N-k devices;
             KV of each prefilled token sent over a per-prefill-device link. k swept 1..N-1.
             STATIC-oracle = best k for each operating point; STATIC-fixed = one k for the whole trace.
  CO-LOCATED: every device takes B_tot/N decode requests and runs TM / FUSED / SPATIAL locally.
tau is set relative to the co-located per-device decode time: tau = slack * t_decode_alone(B_tot/N).
Metric: max prefill tokens/s the whole system serves while every decode request meets TPOT <= tau.
NOTE: the trace() function here normalises demand to the best co-located scheme (biased); use trace2.py instead.
"""
import itertools, pdmodel as M

N, C, LINK = 8, 2048, 16e9

def static_cap(p, m, Btot, ctx, tau, k):
    nd = N - k
    if nd <= 0 or M.t_decode_alone(p, m, Btot / nd, ctx) > tau: return 0.0
    pm, pb = M.prefill_cost(m, C, ctx, 8)
    x_dev = C / max(pm / (p["R8"] * p["e"]), pb / p["BW"])
    return k * min(x_dev, LINK / M.kv_per_tok(m))

def coloc_caps(p, m, Btot, ctx, tau):
    b = Btot / N
    return dict(TM=N * M.tm_rate(p, m, b, ctx, tau)[0],
                FUSED=N * M.fused_rate(p, m, b, ctx, tau)[0],
                SPATIAL=N * M.spatial_rate(p, m, b, ctx, tau)[0])

def table(pn):
    p = M.PLATFORMS[pn]; m = M.model(p["model"])
    print(f"\n=== {pn} ({m['name']}), N={N} devices, prefill tok/s at TPOT <= tau ===")
    print(f"{'Btot':>5}{'ctx':>7}{'slack':>6}{'tau_ms':>8} | {'STATIC*':>9}{'k*':>4} | {'TM':>9}{'FUSED':>9}{'SPATIAL':>9} | {'SP/STATIC*':>10}{'SP/best-coloc-other':>20}")
    for Btot, ctx, slack in itertools.product([32, 128, 512], [4096, 16384], [1.2, 1.5, 2.0, 3.0]):
        tau = slack * M.t_decode_alone(p, m, Btot / N, ctx)
        st = [(static_cap(p, m, Btot, ctx, tau, k), k) for k in range(1, N)]
        sbest, kbest = max(st)
        cc = coloc_caps(p, m, Btot, ctx, tau)
        oth = max(cc["TM"], cc["FUSED"])
        r1 = cc["SPATIAL"] / sbest if sbest > 0 else float('inf')
        r2 = cc["SPATIAL"] / oth if oth > 0 else float('inf')
        print(f"{Btot:>5}{ctx:>7}{slack:>6}{tau*1e3:>8.2f} | {sbest:>9.0f}{(kbest if sbest>0 else 0):>4} | {cc['TM']:>9.0f}{cc['FUSED']:>9.0f}{cc['SPATIAL']:>9.0f} | {r1:>10.2f}{r2:>20.2f}")

if __name__ == "__main__":
    for pn in ["ASIC-compute-rich", "ASIC-DynaCore-example"]:
        table(pn)
