import math, multidev as D, pdmodel as M
N = D.N
windows = [(256, "decode-heavy"), (64, "prefill-heavy"), (192, "mixed"), (32, "prefill-heavy"), (96, "mixed"), (128, "mixed"), (48, "prefill-heavy")]
for pn in ["ASIC-compute-rich", "ASIC-DynaCore-example"]:
    p = M.PLATFORMS[pn]; m = M.model(p["model"])
    for slack in [1.5, 3.0]:
        ctx = 8192; tau = slack * M.t_decode_alone(p, m, 128 / N, ctx)
        def gm(k):
            v = [D.static_cap(p, m, B, ctx, tau, k) for B, _ in windows]
            return sum(math.log(x) for x in v if x > 0) / len(v) if all(x > 0 for x in v) else -1e9 + sum(x > 0 for x in v)
        kfix = max(range(1, N), key=gm)
        print(f"\n=== {pn}, tau={tau*1e3:.1f} ms (slack {slack} at B=128), ctx={ctx}: prefill capacity normalised to STATIC-oracle; 'x' = decode SLO missed ===")
        print(f"{'Btot':>5}  {'STATIC* tok/s':>14}{'k*':>4} | {'STATICfix':>10}{'TM':>7}{'FUSED':>7}{'SPATIAL':>8}")
        for B, lab in windows:
            st = [(D.static_cap(p, m, B, ctx, tau, k), k) for k in range(1, N)]
            so, ko = max(st); sf = D.static_cap(p, m, B, ctx, tau, kfix)
            cc = D.coloc_caps(p, m, B, ctx, tau)
            f = lambda v: ("     x" if v <= 0 else f"{v/so:6.2f}") if so > 0 else ("     x" if v <= 0 else f"{v:6.0f}a")
            print(f"{B:>5}  {so:>14.0f}{ko:>4} | {f(sf):>10}{f(cc['TM']):>7}{f(cc['FUSED']):>7}{f(cc['SPATIAL']):>8}   {lab}")
        print(f"   (STATICfix uses k={kfix} prefill devices of {N}; 'a' = absolute tok/s where static-oracle is infeasible)")
