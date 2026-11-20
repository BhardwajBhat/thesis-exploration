"""Fused mixed batches pay an efficiency penalty eta (multiplies achieved compute AND bandwidth efficiency of the
fused step). RPA (arXiv 2604.15464) gives no mixed-batch number, so eta is swept. TM (separate batches, the
Trainium/Neuron style) and SPATIAL pay none (their kernels are specialised)."""
import copy, itertools, pdmodel as M, multidev as D

def fused_pen(p, m, B, ctx, tau, eta):
    q = copy.deepcopy(p); q["e"] *= eta; q["BW"] *= eta
    return M.fused_rate(q, m, B, ctx, tau)[0]

def gain(p, m, B, ctx, tau, eta):
    tm = M.tm_rate(p, m, B, ctx, tau)[0]; fu = fused_pen(p, m, B, ctx, tau, eta); sp = M.spatial_rate(p, m, B, ctx, tau)[0]
    o = max(tm, fu); return sp / o if o > 0 else float('inf')

def breakeven(p, m, B, ctx, tau):
    """largest eta at which SPATIAL >= best(TM, FUSED_eta); 1.0 means spatial already wins with ideal fusion; 0 = never (TM wins)."""
    if gain(p, m, B, ctx, tau, 1.0) >= 1: return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if gain(p, m, B, ctx, tau, mid) >= 1: lo = mid
        else: hi = mid
    return lo

ETAS = [1.0, 0.9, 0.8, 0.7]
print("SINGLE DEVICE, ctx=4096: spatial gain over best(TM, FUSED*eta); 'be' = fused efficiency at which spatial ties")
print(f"{'platform':24s}{'B':>4}{'slk':>5} | " + "".join(f"{'eta='+str(e):>9}" for e in ETAS) + f"{'be':>7}")
for pn, p in M.PLATFORMS.items():
    m = M.model(p["model"])
    for B, s in itertools.product([p["Bs"][0], p["Bs"][len(p["Bs"])//2], p["Bs"][-1]], [1.5, 3.0]):
        tau = s * M.t_decode_alone(p, m, B, 4096)
        print(f"{pn:24s}{B:>4}{s:>5} | " + "".join(f"{gain(p,m,B,4096,tau,e):>9.2f}" for e in ETAS) + f"{breakeven(p,m,B,4096,tau):>7.2f}")

print("\nMULTI-DEVICE (N=8, compute-rich ASIC, 8B): spatial vs best co-located other, by fused penalty")
p = M.PLATFORMS["ASIC-compute-rich"]; m = M.model("8B")
print(f"{'Btot':>5}{'ctx':>7}{'slk':>5} | " + "".join(f"{'eta='+str(e):>9}" for e in ETAS) + f"{'be':>7}")
for Btot, ctx, s in itertools.product([32, 128, 512], [4096, 16384], [1.5, 3.0]):
    b = Btot / D.N; tau = s * M.t_decode_alone(p, m, b, ctx)
    print(f"{Btot:>5}{ctx:>7}{s:>5} | " + "".join(f"{gain(p,m,b,ctx,tau,e):>9.2f}" for e in ETAS) + f"{breakeven(p,m,b,ctx,tau):>7.2f}")
