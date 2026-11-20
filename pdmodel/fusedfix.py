import copy, itertools, pdmodel as M, pdmodel2 as M2
def fused_energy(p, en, m, B, ctx, tau, lam):
    Np = max(lam * tau, 1); best = None
    dm = B * (m["P"] + M.attn_mac_per_tok(m, ctx)); pm, pb = M.prefill_cost(m, Np, ctx, 8)
    _, dby = M.decode_cost(m, B, ctx); kvd = dby - M.wbytes(m, 4); kvp = pb - M.wbytes(m, 8)
    for wb, rate, emul in ((8, p["R8"], 1), (4, p["R8"] * p["r16"], 2)):
        t = max((dm + pm) / (rate * p["e"]), (M.wbytes(m, wb) + kvd + kvp) / p["BW"])
        if t > tau: continue
        E = (M.wbytes(m, wb) + kvd + kvp) * en["e_byte"] + (dm + pm) * emul * en["e_mac8"] + en["P_static"] * tau
        best = E if best is None else min(best, E)
    return None if best is None else best / (B + Np)
ctx = 4096
print(f"{'platform':24s}{'B':>4}{'slk':>5} | E_FUSED(best fmt)  E_SPATIAL   SP saving")
for pn, p in M.PLATFORMS.items():
    m = M.model(p["model"]); en = M2.ENERGY[pn]
    for B, s in itertools.product([p["Bs"][0], p["Bs"][-1]], [1.5, 3.0]):
        tau = s * M.t_decode_alone(p, m, B, ctx)
        caps = [M.tm_rate(p,m,B,ctx,tau)[0], M.fused_rate(p,m,B,ctx,tau)[0], M.spatial_rate(p,m,B,ctx,tau)[0]]
        lam = 0.9 * min(c for c in caps if c > 0)
        ef = fused_energy(p, en, m, B, ctx, tau, lam); es = M2.energy_per_token(p, en, m, B, ctx, tau, lam, "SPATIAL")[0]
        print(f"{pn:24s}{B:>4}{s:>5} | {ef*1e3 if ef else float('nan'):>14.3f}  {es*1e3:>10.3f}   {((1-es/ef)*100 if ef else float('nan')):>6.1f}%")
