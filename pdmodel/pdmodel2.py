"""
Extension of pdmodel.py:
 (1) SPATIAL+BC: nested-precision weight broadcast. When the decode partition and the prefill partition
     are on the same layer, ONE 8-bit (all bitplanes) weight stream feeds both; decode uses the top 4 planes.
     If the prefill partition makes r chunk-passes per decode step (r = tau / t_chunk), a decode pass can ride
     a prefill pass when r >= 1 (share = 1); otherwise only a fraction r of decode passes can (share = r).
     Chunk size C is optimised (small C -> more passes -> more sharing but more weight bytes per prefill token).
 (2) Energy: E = DRAM_bytes * e_byte + MACs * e_mac(format) + P_static * time.
     Compared at EQUAL WORK over EQUAL TIME: decode batch B at TPOT = tau, plus prefill demand
     lam = 0.9 x (lowest capacity among the schemes that are feasible), for one window of length tau.
     KV260 e_byte is calibrated from the INA260 measurements (0.155 J above idle per 8-bit token for
     ~0.75 GB of weights -> ~200 pJ/B incl. PL logic); other energy numbers are textbook-order assumptions.
"""
import copy, itertools, pdmodel as M

ENERGY = {  # e_byte (J/B), e_mac8 (J/MAC, INT8; FP16-activation MAC costs 2x), P_static (W)
 "ASIC-DynaCore-example": dict(e_byte=31e-12, e_mac8=0.25e-12, P_static=20.0),   # 28nm, HBM
 "ASIC-compute-rich":     dict(e_byte=31e-12, e_mac8=0.08e-12, P_static=30.0),   # 7nm, HBM
 "ASIC-edge-NPU":         dict(e_byte=40e-12, e_mac8=0.05e-12, P_static=0.5),    # 4nm, LPDDR5X
 "VCK5000-AIE":           dict(e_byte=150e-12, e_mac8=0.3e-12, P_static=40.0),   # 7nm AIE, DDR4 (static uncertain)
 "KV260-PL":              dict(e_byte=200e-12, e_mac8=0.5e-12, P_static=4.0),    # calibrated: idle 4.0 W measured
}

def spatial_bc_rate(p, m, B, ctx, tau, return_detail=False):
    dmacs, dby_full = M.decode_cost(m, B, ctx)
    w4 = M.wbytes(m, 4); dby_nonw = dby_full - w4
    if dby_nonw / p["BW"] > tau: return (0.0, None, None) if return_detail else 0.0
    best = (0.0, None, None)
    for k in range(1, p["G"]):
        f = k / p["G"]
        if dmacs / (f * p["R8"] * p["r16"] * p["e"] * M.E_PART) > tau: continue
        comp = (1 - f) * p["R8"] * p["e"] * M.E_PART
        for C in [64, 128, 256, 512, 1024, 2048, 4096, 8192]:
            pm, pb = M.prefill_cost(m, C, ctx, 8)
            mac_tok, by_tok = pm / C, pb / C
            lo, hi = 0.0, comp / mac_tok
            for _ in range(60):                       # bisection on prefill rate x
                x = (lo + hi) / 2
                share = min(1.0, x * tau / C)
                if share >= 1.0 and (C / x) > tau: share = 0.0   # (cannot happen; guard)
                bw_need = (w4 * (1 - share) + dby_nonw) / tau + x * by_tok
                if bw_need <= p["BW"]: lo = x
                else: hi = x
            x = lo * (1 - p["sw"] / tau)
            if x > best[0]: best = (x, f, C)
    return best if return_detail else best[0]

def energy_per_token(p, en, m, B, ctx, tau, lam, scheme):
    """Energy (J) per token (decode + prefill tokens) for one window tau serving B decode tokens + lam*tau prefill tokens."""
    Np = lam * tau
    dmacs, dby = M.decode_cost(m, B, ctx)
    w4, w8 = M.wbytes(m, 4), M.wbytes(m, 8)
    kvd = dby - w4
    if scheme == "TM":
        pm, pb = M.prefill_cost(m, max(Np, 1), ctx, 8)
        by = dby + (pb if Np > 0 else 0); mac8 = pm if Np > 0 else 0; mac16 = dmacs
    elif scheme == "FUSED":
        # pick the format the fused scheme would use: W8A8 if feasible else W4A16 (approximation: use W8A8 bytes/MAC energy)
        pm, pb = M.prefill_cost(m, max(Np, 1), ctx, 8)
        by = w8 + kvd + (pb - w8 if Np > 0 else 0); mac8 = dmacs + (pm if Np > 0 else 0); mac16 = 0
    elif scheme == "SPATIAL":
        C = M.C_SPATIAL
        pm, pb = M.prefill_cost(m, C, ctx, 8)
        by = dby + Np * pb / C; mac8 = Np * pm / C; mac16 = dmacs
    elif scheme == "SPATIAL+BC":
        C = max(64, min(8192, Np))                     # one aligned prefill pass per decode step -> share = 1
        pm, pb = M.prefill_cost(m, C, ctx, 8)
        share = min(1.0, Np / C) if Np > 0 else 0.0
        by = w4 * (1 - share) + kvd + Np * pb / C; mac8 = Np * pm / C; mac16 = dmacs
    E = by * en["e_byte"] + mac8 * en["e_mac8"] + mac16 * 2 * en["e_mac8"] + en["P_static"] * tau
    return E / (B + Np), by

def main():
    print("Capacity = prefill tok/s at TPOT <= slack x decode-alone; energy = J/token at equal work (lam = 0.9 x min feasible capacity)")
    print(f"{'platform':24s}{'B':>4}{'slk':>5} | {'TM':>8}{'FUSED':>8}{'SP':>8}{'SP+BC':>8} {'gainBC':>7} | {'E_TM':>9}{'E_FU':>9}{'E_SP':>9}{'E_BC':>9}  (mJ/tok)")
    ctx = 4096
    for pn, p in M.PLATFORMS.items():
        p = copy.deepcopy(p); m = M.model(p["model"]); en = ENERGY[pn]
        for B, s in itertools.product([p["Bs"][0], p["Bs"][-1]], [1.5, 3.0]):
            tau = s * M.t_decode_alone(p, m, B, ctx)
            caps = dict(TM=M.tm_rate(p, m, B, ctx, tau)[0], FUSED=M.fused_rate(p, m, B, ctx, tau)[0],
                        SPATIAL=M.spatial_rate(p, m, B, ctx, tau)[0], **{"SPATIAL+BC": spatial_bc_rate(p, m, B, ctx, tau)})
            other = max(caps["TM"], caps["FUSED"])
            feas = [v for v in caps.values() if v > 0]
            lam = 0.9 * min(feas)
            Es = {k: (energy_per_token(p, en, m, B, ctx, tau, lam, k)[0] * 1e3 if caps[k] > 0 else float('nan')) for k in caps}
            g = caps["SPATIAL+BC"] / other if other > 0 else float('inf')
            print(f"{pn:24s}{B:>4}{s:>5} | {caps['TM']:>8.0f}{caps['FUSED']:>8.0f}{caps['SPATIAL']:>8.0f}{caps['SPATIAL+BC']:>8.0f} {g:>7.2f} | "
                  f"{Es['TM']:>9.3f}{Es['FUSED']:>9.3f}{Es['SPATIAL']:>9.3f}{Es['SPATIAL+BC']:>9.3f}")
    p = M.PLATFORMS["KV260-PL"]; m = M.model("0.8B"); en = ENERGY["KV260-PL"]
    by8 = M.wbytes(m, 8) + 200 * M.kv_per_tok(m) + m["state"]
    print(f"\nKV260 calibration check: model 8-bit decode, 58 ms/token -> {(by8*en['e_byte'] + m['P']*en['e_mac8'] + en['P_static']*0.058):.3f} J/token (measured 0.388)")

if __name__ == "__main__":
    main()
