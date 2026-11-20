import pdmodel as M, copy
plats = ["ASIC-DynaCore-example","ASIC-compute-rich","ASIC-edge-NPU","VCK5000-AIE","KV260-PL"]
print("Spatial gain over best(TM,FUSED) vs TPOT slack and partition granularity G (ctx=4096)")
print(f"{'platform':24s}{'B':>4}{'G':>4} | " + " ".join(f"s={s:<4}" for s in [1.2,1.5,2.0,3.0,5.0]))
for pn in plats:
    for G in [4, 16, 64]:
        p = copy.deepcopy(M.PLATFORMS[pn]); p["G"] = G; m = M.model(p["model"])
        for B in [p["Bs"][0], p["Bs"][-1]]:
            out = []
            for s in [1.2,1.5,2.0,3.0,5.0]:
                tau = s * M.t_decode_alone(p, m, B, 4096)
                tm = M.tm_rate(p, m, B, 4096, tau)[0]; fu = M.fused_rate(p, m, B, 4096, tau)[0]
                sp = M.spatial_rate(p, m, B, 4096, tau)[0]; o = max(tm, fu)
                out.append(f"{(sp/o if o>0 else float('inf')):6.2f}")
            print(f"{pn:24s}{B:>4}{G:>4} | " + " ".join(out))
print("\nTwo devices incl. FUSED on both (prefill tok/s), tau = 2x static-decode time")
for pn in ["ASIC-DynaCore-example","ASIC-compute-rich"]:
    p = M.PLATFORMS[pn]; m = M.model(p["model"])
    for Bt, ctx in [(16,2048),(64,16384),(128,16384)]:
        tau = 2*M.t_decode_alone(p, m, Bt, ctx)
        pm,pb = M.prefill_cost(m, 2048, ctx, 8)
        st = min(2048/max(pm/(p["R8"]*p["e"]), pb/p["BW"]), 16e9/M.kv_per_tok(m))
        dyn = 2*M.spatial_rate(p,m,Bt//2,ctx,tau)[0]; tmb = 2*M.tm_rate(p,m,Bt//2,ctx,tau)[0]; fub = 2*M.fused_rate(p,m,Bt//2,ctx,tau)[0]
        print(f"  {pn:24s} B={Bt:<4} ctx={ctx:<6} static={st:8.0f} spatial={dyn:8.0f} TM={tmb:8.0f} FUSED={fub:8.0f}")
