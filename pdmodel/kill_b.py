"""
Kill test for front-runner B (energy leg): does the spatial-partition energy saving survive
(1) static power, (2) the fraction of an agent task where prefill and decode are actually concurrent,
(3) idle power during tool waits?

Step 1: per concurrent window, J/token of SPATIAL vs TM vs FUSED(best format), split static/dynamic.
        TM here is the NESTED-precision TM the user's KV260 design already supports: one stored copy,
        decode reads 4 bit-planes, prefill reads 8, switched per step (no second weight copy needed).
Step 2: task level. A single-user agent turn = prefill of tool output (P tokens) + decode (D tokens)
        + tool wait (W s at idle power). Only a fraction phi of the prefill can overlap a decode
        (look-ahead speculation, parallel sub-agents); the rest runs alone on the whole array,
        identically for all schemes. Saving at task level = window saving x overlapped share of energy.
"""
import pdmodel as M, pdmodel2 as M2, fusedfix as FF   # fusedfix prints its table on import
import itertools

def window(pn, B, slack, ctx=4096):
    p = M.PLATFORMS[pn]; m = M.model(p["model"]); en = M2.ENERGY[pn]
    tau = slack * M.t_decode_alone(p, m, B, ctx)
    caps = dict(TM=M.tm_rate(p, m, B, ctx, tau)[0], FUSED=M.fused_rate(p, m, B, ctx, tau)[0], SPATIAL=M.spatial_rate(p, m, B, ctx, tau)[0])
    lam = 0.9 * min(c for c in caps.values() if c > 0)
    Np = lam * tau
    E = dict(TM=M2.energy_per_token(p, en, m, B, ctx, tau, lam, "TM")[0],
             SPATIAL=M2.energy_per_token(p, en, m, B, ctx, tau, lam, "SPATIAL")[0],
             FUSED=FF.fused_energy(p, en, m, B, ctx, tau, lam))
    stat = en["P_static"] * tau / (B + Np)
    return tau, Np, caps, E, stat

print("\n" + "=" * 100)
print("STEP 1: per concurrent window (ctx 4096). J/token in mJ; 'static' = P_static*tau/tokens, same for all schemes")
print(f"{'platform':16s}{'B':>3}{'slk':>5}{'tau ms':>8}{'pre tok':>8} | {'TM':>7}{'FUSED':>7}{'SPAT':>7} {'static':>7} | {'SP vs TM':>9}{'SP vs FU':>9} {'static share':>13}")
for pn in ["KV260-PL", "ASIC-edge-NPU"]:
    for B, s in itertools.product([1, 2, 4], [1.5, 3.0]):
        tau, Np, caps, E, stat = window(pn, B, s)
        best_other = min(v for v in (E["TM"], E["FUSED"]) if v)
        print(f"{pn:16s}{B:>3}{s:>5}{tau*1e3:>8.1f}{Np:>8.0f} | {E['TM']*1e3:>7.1f}{(E['FUSED'] or float('nan'))*1e3:>7.1f}{E['SPATIAL']*1e3:>7.1f} {stat*1e3:>7.1f} | "
              f"{(1-E['SPATIAL']/E['TM'])*100:>8.1f}%{(1-E['SPATIAL']/E['FUSED'])*100 if E['FUSED'] else float('nan'):>8.1f}% {stat/E['SPATIAL']*100:>12.0f}%")

# ---------------- step 2: task level on KV260 ----------------
# Measured KV260 anchors: 4/4 decode 38.6 ms/token, 0.247 J/token (incl. 4.0 W idle); idle 4.0 W.
p = M.PLATFORMS["KV260-PL"]; m = M.model("0.8B"); en = M2.ENERGY["KV260-PL"]
E_DEC = 0.247; T_DEC = 0.0386
pm, pb = M.prefill_cost(m, 2048, 4096, 8)
T_PRE_TOK = max(pm / (p["R8"] * p["e"]), pb / p["BW"]) / 2048                      # whole-array W8 prefill, s/token
E_PRE_TOK = (pb * en["e_byte"] + pm * en["e_mac8"]) / 2048 + en["P_static"] * T_PRE_TOK
win_save = 0.0                                                                     # best KV260 window saving in step 1
for B, s in itertools.product([1, 2, 4], [1.5, 3.0]):
    Ew = window("KV260-PL", B, s)[3]
    win_save = max(win_save, 1 - Ew["SPATIAL"] / min(Ew["TM"], Ew["FUSED"] or 9))
print("\n" + "=" * 100)
print(f"STEP 2: KV260 single-user agent task. prefill {T_PRE_TOK*1e3:.1f} ms/token ({E_PRE_TOK*1e3:.0f} mJ/token), decode {T_DEC*1e3:.1f} ms/token ({E_DEC*1e3:.0f} mJ/token)")
print(f"        window saving used = {win_save*100:.1f}% (best KV260 case from step 1), applied only to the overlapped part")
print(f"{'P tok':>6}{'D tok':>6}{'wait s':>7}{'phi':>5} | {'E/turn J':>9}{'idle-wait share':>16}{'task saving':>12}")
for P, D, W, phi in itertools.product([1000, 4000], [100, 300], [1.0, 5.0], [0.25, 0.5, 1.0]):
    e_pre, e_dec, e_wait = P * E_PRE_TOK, D * E_DEC, W * en["P_static"]
    t_pre, t_dec = P * T_PRE_TOK, D * T_DEC
    overlap_t = min(phi * t_pre, t_dec)                    # concurrency limited by the shorter phase
    e_overlap = overlap_t / t_pre * e_pre + overlap_t / t_dec * e_dec
    tot = e_pre + e_dec + e_wait
    print(f"{P:>6}{D:>6}{W:>7}{phi:>5} | {tot:>9.1f}{e_wait/tot*100:>15.0f}%{win_save*e_overlap/tot*100:>11.1f}%")
