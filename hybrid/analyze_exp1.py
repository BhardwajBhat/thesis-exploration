"""Summarize exp1 and project nested self-speculation speedup on the KV260 from measured ms/token."""
import json, statistics, sys, collections

# KV260 measured (INA260 runs): ms/token at weight/activation bits
T = {8: 58.0, 6: 49.9, 4: 38.6, 3: 35.1}
T_EXTRA_TOK = 2.0        # extra compute per additional verified token: 0.75e9 MAC / ~374e9 MAC/s
STATE_BYTES_8 = 4.72e6   # 18 DeltaNet layers x 16 heads x 128 x 128 elements, 1 B each at 8 bit
BW = 12e9                # KV260 effective DRAM B/s

def speedup(acc, k, b, snap_bits=4):
    t_draft = k * T[b]
    t_verify = T[8] + T_EXTRA_TOK * k
    t_snap = k * STATE_BYTES_8 * snap_bits / 8 / BW * 1e3     # one state snapshot per draft token (ms)
    return (acc + 1) * T[8] / (t_draft + t_verify + t_snap)

for path in sys.argv[1:]:
    r = json.load(open(path)); P = r["prompts"]
    print(f"\n=== {path}  (weights: {r['wfmt']}, {len(P)} prompt runs x {r['gen']} tokens)")
    print("Acceptance (mean accepted tokens per round; draft starts from the 8-bit state)")
    print(f"{'bits':>4} {'p_match(own state)':>19} {'k=2':>6} {'k=4':>6} {'k=8':>6} | speedup on KV260 at k=2/4/8")
    for b in r["bits"]:
        pA = statistics.mean(p["A"][str(b)]["p_match"] for p in P)
        acc = {k: statistics.mean(p["B"][str(b)][str(k)] for p in P) for k in (2, 4, 8)}
        sp = "/".join(f"{speedup(acc[k], k, b):.2f}" for k in (2, 4, 8))
        print(f"{b:>4} {pA:>19.2f} {acc[2]:>6.2f} {acc[4]:>6.2f} {acc[8]:>6.2f} | {sp}")
    print("By task kind, 4-bit draft, k=4 accepted:")
    byk = collections.defaultdict(list)
    for p in P: byk[(p["kind"], p["think"])].append(p["B"]["4"]["4"])
    print("   " + "  ".join(f"{k}{'+think' if t else ''}={statistics.mean(v):.2f}" for (k, t), v in byk.items()))
    print("State snapshot precision (teacher-forced 32 tokens after the snapshot, 8-bit weights): top-1 agreement / mean KL")
    keys = list(P[0]["C"].keys())
    print("   " + "  ".join(f"{k}={statistics.mean(p['C'][k]['top1'] for p in P):.2f}/{statistics.mean(p['C'][k]['kl'] for p in P):.3f}" for k in keys))
    print("   worst prompt: " + "  ".join(f"{k}={min(p['C'][k]['top1'] for p in P):.2f}" for k in keys))
s = r["sizes"]
print(f"\nSizes: DeltaNet state {s['state_elems']/1e6:.2f} M elements/request ({s['state_elems']/1e6:.1f} MB at 8 bit), "
      f"conv state {s['conv_elems']/1e3:.0f} K elements, full-attention KV {s['kv_elems_per_token']} elements/token. "
      f"Snapshot write at 4 bit on KV260: {STATE_BYTES_8*0.5/BW*1e3:.2f} ms vs {T[8]} ms per 8-bit token.")
