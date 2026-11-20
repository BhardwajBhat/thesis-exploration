"""Aggregate KV kill-test runs: per-config accuracy (Wilson 95% CI) and paired differences vs
the bf16 seed-0 baseline (paired bootstrap 95% CI over identical items).

Outputs (kv_killtest/results/):
  per_item.csv    one row per (benchmark, model, config, item): correct/reward, error type
  summary.csv     one row per (benchmark, model, config): n, acc, Wilson CI, paired diff vs bf16_s0 + CI, flips
  summary.json    same, nested; plus the kill-criteria verdict inputs
"""
import csv, glob, json, math, os, re, sys
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS, OUT = os.path.join(HERE, "runs"), os.path.join(HERE, "results")
TAU2_SIMS = os.path.expanduser(os.getenv("TAU2_SIMS", "~/clod/tau2-bench/data/simulations"))
os.makedirs(OUT, exist_ok=True)
rows = []  # dict(bench, model, cfg, item, score, err)

# ---------------- BFCL ----------------
for run in sorted(glob.glob(os.path.join(RUNS, "bfcl_*"))):
    m = re.match(r"bfcl_(\w+?)_(\w+_s\d)$", os.path.basename(run))
    if not m: continue
    model, cfg = m.groups()
    for res in glob.glob(os.path.join(run, "result", "*", "*", "*_result.json")):
        cat = re.search(r"BFCL_v4_(.+)_result\.json", res).group(1)
        ids = [json.loads(l)["id"] for l in open(res)]
        sc = res.replace("/result/", "/score/").replace("_result.json", "_score.json")
        if not os.path.exists(sc): continue
        lines = [json.loads(l) for l in open(sc)]
        failed = {e["id"]: e.get("error", {}).get("error_type", "?") if isinstance(e.get("error"), dict) else "?"
                  for e in lines[1:]}
        for i in ids:
            if cat.startswith("memory") and "prereq" in i: continue
            rows.append(dict(bench=f"bfcl_{cat}", model=model, cfg=cfg, item=i,
                             score=0.0 if i in failed else 1.0, err=failed.get(i, "")))

# ---------------- tau2 ----------------
for res in glob.glob(os.path.join(TAU2_SIMS, "kvkill_tau2_*", "results.json")):
    m = re.match(r"kvkill_tau2_(\w+?)_(\w+_s\d)_(\w+)$", os.path.basename(os.path.dirname(res)))
    if not m: continue
    model, cfg, domain = m.groups()
    for s in json.load(open(res))["simulations"]:
        ri = s.get("reward_info") or {}
        db = (ri.get("db_check") or {}).get("db_match")
        rows.append(dict(bench=f"tau2_{domain}", model=model, cfg=cfg, item=str(s["task_id"]),
                         score=float(ri.get("reward") or 0.0), err=s.get("termination_reason", "")))
        if db is not None:
            rows.append(dict(bench=f"tau2_{domain}_dbonly", model=model, cfg=cfg, item=str(s["task_id"]),
                             score=1.0 if db else 0.0, err=""))

with open(os.path.join(OUT, "per_item.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["bench", "model", "cfg", "item", "score", "err"]); w.writeheader(); w.writerows(rows)

# ---------------- stats ----------------
def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"),) * 2
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h

def paired_boot(a, b, B=10000, seed=0):
    d = np.asarray(a) - np.asarray(b)
    idx = np.random.default_rng(seed).integers(0, len(d), (B, len(d)))
    bs = d[idx].mean(1)
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)

by = defaultdict(dict)
for r in rows: by[(r["bench"], r["model"], r["cfg"])][r["item"]] = r["score"]
groups = defaultdict(dict)
for (b, m, c), d in by.items(): groups[(b, m)][c] = d
# pooled BFCL multi-turn (base + long_context) as one paired set
for m in {m for (b, m) in groups}:
    pooled = defaultdict(dict)
    for cat in ("bfcl_multi_turn_base", "bfcl_multi_turn_long_context"):
        for c, d in groups.get((cat, m), {}).items():
            pooled[c].update({f"{cat}:{k}": v for k, v in d.items()})
    if pooled: groups[("bfcl_multi_turn_pooled", m)] = dict(pooled)

summary = []
for (b, m), cfgs in sorted(groups.items()):
    base = cfgs.get("bf16_s0")
    for c, d in sorted(cfgs.items()):
        n = len(d); k = sum(d.values()); lo, hi = wilson(k, n)
        row = dict(bench=b, model=m, cfg=c, n=n, acc=round(100 * k / n, 2),
                   ci_lo=round(100 * lo, 2), ci_hi=round(100 * hi, 2))
        if base and c != "bf16_s0":
            common = sorted(set(d) & set(base))
            dm, dlo, dhi = paired_boot([d[i] for i in common], [base[i] for i in common])
            row.update(n_paired=len(common), diff_vs_bf16_s0=round(100 * dm, 2),
                       diff_ci_lo=round(100 * dlo, 2), diff_ci_hi=round(100 * dhi, 2),
                       flips_up=sum(d[i] > base[i] for i in common),
                       flips_down=sum(d[i] < base[i] for i in common))
        summary.append(row)

keys = ["bench", "model", "cfg", "n", "acc", "ci_lo", "ci_hi", "n_paired", "diff_vs_bf16_s0",
        "diff_ci_lo", "diff_ci_hi", "flips_up", "flips_down"]
with open(os.path.join(OUT, "summary.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(summary)
json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)

# error-type breakdown per config (BFCL), for the "which turns fail" question
err = defaultdict(lambda: defaultdict(int))
for r in rows:
    if r["bench"].startswith("bfcl") and r["score"] == 0: err[(r["bench"], r["model"], r["cfg"])][r["err"]] += 1
with open(os.path.join(OUT, "bfcl_error_types.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["bench", "model", "cfg", "error_type", "count"])
    for (b, m, c), d in sorted(err.items()):
        for e, n in sorted(d.items(), key=lambda x: -x[1]): w.writerow([b, m, c, e, n])

for s in summary:
    extra = (f"  diff {s['diff_vs_bf16_s0']:+.1f} [{s['diff_ci_lo']:+.1f},{s['diff_ci_hi']:+.1f}] "
             f"flips +{s['flips_up']}/-{s['flips_down']}") if "diff_vs_bf16_s0" in s else ""
    print(f"{s['bench']:32s} {s['model']:3s} {s['cfg']:9s} n={s['n']:4d} acc {s['acc']:5.1f} "
          f"[{s['ci_lo']:.1f},{s['ci_hi']:.1f}]{extra}")
