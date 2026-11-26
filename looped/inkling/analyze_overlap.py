"""Analyze Inkling 2 Mini expert overlap across loop passes from raw_routing.pt (expert_overlap.py output).

Metrics per (token, looped layer), top-8 sets S1, S2, S3 for passes 1-3:
  unique      |S1 ∪ S2 ∪ S3|  (floor 8 = identical, 24 = disjoint). Kill if mean > ~12.
  jaccard     |Si ∩ Sj| / |Si ∪ Sj| for pass pairs (1,2), (2,3), (1,3)
  exact       Si == Sj (identical expert set)
  overlap     |Si ∩ Sj| / 8
  p1_recall@K fraction of S2, S3 experts found in pass-1's top-K score ranking (pass-1 predictability: what a
              prefetcher that fetches pass-1's top-K would hit), K in {8, 12, 16, 24, 32}
  weight_cov  routing-weight mass of S2/S3 covered by S1 (overlap weighted by how much each expert matters)
Baselines: pass-1 of the SAME layer for the NEXT token (temporal reuse a normal MoE cache gets), and random
(expected unique for 3 independent 8-of-128 draws = 128*(1-(120/128)^3) = 22.5).
Also: batch-level union (unique experts touched per layer per pass vs all 3 passes over a window of tokens).

Usage: python -I analyze_overlap.py raw_routing.pt summary.json
"""
import json, sys
from collections import defaultdict
import torch

raw = torch.load(sys.argv[1], weights_only=False)
meta, items = raw["meta"], raw["items"]
ls, le, P, K, E = meta["loop_start"], meta["loop_end"], meta["num_passes"], meta["top_k"], meta["n_experts"]
KS = [8, 12, 16, 24, 32]


def onehot(idx):  # [T, 8] -> [T, E] bool
    m = torch.zeros(idx.shape[0], E, dtype=torch.bool)
    m.scatter_(1, idx.long(), True)
    return m


def stats(subset):
    acc = defaultdict(list)
    per_layer = defaultdict(lambda: defaultdict(list))
    for it in subset:
        for li in range(ls, le):
            L = it["layers"][li]
            M = [onehot(L[p]["idx"]) for p in range(P)]
            W = [torch.zeros(M[0].shape).scatter_(1, L[p]["idx"].long(), L[p]["w"].float()) for p in range(P)]
            uni = (M[0] | M[1] | M[2]).sum(1).float()
            acc["unique"].append(uni); per_layer[li]["unique"].append(uni)
            for a, b in [(0, 1), (1, 2), (0, 2)]:
                inter = (M[a] & M[b]).sum(1).float(); union = (M[a] | M[b]).sum(1).float()
                acc[f"jaccard_{a+1}{b+1}"].append(inter / union)
                acc[f"exact_{a+1}{b+1}"].append((inter == K).float())
                acc[f"overlap_{a+1}{b+1}"].append(inter / K)
                per_layer[li][f"overlap_{a+1}{b+1}"].append(inter / K)
            acc["exact_all3"].append(((M[0] & M[1] & M[2]).sum(1) == K).float())
            # pass-1 predictability: recall of later-pass experts in pass-1 top-K ranking
            r = L[0]["rank32"].long()
            for k in KS:
                R = torch.zeros_like(M[0]); R.scatter_(1, r[:, :k], True)
                for p in (1, 2):
                    acc[f"p1_recall@{k}_pass{p+1}"].append((R & M[p]).sum(1).float() / K)
            for p in (1, 2):
                wsum = W[p].sum(1)
                acc[f"weight_cov_pass{p+1}"].append((W[p] * M[0]).sum(1) / wsum)
            # temporal baseline: pass-1 of token t vs pass-1 of token t+1 (same layer)
            if M[0].shape[0] > 1:
                acc["temporal_overlap_p1"].append((M[0][:-1] & M[0][1:]).sum(1).float() / K)
                acc["temporal_unique3_p1"].append((M[0][:-2] | M[0][1:-1] | M[0][2:]).sum(1).float()
                                                  if M[0].shape[0] > 2 else torch.tensor([]))
            # window union: distinct experts loaded for a window of W tokens, pass-1 only vs all passes
            for w in (1, 4, 16, 64):
                T = M[0].shape[0]
                for s in range(0, T - w + 1, w):
                    u1 = M[0][s:s+w].any(0).sum().float()
                    u3 = (M[0][s:s+w] | M[1][s:s+w] | M[2][s:s+w]).any(0).sum().float()
                    acc[f"win{w}_unique_pass1"].append(u1.view(1)); acc[f"win{w}_unique_all3"].append(u3.view(1))
    out = {k: round(torch.cat(v).mean().item(), 4) for k, v in acc.items() if sum(x.numel() for x in v)}
    out["unique_hist"] = {int(u): c for u, c in zip(*torch.unique(torch.cat(acc["unique"]), return_counts=True))}
    out["unique_hist"] = {k: int(v) for k, v in out["unique_hist"].items()}
    out["n_token_layer"] = int(torch.cat(acc["unique"]).numel())
    out["per_layer_unique"] = {li: round(torch.cat(d["unique"]).mean().item(), 3) for li, d in per_layer.items()}
    out["per_layer_overlap_12"] = {li: round(torch.cat(d["overlap_12"]).mean().item(), 3) for li, d in per_layer.items()}
    out["per_layer_overlap_23"] = {li: round(torch.cat(d["overlap_23"]).mean().item(), 3) for li, d in per_layer.items()}
    return out


summary = {"meta": meta, "random_expected_unique": round(E * (1 - ((E - K) / E) ** 3), 2),
           "all": stats(items)}
for cat in sorted({it["cat"] for it in items}):
    summary[cat] = stats([it for it in items if it["cat"] == cat])
# prompt vs generated tokens
pre, gen = [], []
for it in items:
    n = it["n_prompt"]
    cut = lambda d, a, b: {li: {p: {k: v[a:b] for k, v in L.items()} for p, L in d.items()} for li, d in it["layers"].items()}
    pre.append({**it, "layers": cut(it["layers"], 0, n)}); gen.append({**it, "layers": cut(it["layers"], n, None)})
summary["prompt_tokens"] = stats(pre)
summary["generated_tokens"] = stats(gen)
summary["items"] = [{"cat": it["cat"], "prompt": it["prompt"], "n_prompt": it["n_prompt"], "n_total": it["n_total"],
                     "response_head": it["response"][:300]} for it in items]
json.dump(summary, open(sys.argv[2], "w"), indent=1)

a = summary["all"]
print(f"n token-layers {a['n_token_layer']}; random unique {summary['random_expected_unique']}")
for name in ["all", "chat", "code", "math", "tool", "prompt_tokens", "generated_tokens"]:
    s = summary[name]
    print(f"{name:17s} unique {s['unique']:.2f}  J12 {s['jaccard_12']:.3f} J23 {s['jaccard_23']:.3f} J13 {s['jaccard_13']:.3f}"
          f"  exact12 {s['exact_12']:.3f} exact23 {s['exact_23']:.3f} all3 {s['exact_all3']:.3f}"
          f"  ov12 {s['overlap_12']:.3f} ov23 {s['overlap_23']:.3f}  temporal_ov {s['temporal_overlap_p1']:.3f}")
print("pass-1 recall@K (pass2 / pass3):", {k: (a[f'p1_recall@{k}_pass2'], a[f'p1_recall@{k}_pass3']) for k in KS})
print("weight coverage by S1:", a["weight_cov_pass2"], a["weight_cov_pass3"])
print("window unique experts/layer, pass1 vs all3:", {w: (a[f'win{w}_unique_pass1'], a[f'win{w}_unique_all3']) for w in (1, 4, 16, 64)})
print("per-layer unique:", a["per_layer_unique"])
print("hist:", a["unique_hist"])
