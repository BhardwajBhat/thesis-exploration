"""Edge-STRIDE kill test step 2: replay recorded single-agent d1 traces on a KV260-like batch-1 accelerator.

Workload from the traces (per turn): decode tokens D, the fraction of the turn at which the probe dispatched (d/D),
whether the turn calls a tool, whether the guess was exactly right (spec_used) and fork-eligible (no visible text
before the call), and the new input tokens the NEXT turn must prefill (tool result chars / 4 + call tokens).
Everything is re-costed on the edge device (time in seconds):

  t1        batch-1 decode step (KV260 Qwen3.5-0.8B at 8/6: 58 ms)
  s         batch-2 step overhead: a step with two streams costs t1*(1+s)  (the key hardware unknown; sweep)
  prate     prefill throughput, tokens/s (sweep; token-serial GEMV prefill would be ~17 tok/s)
  T         tool latency (2 s floor as in the GPU runs; sweep)
  probe     24-token side decode at the probe point (shares weight reads like a fork: main slowed by s)

Policies, depth <= 1 (at most two streams, one unverified turn in flight):
  serial    D*t1 + prefill + T per turn
  spork     probe + early tool dispatch; next turn starts at max(main end, tool return)
  la        spork + fork: when the speculative tool returns, prefill its result and decode the next turn in a
            second stream; commit iff exact match and fork-eligible, else discard (wasted work slowed main)
Energy: P_idle 4.0 W whenever the board is on; +P_dyn 2.69 W while computing (0.388 J/token at 58 ms measured);
optional dyn_b2 multiplier on the dynamic power during two-stream steps.
Output: total-time and total-energy speedup vs serial, ratio of sums over tasks, bootstrap 95% CI.

Usage: python kv260_replay.py 'results/tau2_*/d1' ['results/bfcl_*/d1' ...]
"""
import glob, json, random, sys

P_IDLE, P_DYN = 4.0, 0.388 / 0.058 - 4.0
PROBE_TOK = 24


def load(path):
    d = json.load(open(path))
    turns = []
    for t in d["turns"]:
        if t.get("error") or t.get("main_end_ms") is None:
            return None
        D = max(1, t.get("total_decode_tokens") or 1)
        disp = t.get("spec_tool_dispatch_ms") or t.get("probe_dispatch_ms")
        frac = min(1.0, disp / t["main_end_ms"]) if (disp and t.get("spec_dispatched")) else None
        tool = bool(t.get("baseline_has_tool_call"))
        txt = t.get("baseline_text") or ""
        import re
        vis = re.sub(r"^\s*<think>.*?</think>", "", txt, count=1, flags=re.DOTALL).split("<tool_call>")[0].strip()
        acc = bool(t.get("spec_used")) and tool
        new_in = (t.get("tool_result_len") or 0) / 4 + (t.get("tool_call_tokens") or 0) + 10
        turns.append(dict(D=D, frac=frac, tool=tool, acc=acc, fork_ok=acc and vis == "",
                          disp=frac is not None, new_in=new_in, mode=t.get("mode")))
    for i in range(1, len(turns)):  # forced-conclusion turn cannot be forked into
        if turns[i]["mode"] == "forced_conclusion":
            turns[i - 1]["fork_ok"] = False
    return turns


def run(turns, policy, t1, s, prate, T, dyn_b2=1.0):
    """Return (wall seconds, energy joules)."""
    wall = 0.0
    active1 = active2 = 0.0  # compute seconds at one / two streams
    carry_tokens = 0          # tokens of the current turn already decoded by a committed fork
    carry_prefilled = False
    t2 = t1 * (1 + s)
    first_prefill = 0.0
    for i, x in enumerate(turns):
        nxt = turns[i + 1] if i + 1 < len(turns) else None
        P_next = (x["new_in"] / prate) if nxt else 0.0
        D_rem = max(0, x["D"] - carry_tokens)
        # prefill of this turn's new input happens at its start unless a committed fork already did it
        pre = 0.0 if (carry_prefilled or i == 0) else turns[i - 1]["new_in"] / prate
        carry_tokens, carry_prefilled = 0, False
        start = wall
        wall += pre; active1 += pre
        if policy == "serial" or not x["disp"]:
            wall += D_rem * t1; active1 += D_rem * t1
            if x["tool"]:
                wall += T
            continue
        # tokens before the probe point (of the remaining)
        tok_before = max(0, int(x["frac"] * x["D"]) - (x["D"] - D_rem))
        wall += tok_before * t1; active1 += tok_before * t1
        # probe: 24 steps with two streams; main advances 24 tokens meanwhile (capped by what is left)
        pt = min(PROBE_TOK, max(0, D_rem - tok_before))
        wall += PROBE_TOK * t2; active2 += PROBE_TOK * t2
        left = max(0, D_rem - tok_before - pt)
        tool_back = wall + T  # speculative tool result returns
        if policy == "spork" or nxt is None:
            # main finishes alone
            wall += left * t1; active1 += left * t1
            if x["tool"]:
                wall = max(wall, tool_back) if x["acc"] else wall + T
            continue
        # la: main decodes alone until the tool returns, then the fork prefills and decodes alongside
        alone = min(left, int(T / t1))
        wall += alone * t1; active1 += alone * t1
        left -= alone
        if left <= 0:
            # main finished before the tool returned: same as spork (no overlap)
            if x["tool"]:
                wall = max(wall, tool_back) if x["acc"] else wall + T
            continue
        # fork prefill (compute-bound; main stalls)
        wall += P_next; active1 += P_next
        # both decode until main ends
        wall += left * t2; active2 += left * t2
        fork_tok = min(nxt["D"], left)
        if x["fork_ok"]:
            carry_tokens, carry_prefilled = fork_tok, True
        elif x["tool"]:
            # wasted fork; next turn needs the real tool result
            wall = max(wall, tool_back) if x["acc"] else wall + T
    energy = P_IDLE * wall + P_DYN * (active1 + dyn_b2 * active2)
    return wall, energy


def speedups(tasks, policy, **kw):
    per = [(run(t, "serial", **kw), run(t, policy, **kw)) for t in tasks]
    def ratio(idx):
        S = sum(per[i][0][0] for i in idx); P = sum(per[i][1][0] for i in idx)
        SE = sum(per[i][0][1] for i in idx); PE = sum(per[i][1][1] for i in idx)
        return S / P, SE / PE
    full = ratio(range(len(per)))
    rng = random.Random(0)
    boots = [ratio([rng.randrange(len(per)) for _ in per]) for _ in range(2000)]
    lo = sorted(b[0] for b in boots)[50]; hi = sorted(b[0] for b in boots)[1949]
    elo = sorted(b[1] for b in boots)[50]; ehi = sorted(b[1] for b in boots)[1949]
    return full, (lo, hi), (elo, ehi)


def main():
    for pat in sys.argv[1:]:
        tasks = [t for t in (load(f) for f in sorted(glob.glob(pat + "/*.json"))) if t]
        n_tool = sum(x["tool"] for t in tasks for x in t)
        n_acc = sum(x["acc"] for t in tasks for x in t)
        n_fork = sum(x["fork_ok"] for t in tasks for x in t)
        Ds = sorted(x["D"] for t in tasks for x in t)
        print(f"\n### {pat}: {len(tasks)} tasks, {sum(len(t) for t in tasks)} turns, {n_tool} tool turns, "
              f"exact {n_acc / max(1, n_tool):.2f}, fork-eligible {n_fork / max(1, n_tool):.2f}, "
              f"median decode tokens/turn {Ds[len(Ds) // 2]}")
        print("policy  s     prate  T    | time speedup [95% CI]   | energy saving (1 - E/E_serial) [CI]")
        for policy in ("spork", "la"):
            for s in ((0.0,) if policy == "spork" else (0.2, 0.5, 1.0)):
                for prate in (17, 100, 500):
                    for T in (0.5, 2.0):
                        (sp, en), (lo, hi), (elo, ehi) = speedups(tasks, policy, t1=0.058, s=s, prate=prate, T=T)
                        print(f"{policy:6s}  {s:<4} {prate:>5}  {T:<4} | {sp:5.3f} [{lo:.3f}, {hi:.3f}]     "
                              f"| {1 - 1 / en:+.1%} [{1 - 1 / elo:+.1%}, {1 - 1 / ehi:+.1%}]")


if __name__ == "__main__":
    main()
