"""
Experiment 1 for direction #1 (hybrid-model state hardware): is nested self-speculation worth it on a
hybrid Gated-DeltaNet model, and what do recurrent-state snapshots cost?

Model: Qwen3.5-0.8B (18 Gated DeltaNet layers + 6 full-attention layers).
Weights: emulated NESTED precision. One 8-bit symmetric per-group code (group 128 along the input dim)
per Linear; lower precisions read only the top b bit-planes of that same code (mid-point reconstruction).
This is round-to-nearest, NOT the user's calibrated KV260 codes, so 4/3-bit numbers are pessimistic.
lm_head stays BF16 (tied embedding). Activations stay BF16 (weight-only emulation).

A. Acceptance, draft with its OWN state: teacher-force the b-bit model over the 8-bit greedy trajectory;
   at each position, does its argmax equal the 8-bit token? Run length of matches = accepted tokens.
B. Acceptance, draft STARTS FROM THE 8-BIT STATE (true self-speculation): at sampled positions, copy the
   8-bit cache, swap weights to b bits, draft k tokens autoregressively, count matches.
C. Snapshot precision: at sampled positions, replace the DeltaNet recurrent states with a nested b-bit
   version (per-row scale) and continue 8-bit greedy for 32 tokens; agreement with the uninterrupted run.
D. Sizes and a KV260 speedup model from the user's measured ms/token.
"""
import argparse, copy, json, os, statistics, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

PROMPTS = [
    # tool calling (BFCL-like)
    ("tool", "You have access to these functions:\n" + json.dumps([
        {"name": "get_weather", "description": "Get current weather", "parameters": {"type": "object", "properties": {
            "city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}},
        {"name": "search_flights", "description": "Search flights", "parameters": {"type": "object", "properties": {
            "origin": {"type": "string"}, "destination": {"type": "string"}, "date": {"type": "string"}}, "required": ["origin", "destination", "date"]}}])
     + "\nRespond with a JSON function call only.\nUser: Find me a flight from Los Angeles to Tokyo on 2026-12-20, and also tell me the weather in Tokyo in celsius."),
    ("tool", "Tools: [{\"name\": \"run_sql\", \"parameters\": {\"query\": \"string\"}}, {\"name\": \"send_email\", \"parameters\": {\"to\": \"string\", \"subject\": \"string\", \"body\": \"string\"}}]\n"
             "Task: count the orders placed in November 2026 in table `orders` (column created_at), then email the count to ops@example.com. Output the tool calls as JSON, one per line."),
    ("tool", "The previous tool returned: {\"results\": [{\"title\": \"Gated DeltaNet\", \"year\": 2024, \"citations\": 512}, {\"title\": \"Mamba-2\", \"year\": 2024, \"citations\": 1490}, {\"title\": \"RWKV-7\", \"year\": 2025, \"citations\": 233}]}\n"
             "Summarize the results for the user in two sentences and recommend which paper to read first."),
    # code
    ("code", "Write a Python function that parses an Apache access log line into a dict with ip, timestamp, method, path, status and size. Include a short docstring."),
    ("code", "Explain what this does and fix the bug:\n```python\ndef mean(xs):\n    total = 0\n    for x in xs:\n        total += x\n    return total / len(xs) if xs else total / 0\n```"),
    ("code", "Write a bash one-liner that finds the 10 largest files under the current directory, then explain each part."),
    # chat / reasoning
    ("chat", "Explain to a new graduate student why decode in LLM inference is memory-bandwidth bound while prefill is compute bound."),
    ("chat", "Plan a 3-day itinerary for Kyoto in December for someone who likes temples and food. Be concise."),
    ("math", "A train leaves at 9:40 and arrives at 13:05, stopping twice for 12 minutes each. What is the moving time? Show the steps."),
    ("math", "Solve for x: 3(x - 4) + 2 = 5x - 10. Show the steps."),
    ("chat", "Write a short, polite email declining a meeting invitation because of a conflict, and propose two alternative times."),
    ("chat", "List five practical tips for reducing energy use of an always-on home server."),
]

# ---------------- nested quantization ----------------
def reduce_code(code, b, fmt):
    """b-bit view of an 8-bit symmetric code (|code| <= 127).
    bp : two's-complement top-b bit-planes, midpoint reconstruction
    sm : sign-magnitude top bit-planes (sign + b-1 magnitude bits), midpoint, zero bucket decodes to 0
    rtn: round to nearest at b bits (not a pure bit-plane read; reference)"""
    if b >= 8: return code
    step = 2 ** (8 - b)
    if fmt == "bp":
        return torch.floor(code / step) * step + step / 2
    if fmt == "sm":
        mstep = 2 ** (7 - (b - 1)); mag = torch.floor(code.abs() / mstep)
        return torch.sign(code) * torch.where(mag > 0, mag * mstep + mstep / 2, torch.zeros_like(mag))
    return torch.round(code / step) * step

def nested_views(w, bits_list, group=128, fmt="bp"):
    out_f, in_f = w.shape
    wf = w.float().reshape(out_f, in_f // group, group)
    scale = wf.abs().amax(-1, keepdim=True).clamp(min=1e-8) / 127.0
    code = torch.clamp(torch.round(wf / scale), -127, 127)
    views = {}
    for b in bits_list:
        q = reduce_code(code, b, fmt)
        views[b] = (q * scale).reshape(out_f, in_f).to(w.dtype)
    return views

class Nested:
    def __init__(self, model, bits_list, fmt="bp"):
        self.mods, self.views = [], []
        for n, mod in model.named_modules():
            if isinstance(mod, torch.nn.Linear) and "lm_head" not in n and mod.in_features % 128 == 0:
                self.mods.append(mod); self.views.append(nested_views(mod.weight.data, bits_list, fmt=fmt))
        self.cur = None
    def set(self, b):
        if b == self.cur: return
        for mod, v in zip(self.mods, self.views): mod.weight.data = v[b]
        self.cur = b

# ---------------- helpers ----------------
def enc(tok, text, think):
    r = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                return_tensors="pt", enable_thinking=think)
    return (r["input_ids"] if hasattr(r, "keys") else r).cuda()

@torch.no_grad()
def greedy(m, ids, n, cache=None):
    """incremental greedy decode; returns tokens and the cache after consuming ids + tokens[:-1]"""
    out = m(ids, past_key_values=cache, use_cache=True)
    cache, toks = out.past_key_values, []
    nxt = out.logits[:, -1].argmax(-1, keepdim=True)
    for _ in range(n):
        toks.append(nxt.item())
        out = m(nxt, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
    return toks, cache

def recurrent_tensors(cache):
    """(layer, key) for every DeltaNet recurrent-state tensor in the cache (stored in a dict per layer)"""
    res = []
    for i, L in enumerate(cache.layers):
        d = getattr(L, "recurrent_states", None)
        if isinstance(d, dict):
            res += [(i, k) for k, v in d.items() if isinstance(v, torch.Tensor)]
    return res

def nested_state(t, b, fmt="bp", group=128):
    """b-bit view of an 8-bit symmetric code of the state. fmt='bp': top b bit-planes, midpoint
    reconstruction (what nested storage gives for free); fmt='rtn': round-to-nearest at b bits."""
    if b >= 16: return t
    sh = t.shape
    tf = t.float().reshape(*sh[:-1], sh[-1] // group, group)
    scale = tf.abs().amax(-1, keepdim=True).clamp(min=1e-12) / 127.0
    code = torch.clamp(torch.round(tf / scale), -127, 127)
    code = reduce_code(code, b, fmt)
    return (code * scale).reshape(sh).to(t.dtype)

def runs(match, k):
    """mean accepted tokens (run of matches capped at k) over all start positions"""
    n = len(match); acc = []
    for s in range(n):
        r = 0
        while r < k and s + r < n and match[s + r]: r += 1
        acc.append(r)
    return statistics.mean(acc)

C_KEYS = ["rtn-8", "rtn-6", "bp-6", "sm-6", "rtn-4", "bp-4", "sm-4", "sm-3"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", type=int, default=160); ap.add_argument("--bits", default="6,4,3")
    ap.add_argument("--stride", type=int, default=8); ap.add_argument("--kmax", type=int, default=8)
    ap.add_argument("--wfmt", default="bp", choices=["bp", "sm", "rtn"])
    ap.add_argument("--state-group", type=int, default=128)
    ap.add_argument("--nprompts", type=int, default=len(PROMPTS))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results", "exp1.json"))
    a = ap.parse_args()
    bits = [int(x) for x in a.bits.split(",")]
    torch.manual_seed(0)
    mid = "Qwen/Qwen3.5-0.8B"
    tok = AutoTokenizer.from_pretrained(mid)
    m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16).cuda().eval()
    nest = Nested(m, [8] + bits, fmt=a.wfmt)
    res = dict(model=mid, gen=a.gen, bits=bits, wfmt=a.wfmt, state_group=a.state_group, prompts=[])
    t0 = time.time()
    for pi, (kind, text) in enumerate(PROMPTS[:a.nprompts]):
        for think in (False, True) if kind in ("math", "code") else (False,):
            ids = enc(tok, text, think)
            nest.set(8)
            ref, _ = greedy(m, ids, a.gen)
            full = torch.cat([ids, torch.tensor([ref], device="cuda")], 1)
            P = ids.shape[1]
            rec = dict(kind=kind, think=think, prompt_len=P, A={}, B={}, C={})
            # A: own-state teacher forcing
            for b in bits:
                nest.set(b)
                with torch.no_grad(): lg = m(full[:, :-1]).logits[0, P - 1:]
                match = (lg.argmax(-1) == full[0, P:]).tolist()
                rec["A"][b] = dict(p_match=statistics.mean(match), acc={k: runs(match, k) for k in (2, 4, 8)})
            # B and C: sampled starts along the trajectory, 8-bit state
            starts = list(range(0, a.gen - max(a.kmax, 32), a.stride))
            nest.set(8)
            with torch.no_grad():
                out = m(ids, use_cache=True); cache = out.past_key_values
            pos, Bacc, Cagree = 0, {b: [] for b in bits}, {k: [] for k in C_KEYS}
            for s in starts:
                with torch.no_grad():
                    while pos < s:
                        cache = m(torch.tensor([[ref[pos]]], device="cuda"), past_key_values=cache, use_cache=True).past_key_values
                        pos += 1
                # cache has consumed ids + ref[:s]. Like self-speculation after an accepted token ref[s]:
                # the draft feeds ref[s] and proposes ref[s+1..s+k]; compare with the 8-bit trajectory.
                for b in bits:
                    c2 = copy.deepcopy(cache); nest.set(b)
                    drafted, nxt = [], torch.tensor([[ref[s]]], device="cuda")
                    with torch.no_grad():
                        for _ in range(a.kmax):
                            o = m(nxt, past_key_values=c2, use_cache=True); c2 = o.past_key_values
                            nxt = o.logits[:, -1].argmax(-1, keepdim=True); drafted.append(nxt.item())
                    r = 0
                    while r < a.kmax and drafted[r] == ref[s + 1 + r]: r += 1
                    Bacc[b].append(r)
                    nest.set(8)
                # C: snapshot the recurrent state at reduced precision, then TEACHER-FORCE the next 32 reference
                # tokens at 8-bit weights; compare logits with the unmodified state (top-1 agreement, mean KL).
                rt = recurrent_tensors(cache)
                feed = torch.tensor([ref[s:s + 32]], device="cuda")
                with torch.no_grad():
                    lref = m(feed, past_key_values=copy.deepcopy(cache), use_cache=True).logits[0].float()
                for key in C_KEYS:
                    fmt, sb = key.split("-"); sb = int(sb)
                    c3 = copy.deepcopy(cache)
                    for (li, k) in rt:
                        c3.layers[li].recurrent_states[k] = nested_state(c3.layers[li].recurrent_states[k], sb, fmt, a.state_group)
                    with torch.no_grad():
                        lq = m(feed, past_key_values=c3, use_cache=True).logits[0].float()
                    top1 = (lq.argmax(-1) == lref.argmax(-1)).float().mean().item()
                    kl = torch.nn.functional.kl_div(lq.log_softmax(-1), lref.log_softmax(-1), log_target=True, reduction="batchmean").item()
                    Cagree[key].append((top1, kl))
            for b in bits:
                v = Bacc[b]
                rec["B"][b] = {k: statistics.mean(min(x, k) for x in v) for k in (2, 4, 8)}
            rec["C"] = {k: dict(top1=statistics.mean(x[0] for x in v), kl=statistics.mean(x[1] for x in v)) for k, v in Cagree.items()}
            rec["n_recurrent_tensors"] = len(rt)
            res["prompts"].append(rec)
            print(f"[{time.time()-t0:5.0f}s] {kind:4} think={int(think)} P={P:4} "
                  + " ".join(f"b{b}: pA={rec['A'][b]['p_match']:.2f} B@4={rec['B'][b][4]:.2f}" for b in bits)
                  + "  C top1/KL: " + " ".join(f"{k}={rec['C'][k]['top1']:.2f}/{rec['C'][k]['kl']:.3f}" for k in C_KEYS), flush=True)
    # state sizes
    tc = m.config.get_text_config() if hasattr(m.config, "get_text_config") else m.config
    nlin = sum(t == "linear_attention" for t in tc.layer_types); nfull = len(tc.layer_types) - nlin
    st_elems = nlin * tc.linear_num_value_heads * tc.linear_key_head_dim * tc.linear_value_head_dim
    conv_elems = nlin * (2 * tc.linear_num_key_heads * tc.linear_key_head_dim + tc.linear_num_value_heads * tc.linear_value_head_dim) * (tc.linear_conv_kernel_dim - 1)
    kv_per_tok = nfull * 2 * tc.num_key_value_heads * tc.head_dim
    st0 = cache.layers[rt[0][0]].recurrent_states[rt[0][1]]
    res["state_tensor"] = dict(shape=list(st0.shape), dtype=str(st0.dtype))
    res["sizes"] = dict(linear_layers=nlin, full_layers=nfull, state_elems=st_elems, conv_elems=conv_elems, kv_elems_per_token=kv_per_tok)
    json.dump(res, open(a.out, "w"), indent=1)
    print("# wrote", a.out)

if __name__ == "__main__":
    main()
