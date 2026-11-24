"""
Experiment 1b: does a draft with a TRIMMED vocabulary head keep its acceptance?
Draft = nested b-bit body + lm_head restricted to the N most frequent tokens (frequency counted on corpora
unrelated to the test prompts: wikitext-103 + gsm8k + SWE-bench problem statements). The target is the full
8-bit model. Same protocol as exp1 part B (draft starts from the 8-bit state, k tokens, count matches).
"""
import argparse, collections, copy, json, os, statistics, time
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from exp1_nested_spec import PROMPTS, Nested, enc, greedy

def token_ranks(tok, n_docs=4000):
    cnt = collections.Counter()
    srcs = [("Salesforce/wikitext", "wikitext-103-raw-v1", "train", "text"),
            ("openai/gsm8k", "main", "train", "question"), ("openai/gsm8k", "main", "train", "answer"),
            ("SWE-bench/SWE-bench_Verified", None, "test", "problem_statement")]
    for name, cfg, split, col in srcs:
        ds = load_dataset(name, cfg, split=split)
        for row in ds.select(range(min(n_docs, len(ds)))):
            if row[col]: cnt.update(tok(row[col])["input_ids"])
    # chat-template special tokens must stay in the draft vocabulary
    for t in tok.all_special_ids: cnt[t] += 10**9
    for s in ("<|im_start|>", "<|im_end|>", "<think>", "</think>", "<tool_call>", "</tool_call>"):
        i = tok.convert_tokens_to_ids(s)
        if isinstance(i, int) and i >= 0: cnt[i] += 10**9
    return [t for t, _ in cnt.most_common()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", type=int, default=160); ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--k", type=int, default=8); ap.add_argument("--bits", default="4,3")
    ap.add_argument("--vocab", default="8192,16384,32768,65536,248320")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results", "exp1b.json"))
    a = ap.parse_args()
    bits = [int(x) for x in a.bits.split(",")]; vocabs = [int(x) for x in a.vocab.split(",")]
    mid = "Qwen/Qwen3.5-0.8B"
    tok = AutoTokenizer.from_pretrained(mid)
    m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16).cuda().eval()
    nest = Nested(m, [8] + bits, fmt="bp")
    ranks = token_ranks(tok)
    V = m.config.get_text_config().vocab_size
    masks = {}
    for n in vocabs:
        keep = torch.zeros(V, dtype=torch.bool, device="cuda")
        keep[torch.tensor(ranks[:n], device="cuda")] = True
        if n >= V: keep[:] = True
        masks[n] = keep
    res = dict(vocabs=vocabs, bits=bits, k=a.k, n_ranked=len(ranks), runs=[])
    cover = collections.defaultdict(list); acc = collections.defaultdict(list)
    t0 = time.time()
    for kind, text in PROMPTS:
        for think in (False, True) if kind in ("math", "code") else (False,):
            ids = enc(tok, text, think); nest.set(8)
            ref, _ = greedy(m, ids, a.gen)
            for n in vocabs: cover[n].append(statistics.mean(masks[n][t].item() for t in ref))
            with torch.no_grad(): cache = m(ids, use_cache=True).past_key_values
            pos = 0
            for s in range(0, a.gen - a.k - 1, a.stride):
                with torch.no_grad():
                    while pos < s:
                        cache = m(torch.tensor([[ref[pos]]], device="cuda"), past_key_values=cache, use_cache=True).past_key_values
                        pos += 1
                for b in bits:
                    for n in vocabs:
                        c2 = copy.deepcopy(cache); nest.set(b); nxt = torch.tensor([[ref[s]]], device="cuda"); dr = []
                        with torch.no_grad():
                            for _ in range(a.k):
                                o = m(nxt, past_key_values=c2, use_cache=True); c2 = o.past_key_values
                                lg = o.logits[:, -1].masked_fill(~masks[n], float("-inf"))
                                nxt = lg.argmax(-1, keepdim=True); dr.append(nxt.item())
                        r = 0
                        while r < a.k and dr[r] == ref[s + 1 + r]: r += 1
                        acc[(b, n)].append(r)
                        nest.set(8)
            print(f"[{time.time()-t0:4.0f}s] {kind} think={int(think)} " + " ".join(
                f"b{b}/V{n//1024}k={statistics.mean(acc[(b, n)]):.2f}" for b in bits for n in vocabs), flush=True)
    res["coverage"] = {n: statistics.mean(v) for n, v in cover.items()}
    res["accepted"] = {f"{b}/{n}": {kk: statistics.mean(min(x, kk) for x in acc[(b, n)]) for kk in (2, 4, 8)} for b in bits for n in vocabs}
    json.dump(res, open(a.out, "w"), indent=1); print("# wrote", a.out)

if __name__ == "__main__":
    main()
