"""Inkling 2 Mini: expert overlap across the 3 loop passes (looped/INKLING.md question 1).

For each prompt: greedy-generate a response, then run ONE teacher-forced forward over prompt+response with hooks on
the MoE routers of the looped layers (6-17). Each looped router is called 3 times per forward (pass 1, 2, 3), so the
call order gives the pass index. Per token and layer we record the selected top-8 and the pass-1 score ranking.

Usage: CUDA_VISIBLE_DEVICES=2 python -I expert_overlap.py --out raw_routing.pt [--max-new 384]
"""
import argparse, json, time
import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

MID = "thinkingmachines/Inkling-2-Mini"

TOOLS = [
    {"type": "function", "function": {"name": "get_flight_status", "description": "Get status of a flight",
     "parameters": {"type": "object", "properties": {"flight_number": {"type": "string"}, "date": {"type": "string"}},
                    "required": ["flight_number", "date"]}}},
    {"type": "function", "function": {"name": "search_orders", "description": "Search a user's orders",
     "parameters": {"type": "object", "properties": {"user_id": {"type": "string"}, "status": {"type": "string",
                    "enum": ["pending", "delivered", "cancelled"]}}, "required": ["user_id"]}}},
    {"type": "function", "function": {"name": "convert_currency", "description": "Convert an amount between currencies",
     "parameters": {"type": "object", "properties": {"amount": {"type": "number"}, "from": {"type": "string"},
                    "to": {"type": "string"}}, "required": ["amount", "from", "to"]}}},
    {"type": "function", "function": {"name": "read_file", "description": "Read a file from the workspace",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
]

PROMPTS = [
    ("chat", "What are some good ways to keep houseplants alive in a dry apartment?", False),
    ("chat", "Explain to a 10-year-old why the moon has phases.", False),
    ("chat", "Write a short, friendly email declining a meeting invitation next Tuesday.", False),
    ("chat", "Compare living in a big city versus a small town in a few sentences.", False),
    ("code", "Write a Python function that returns the longest palindromic substring of a string.", False),
    ("code", "In C, implement a ring buffer with push and pop for ints, fixed capacity 64.", False),
    ("code", "Why does this JavaScript print 3 three times? for (var i=0;i<3;i++) setTimeout(()=>console.log(i));", False),
    ("code", "Write a SQL query that returns the top 3 customers by total order amount from orders(customer_id, amount).", False),
    ("math", "A train travels 180 km in 2.5 hours. What is its average speed in m/s?", False),
    ("math", "Solve for x: 3x^2 - 12x + 9 = 0.", False),
    ("math", "How many ways can 5 people sit around a round table?", False),
    ("math", "What is the sum of the first 50 odd numbers?", False),
    ("tool", "Is flight UA 288 on time today (2026-11-26)?", True),
    ("tool", "Show me the pending orders for user u_4821.", True),
    ("tool", "How much is 250 US dollars in euros?", True),
    ("tool", "Open the file src/main.py and tell me what it does.", True),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new", type=int, default=384)
    args = ap.parse_args()

    proc = AutoProcessor.from_pretrained(MID)
    model = AutoModelForMultimodalLM.from_pretrained(MID, dtype=torch.bfloat16, device_map="cuda:0").eval()
    lm = model.model.language_model
    cfg = lm.config
    ls, le, npass = cfg.loop_layer_index_start, cfg.loop_layer_index_end, cfg.loop_num_passes
    print("loop layers", ls, le, "passes", npass, flush=True)

    rec = {"on": False, "calls": {}, "data": {}}

    def make_hook(li):
        def hook(mod, inp, out):
            if not rec["on"]:
                return
            logits, weights, idx = out
            p = rec["calls"].get(li, 0)
            rec["calls"][li] = p + 1
            scores = mod.score_fn(logits.float()) + mod.e_score_correction_bias
            rank = torch.argsort(scores, dim=-1, descending=True)[:, :32]  # top-32 ranking (for prefetch recall)
            rec["data"][(li, p)] = (idx.cpu(), weights.float().cpu(), rank.cpu())
        return hook

    for li, layer in enumerate(lm.layers):
        layer.mlp.gate.register_forward_hook(make_hook(li))

    results = []
    for k, (cat, text, use_tools) in enumerate(PROMPTS):
        msgs = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        kw = {"tools": TOOLS} if use_tools else {}
        inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                       return_tensors="pt", **kw).to("cuda:0")
        n_prompt = inp["input_ids"].shape[-1]
        t = time.time()
        with torch.no_grad():
            gen = model.generate(**inp, max_new_tokens=args.max_new, do_sample=False)
        tg = time.time() - t
        full = gen[:, :]
        rec["on"], rec["calls"], rec["data"] = True, {}, {}
        with torch.no_grad():
            model(input_ids=full, attention_mask=torch.ones_like(full), use_cache=False)
        rec["on"] = False
        assert all(rec["calls"][li] == (npass if ls <= li < le else 1) for li in range(cfg.num_hidden_layers)), rec["calls"]
        item = {"cat": cat, "prompt": text, "n_prompt": n_prompt, "n_total": int(full.shape[-1]),
                "gen_s": tg, "response": proc.decode(gen[0][n_prompt:]),
                "layers": {}}
        for li in range(cfg.num_hidden_layers):
            ps = range(npass) if ls <= li < le else range(1)
            item["layers"][li] = {p: {"idx": rec["data"][(li, p)][0].to(torch.uint8),
                                      "w": rec["data"][(li, p)][1].half(),
                                      "rank32": rec["data"][(li, p)][2].to(torch.uint8)} for p in ps}
        results.append(item)
        print(f"[{k+1}/{len(PROMPTS)}] {cat} prompt={n_prompt} total={full.shape[-1]} gen={tg:.0f}s", flush=True)

    meta = {"model": MID, "loop_start": ls, "loop_end": le, "num_passes": npass,
            "top_k": cfg.num_experts_per_tok, "n_experts": cfg.n_routed_experts, "dtype": "bf16 (dequantized on load)",
            "transformers": __import__("transformers").__version__, "max_new": args.max_new}
    torch.save({"meta": meta, "items": results}, args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
