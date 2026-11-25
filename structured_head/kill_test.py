#!/usr/bin/env python3
"""Trace grammar-constrained legal-token sets and project a sparse KV260 output head.

This is a software-only kill test. It replays valid structured outputs through XGrammar,
records the legal vocabulary at every token boundary, rounds legal rows to implementable
vocabulary blocks, and applies the measured Qwen3.5-0.8B KV260 component timings.
"""
import argparse, json, math, os, statistics, time
from collections import Counter, defaultdict

import numpy as np
import torch
import xgrammar as xgr
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen3.5-0.8B"
VOCAB = 248_320
# User's measured 8/6 KV260 decomposition (ms/token).
BODY_MS = 33.1
HEAD_MS = 12.7
FIXED_MS = 58.0 - BODY_MS - HEAD_MS
BLOCKS = [1, 8, 16, 32, 64, 128, 256]


def obj_schema(name, args):
    return {
        "type": "object",
        "properties": {
            "name": {"const": name},
            "arguments": {
                "type": "object",
                "properties": args,
                "required": list(args),
                "additionalProperties": False,
            },
        },
        "required": ["name", "arguments"],
        "additionalProperties": False,
    }


def cases():
    """BFCL/tau2-shaped schemas with valid compact outputs; no benchmark dependency."""
    s = lambda: {"type": "string"}
    enum = lambda xs: {"type": "string", "enum": xs}
    integer = lambda lo=0, hi=10**9: {"type": "integer", "minimum": lo, "maximum": hi}
    number = lambda: {"type": "number"}
    arrs = lambda: {"type": "array", "items": {"type": "string"}}
    rows = [
        ("flight", obj_schema("search_flights", {"origin": s(), "destination": s(), "date": s()}),
         {"name":"search_flights","arguments":{"origin":"Los Angeles","destination":"Tokyo","date":"2026-12-20"}}),
        ("weather", obj_schema("get_weather", {"city": s(), "unit": enum(["celsius", "fahrenheit"])}),
         {"name":"get_weather","arguments":{"city":"Tokyo","unit":"celsius"}}),
        ("sql", obj_schema("run_sql", {"query": s()}),
         {"name":"run_sql","arguments":{"query":"SELECT COUNT(*) FROM orders WHERE created_at >= '2026-11-01' AND created_at < '2026-12-01'"}}),
        ("email", obj_schema("send_email", {"to": s(), "subject": s(), "body": s()}),
         {"name":"send_email","arguments":{"to":"ops@example.com","subject":"November order count","body":"The count is 412."}}),
        ("retail_lookup", obj_schema("lookup_order", {"order_id": s()}),
         {"name":"lookup_order","arguments":{"order_id":"ORD-928104"}}),
        ("retail_refund", obj_schema("issue_refund", {"order_id": s(), "amount": number(), "reason": enum(["damaged","missing","late","other"])}),
         {"name":"issue_refund","arguments":{"order_id":"ORD-928104","amount":39.95,"reason":"damaged"}}),
        ("airline_change", obj_schema("change_flight", {"ticket_id": s(), "new_flight": s(), "cabin": enum(["economy","premium_economy","business","first"])}),
         {"name":"change_flight","arguments":{"ticket_id":"TKT-77192","new_flight":"UA837","cabin":"economy"}}),
        ("calendar", obj_schema("create_event", {"title": s(), "start": s(), "end": s(), "attendees": arrs()}),
         {"name":"create_event","arguments":{"title":"Thesis meeting","start":"2026-12-03T14:00:00-08:00","end":"2026-12-03T14:30:00-08:00","attendees":["advisor@uci.edu"]}}),
        ("file_search", obj_schema("find_files", {"root": s(), "pattern": s(), "max_results": integer(1,100)}),
         {"name":"find_files","arguments":{"root":"/home/user/project","pattern":"*.py","max_results":10}}),
        ("shell", obj_schema("run_command", {"command": s(), "timeout_seconds": integer(1,3600)}),
         {"name":"run_command","arguments":{"command":"find . -type f -printf '%s %p\\n' | sort -nr | head -10","timeout_seconds":30}}),
        ("coordinates", obj_schema("set_destination", {"latitude": number(), "longitude": number(), "mode": enum(["walk","drive","transit"])}),
         {"name":"set_destination","arguments":{"latitude":33.6405,"longitude":-117.8443,"mode":"walk"}}),
        ("lights", obj_schema("set_lights", {"room": enum(["kitchen","bedroom","office","living_room"]), "brightness": integer(0,100), "color": enum(["warm","neutral","cool"])}),
         {"name":"set_lights","arguments":{"room":"office","brightness":35,"color":"warm"}}),
        ("database", obj_schema("insert_record", {"table": enum(["users","orders","events"]), "values": {"type":"object","additionalProperties":{"type":["string","number","boolean","null"]}}}),
         {"name":"insert_record","arguments":{"table":"events","values":{"kind":"login","user_id":381,"success":True}}}),
        ("search", obj_schema("web_search", {"query": s(), "domains": arrs(), "limit": integer(1,20)}),
         {"name":"web_search","arguments":{"query":"FPGA grammar constrained decoding","domains":["arxiv.org","github.com"],"limit":10}}),
        ("message", obj_schema("send_message", {"channel": enum(["email","sms","slack"]), "recipient": s(), "message": s()}),
         {"name":"send_message","arguments":{"channel":"slack","recipient":"#architecture","message":"The experiment completed successfully."}}),
    ]
    return [(n, sc, json.dumps(out, separators=(",", ":"), ensure_ascii=False)) for n, sc, out in rows]


def unpack_mask(bitmask, vocab):
    # XGrammar stores token i in bit i%32 of int32 word i//32.
    u = bitmask.detach().cpu().numpy().view(np.uint32).reshape(-1)
    bits = np.unpackbits(u.view(np.uint8), bitorder="little")[:vocab]
    return np.flatnonzero(bits).astype(np.int32)


def classify(prefix):
    # Coarse lexical phase, sufficient to expose free-string states.
    in_string = False; escaped = False
    for c in prefix:
        if escaped: escaped = False; continue
        if c == "\\" and in_string: escaped = True; continue
        if c == '"': in_string = not in_string
    if in_string: return "inside_string"
    if not prefix: return "start"
    if prefix[-1] in "{[,:": return "structure"
    return "outside_string"


def percentile(xs, q):
    return float(np.percentile(np.asarray(xs, dtype=float), q)) if xs else float("nan")


def project(active_fraction, grammar_us=0.0):
    sparse_head = HEAD_MS * active_fraction
    total = BODY_MS + FIXED_MS + sparse_head + grammar_us / 1000.0
    return {"head_ms": sparse_head, "total_ms": total, "speedup": 58.0 / total}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--out", default="structured_head/results/kill_test.json")
    ap.add_argument("--max-cases", type=int, default=0)
    ap.add_argument("--local-files-only", action="store_true", default=True)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=a.local_files_only)
    ti = xgr.TokenizerInfo.from_huggingface(tok, vocab_size=VOCAB)
    compiler = xgr.GrammarCompiler(ti, max_threads=8)
    bitmask = xgr.allocate_token_bitmask(1, VOCAB)
    all_steps, summaries = [], []

    selected = cases()[:a.max_cases or None]
    for ci, (name, schema, text) in enumerate(selected):
        t0 = time.perf_counter()
        compiled = compiler.compile_json_schema(schema, any_whitespace=False, separators=(",", ":"))
        compile_ms = (time.perf_counter() - t0) * 1e3
        matcher = xgr.GrammarMatcher(compiled)
        ids = tok.encode(text, add_special_tokens=False)
        decoded = ""
        case_steps = []
        for pos, tid in enumerate(ids):
            t0 = time.perf_counter_ns()
            need_apply = matcher.fill_next_token_bitmask(bitmask)
            fill_us = (time.perf_counter_ns() - t0) / 1e3
            legal = unpack_mask(bitmask[0], VOCAB) if need_apply else np.arange(VOCAB, dtype=np.int32)
            row = {
                "case": name, "pos": pos, "token_id": tid, "token": tok.decode([tid]),
                "phase": classify(decoded), "legal": int(legal.size), "legal_fraction": float(legal.size / VOCAB),
                "fill_us": fill_us, "need_apply": bool(need_apply), "blocks": {},
            }
            for g in BLOCKS:
                active = np.unique(legal // g).size
                rounded_rows = min(VOCAB, active * g)
                row["blocks"][str(g)] = {"active": int(active), "fraction": float(rounded_rows / VOCAB)}
            if not matcher.accept_token(tid):
                raise RuntimeError(f"{name}: grammar rejected token {pos} id={tid} text={tok.decode([tid])!r}; prefix={decoded!r}")
            decoded += tok.decode([tid])
            case_steps.append(row); all_steps.append(row)
        summary = {
            "name": name, "tokens": len(ids), "compile_ms": compile_ms,
            "fill_us_mean": statistics.mean(x["fill_us"] for x in case_steps),
            "legal_fraction_mean": statistics.mean(x["legal_fraction"] for x in case_steps),
            "blocks": {},
        }
        for g in BLOCKS:
            fs = [x["blocks"][str(g)]["fraction"] for x in case_steps]
            p = project(statistics.mean(fs), summary["fill_us_mean"])
            summary["blocks"][str(g)] = {"fraction_mean": statistics.mean(fs), **p}
        summaries.append(summary)
        print(f"[{ci+1:02}/{len(selected)}] {name:16s} tok={len(ids):3d} legal={summary['legal_fraction_mean']*100:6.2f}% "
              f"B64={summary['blocks']['64']['fraction_mean']*100:6.2f}% speedup={summary['blocks']['64']['speedup']:.3f}x fill={summary['fill_us_mean']:.1f}us", flush=True)

    aggregate = {"steps": len(all_steps), "cases": len(summaries), "blocks": {}, "phases": {}}
    for g in BLOCKS:
        fs = [x["blocks"][str(g)]["fraction"] for x in all_steps]
        p = project(statistics.mean(fs), statistics.mean(x["fill_us"] for x in all_steps))
        aggregate["blocks"][str(g)] = {
            "fraction_mean": statistics.mean(fs), "fraction_median": statistics.median(fs),
            "fraction_p90": percentile(fs, 90), **p,
        }
    for phase in sorted({x["phase"] for x in all_steps}):
        ss = [x for x in all_steps if x["phase"] == phase]
        aggregate["phases"][phase] = {
            "steps": len(ss), "legal_fraction_mean": statistics.mean(x["legal_fraction"] for x in ss),
            "block64_fraction_mean": statistics.mean(x["blocks"]["64"]["fraction"] for x in ss),
        }
    fill = [x["fill_us"] for x in all_steps]
    aggregate["fill_us"] = {"mean": statistics.mean(fill), "median": statistics.median(fill), "p90": percentile(fill,90), "max": max(fill)}
    # Fraction of an entire generation that must be structured to reach task-level thresholds.
    structured_speed = aggregate["blocks"]["64"]["speedup"]
    aggregate["task_speedup"] = {}
    for share in [0.1,0.25,0.5,0.75,1.0]:
        new_avg = (1-share)*58.0 + share*(58.0/structured_speed)
        aggregate["task_speedup"][str(share)] = 58.0/new_avg

    verdict = "continue" if aggregate["blocks"]["64"]["speedup"] >= 1.15 else "kill"
    result = {
        "model": a.model, "vocab": VOCAB, "timing_ms": {"body":BODY_MS,"head":HEAD_MS,"fixed":FIXED_MS,"baseline":58.0},
        "block_sizes": BLOCKS, "aggregate": aggregate, "cases": summaries, "steps": all_steps, "verdict_at_block64": verdict,
        "limitations": [
            "Synthetic BFCL/tau2-shaped schemas and valid outputs; replace/extend with benchmark traces.",
            "Projection scales head time with rounded active rows and does not yet charge fragmented DDR burst inefficiency.",
            "XGrammar host fill time is charged, but CPU-to-FPGA index transfer and FPGA scheduler overhead are not.",
        ],
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out,"w") as f: json.dump(result,f,indent=2)
    print("\nAggregate")
    for g in BLOCKS:
        z=aggregate["blocks"][str(g)]
        print(f"  block {g:3}: rows={z['fraction_mean']*100:6.2f}% p90={z['fraction_p90']*100:6.2f}% head={z['head_ms']:5.2f}ms speedup={z['speedup']:.3f}x")
    print("  structured-share task speedup (block 64):", aggregate["task_speedup"])
    print("  VERDICT:", verdict.upper(), "(software lower-bound model)")
    print("# wrote", a.out)

if __name__ == "__main__": main()
