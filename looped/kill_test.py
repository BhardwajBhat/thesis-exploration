#!/usr/bin/env python3
"""Cheap Ouro-1.4B recurrent-depth kill test.

Measures whether extra recurrent steps materially improve next-token prediction,
how often shallow steps already agree with depth 4, whether Ouro's learned exit
gate tracks a token-level oracle, and fixed-depth GPU latency.

Requires transformers==4.55.0 (the released Ouro remote code is incompatible
with newer Transformers cache/rope APIs). GPU 0 is selected by the shell.
"""
from __future__ import annotations

import argparse
import json
import math
import time
import urllib.request
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

MODEL = "ByteDance/Ouro-1.4B"
TEXT_URL = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/test.txt"


def load_text() -> str:
    try:
        return urllib.request.urlopen(TEXT_URL, timeout=30).read().decode("utf-8")
    except Exception:
        return ("Recurrent neural networks process a sequence one element at a time. "
                "Looped language models instead reuse a shared block at recurrent depth. ") * 1000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=48)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--output", default="looped/results/ouro_kill.json")
    args = ap.parse_args()

    torch.manual_seed(0)
    config = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    config.pad_token_id = config.eos_token_id
    config.use_cache = False
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, config=config, trust_remote_code=True,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    ids = tokenizer(load_text(), add_special_tokens=False, return_tensors="pt").input_ids[0]
    need = args.windows * (args.seq_len + 1)
    if ids.numel() < need:
        ids = ids.repeat(math.ceil(need / ids.numel()))
    # Spread windows through the corpus rather than using a contiguous header.
    starts = torch.linspace(0, ids.numel() - args.seq_len - 1, args.windows).long()
    windows = torch.stack([ids[s:s + args.seq_len + 1] for s in starts])

    nll_sum = torch.zeros(4, dtype=torch.float64)
    correct_sum = torch.zeros(4, dtype=torch.long)
    agreement = torch.zeros(4, dtype=torch.long)
    oracle_hist = torch.zeros(4, dtype=torch.long)
    gate_hist = torch.zeros(4, dtype=torch.long)
    gate_oracle_match = 0
    gate_abs_error = 0.0
    # Confidence-margin policy data: can an implementable local signal predict
    # that the current step already matches the depth-4 token?
    margin_chunks = []
    match_chunks = []
    token_count = 0

    model.model.total_ut_steps = 4
    with torch.inference_mode():
        for batch in windows.split(args.batch_size):
            x, labels = batch[:, :-1].cuda(), batch[:, 1:].cuda()
            outputs, hidden, gates = model.model(input_ids=x, use_cache=False)
            logits = [model.lm_head(h).float() for h in hidden]
            pred = torch.stack([z.argmax(-1) for z in logits])  # [depth,batch,seq]
            losses = torch.stack([
                F.cross_entropy(z.flatten(0, 1), labels.flatten(), reduction="none").view_as(labels)
                for z in logits
            ])
            for d in range(4):
                nll_sum[d] += losses[d].sum().cpu()
                agreement[d] += (pred[d] == pred[3]).sum().cpu()
            # Dataset labels are only a secondary quality sanity check.
            correct_sum += (pred == labels.unsqueeze(0)).sum(dim=(1, 2)).cpu()

            match = pred == pred[3:4]
            # Top-1 minus top-2 logit is cheap and available at each step.
            margins = torch.stack([z.topk(2, dim=-1).values.diff(dim=-1).neg().squeeze(-1) for z in logits])
            margin_chunks.append(margins[:3].flatten(1).cpu())
            match_chunks.append(match[:3].flatten(1).cpu())
            oracle = match.float().argmax(dim=0)  # first depth agreeing with depth 4
            oracle_hist += torch.bincount(oracle.flatten().cpu(), minlength=4)

            remaining = torch.ones_like(gates[0].squeeze(-1), dtype=torch.float32)
            probs = []
            for d, g in enumerate(gates):
                hazard = torch.sigmoid(g.squeeze(-1).float())
                if d < 3:
                    probs.append(hazard * remaining)
                    remaining = remaining * (1 - hazard)
                else:
                    probs.append(remaining)
            gate_depth = torch.stack(probs, dim=-1).argmax(-1)
            gate_hist += torch.bincount(gate_depth.flatten().cpu(), minlength=4)
            gate_oracle_match += (gate_depth == oracle).sum().item()
            gate_abs_error += (gate_depth.float() - oracle.float()).abs().sum().item()
            token_count += labels.numel()

    # Fixed-depth forward latency, after warmup. This excludes generation/cache bugs
    # in the public release and directly measures the recurrent compute slope.
    bench = windows[:args.batch_size, :-1].cuda()
    latency_ms = {}
    with torch.inference_mode():
        for d in range(1, 5):
            model.model.total_ut_steps = d
            for _ in range(3):
                model.model(input_ids=bench, use_cache=False)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(20):
                model.model(input_ids=bench, use_cache=False)
            torch.cuda.synchronize()
            latency_ms[str(d)] = (time.perf_counter() - t0) * 1000 / 20

    # Calibrate sequential exit thresholds on half the tokens, evaluate on the
    # other half. At each depth choose the lowest margin retaining >=99% train
    # precision relative to depth 4, maximizing safe coverage.
    margins = torch.cat(margin_chunks, dim=1)
    matches = torch.cat(match_chunks, dim=1)
    split = margins.shape[1] // 2
    train_alive = torch.ones(split, dtype=torch.bool)
    eval_alive = torch.ones(margins.shape[1] - split, dtype=torch.bool)
    eval_depth = torch.full((eval_alive.numel(),), 4, dtype=torch.long)
    thresholds = []
    for d in range(3):
        scores = margins[d, :split][train_alive]
        good = matches[d, :split][train_alive]
        order = scores.argsort(descending=True)
        precision = good[order].float().cumsum(0) / torch.arange(1, len(order) + 1)
        valid = torch.where(precision >= 0.99)[0]
        if len(valid):
            k = valid[-1].item()
            threshold = scores[order[k]].item()
        else:
            threshold = float("inf")
        thresholds.append(threshold)
        train_exit = train_alive & (margins[d, :split] >= threshold)
        train_alive &= ~train_exit
        eval_exit = eval_alive & (margins[d, split:] >= threshold)
        eval_depth[eval_exit] = d + 1
        eval_alive &= ~eval_exit
    eval_index = torch.arange(eval_depth.numel())
    policy_match = torch.ones(eval_depth.numel(), dtype=torch.bool)
    shallow = eval_depth < 4
    policy_match[shallow] = matches[eval_depth[shallow] - 1, split + eval_index[shallow]]

    nll = (nll_sum / token_count).tolist()
    result = {
        "model": MODEL,
        "gpu": torch.cuda.get_device_name(0),
        "windows": args.windows,
        "seq_len": args.seq_len,
        "tokens": token_count,
        "depth": {
            str(d + 1): {
                "nll": nll[d],
                "perplexity": math.exp(min(nll[d], 20)),
                "label_top1_accuracy": correct_sum[d].item() / token_count,
                "agreement_with_depth4": agreement[d].item() / token_count,
                "latency_ms_batch": latency_ms[str(d + 1)],
                "latency_relative": latency_ms[str(d + 1)] / latency_ms["1"],
            } for d in range(4)
        },
        "oracle_first_depth_agreeing_with_depth4": {
            str(d + 1): oracle_hist[d].item() / token_count for d in range(4)
        },
        "learned_gate_argmax_depth": {
            str(d + 1): gate_hist[d].item() / token_count for d in range(4)
        },
        "gate_oracle_exact_match": gate_oracle_match / token_count,
        "gate_oracle_mean_abs_depth_error": gate_abs_error / token_count,
        "calibrated_margin_policy": {
            "target_train_precision": 0.99,
            "thresholds_depth1_to_3": thresholds,
            "eval_mean_depth": eval_depth.float().mean().item(),
            "eval_fraction_compute_vs_depth4": eval_depth.float().mean().item() / 4,
            "eval_agreement_with_depth4": policy_match.float().mean().item(),
            "eval_exit_distribution": {
                str(d): (eval_depth == d).float().mean().item() for d in range(1, 5)
            },
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
