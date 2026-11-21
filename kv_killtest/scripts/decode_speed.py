"""Batch-1 decode throughput vs. KV-cache dtype at long context.

Usage: decode_speed.py <hf_model> <kv_dtype> <out_csv> [ctx,ctx,...]
One engine per KV dtype. For each context length: one random-token prompt of that
length, then generate N tokens with ignore_eos. Decode tok/s = N_extra / (t(1+N) - t(1)),
which subtracts prefill + scheduling overhead. Prefix caching is off so every
repetition re-prefills. Qwen3 native context is 32k (40960 positions); YaRN x4 is
enabled for every context so all points use one engine config (it does not change
per-token decode cost).
"""
import csv, os, sys, time, statistics
import numpy as np
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

model, kv, out = sys.argv[1], sys.argv[2], sys.argv[3]
ctxs = [int(c) for c in (sys.argv[4] if len(sys.argv) > 4 else "8192,32768,130000").split(",")]
N, REPS = 256, 3
llm = LLM(model=model, dtype="bfloat16", kv_cache_dtype=kv, seed=0,
          max_model_len=min(max(ctxs) + N + 64, 131072), enable_prefix_caching=False,
          gpu_memory_utilization=0.6, max_num_seqs=1,
          hf_overrides={"rope_parameters": {"rope_type": "yarn", "factor": 4.0, "rope_theta": 1000000,
                                            "original_max_position_embeddings": 32768},
                        "max_position_embeddings": 131072})
kv_tokens = llm.llm_engine.vllm_config.cache_config.num_gpu_blocks * \
            llm.llm_engine.vllm_config.cache_config.block_size
rng = np.random.default_rng(0)

def run(ids, n):
    sp = SamplingParams(temperature=0, max_tokens=n, ignore_eos=True, min_tokens=n)
    t = time.perf_counter(); llm.generate([TokensPrompt(prompt_token_ids=ids)], sp, use_tqdm=False)
    return time.perf_counter() - t

rows = []
for ctx in ctxs:
    ids = rng.integers(1000, 100000, size=ctx).tolist()
    run(ids, 2)  # warmup at this length
    t1 = [run(ids, 1) for _ in range(REPS)]
    tN = [run(ids, N + 1) for _ in range(REPS)]
    dec = N / (statistics.median(tN) - statistics.median(t1))
    rows.append(dict(model=os.path.basename(model.rstrip("/")), kv_dtype=kv, ctx=ctx,
                     decode_tok_s=round(dec, 2), prefill_s=round(statistics.median(t1), 3),
                     kv_pool_tokens=kv_tokens))
    print(rows[-1], flush=True)

new = not os.path.exists(out)
with open(out, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); new and w.writeheader(); w.writerows(rows)
