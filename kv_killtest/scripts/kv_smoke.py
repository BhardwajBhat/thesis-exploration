"""Smoke test: does vLLM run Qwen3 with a given KV-cache dtype on this GPU, and do outputs stay sane?"""
import sys, time
from vllm import LLM, SamplingParams
model, kv = sys.argv[1], sys.argv[2]
llm = LLM(model=model, dtype="bfloat16", kv_cache_dtype=kv, max_model_len=8192,
          gpu_memory_utilization=0.5, seed=0)
print("ATTN_BACKEND", llm.llm_engine.vllm_config.attention_config if hasattr(llm.llm_engine, "vllm_config") else "?")
msgs = [[{"role": "user", "content": "List the first 10 prime numbers, comma-separated, nothing else."}],
        [{"role": "user", "content": "Return JSON {\"name\": ..., \"args\": {...}} calling get_weather for Paris in celsius. Only JSON."}]]
sp = SamplingParams(temperature=0, max_tokens=64)
t = time.time()
outs = llm.chat(msgs, sp, chat_template_kwargs={"enable_thinking": False})
for o in outs: print(f"[{kv}] {o.outputs[0].text.strip()!r}")
print(f"[{kv}] ok {time.time()-t:.1f}s")
