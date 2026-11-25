# Grammar-aware sparse output-head kill test

Question: can a JSON/tool-call grammar skip enough rows of Qwen3.5-0.8B's 248,320-row output head to give at least 1.15x end-to-end speedup on the measured KV260 decoder?

## Run

```bash
uv pip install --python .venv/bin/python --no-deps xgrammar==0.1.29
.venv/bin/python structured_head/kill_test.py
```

The script uses 15 BFCL/tau2-shaped schemas and valid compact JSON tool calls. At each output token it asks XGrammar for the exact legal-token bitmask, measures mask-generation time, rounds legal rows to vocabulary blocks, and applies the measured KV260 8/6 decomposition:

- body: 33.1 ms/token
- full output head: 12.7 ms/token
- fixed/non-GEMV: 12.2 ms/token
- total: 58.0 ms/token

Projection is deliberately optimistic: sparse-head time scales linearly with active rows. It does not yet charge random DDR access, DMA command overhead, index transfer, or FPGA scheduling.

## First result

500 structured-output token boundaries:

| Vocabulary block | Mean rows read | Projected speedup |
|---:|---:|---:|
| exact rows | 44.88% | 1.137x |
| 16 rows | 45.37% | 1.136x |
| 64 rows | 45.45% | 1.135x |
| 256 rows | 45.70% | 1.135x |

The 90th percentile token needs essentially the full head (99.92% at block 64). Free-form JSON strings dominate: SQL, shell commands, messages, dates, paths, and user-provided values make much of a byte-level vocabulary legal.

Even if every generated token were inside a constrained tool call, the optimistic block-64 projection is only 1.135x. If structured output is 50% of generated tokens, task-level speedup is 1.063x; at 25%, it is 1.031x.

Some enum/numeric-heavy schemas individually reach 1.18-1.28x, but realistic free-string-heavy schemas are around 1.05-1.14x.

## Initial verdict

**Kill the general-purpose grammar-aware sparse-head thesis direction.** It misses the 1.15x threshold in an optimistic model, before charging hardware inefficiencies.

A much narrower enum/action-language accelerator could survive technically, but it would need a strong real workload where most output positions have closed vocabularies. Generic agent tool calling does not provide that property.

## Outputs

`results/kill_test.json` contains per-case summaries and every token-boundary mask statistic.
