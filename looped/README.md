# Looped-language-model kill test

Tests whether adaptive recurrent depth has enough headroom to justify a hardware thesis.
The model is `ByteDance/Ouro-1.4B` (four recurrent passes); results are from GPU 0,
an RTX 4090.

## Run

The released Ouro remote code works with Transformers 4.55 but not the repository's
5.20 installation. A temporary compatible environment was used:

```bash
uv venv --python 3.10 /tmp/venv-ouro
uv pip install --python /tmp/venv-ouro/bin/python transformers==4.55.0
uv pip install --python /tmp/venv-ouro/bin/python torch==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
CUDA_VISIBLE_DEVICES=0 /tmp/venv-ouro/bin/python looped/kill_test.py
```

The script evaluates 3,072 WikiText-2 next-token positions. It obtains all four
intermediate recurrent outputs, evaluates the released exit gate, calibrates a
simple logit-margin exit policy on half the tokens, and tests it on the other half.
It also measures fixed-depth forward latency after warmup.

## Result

See `results/ouro_kill.json`.

| Depth | NLL | Top-1 agreement with depth 4 | Relative latency |
|---:|---:|---:|---:|
| 1 | 6.036 | 49.8% | 1.00x |
| 2 | 4.452 | 68.8% | 1.99x |
| 3 | 3.736 | 86.1% | 2.97x |
| 4 | 3.591 | 100% | 3.96x |

There is real oracle headroom: the first depth agreeing with depth 4 is depth
1/2/3/4 for 49.8/22.1/16.8/11.3% of tokens. An oracle therefore uses about
1.91 steps on average instead of four.

The available gates do not realize this headroom:

- The released learned gate's most likely depth is 4 for 90.4% of tokens.
- Its exact match with the agreement oracle is only 12.2%.
- A calibrated logit-margin policy uses 3.236 steps on held-out tokens (80.9% of
  full recurrent compute) while retaining 99.0% top-1 agreement with depth 4.
- This is only a ~1.24x ideal compute reduction, before controller, compaction,
  and intermediate LM-head costs. Computing a full-vocabulary margin at every
  depth is itself unsuitable for edge hardware.

## Verdict

**Do not pursue generic adaptive-depth serving.** CDB already covers GPU
scheduling, while the cheap implementable policy leaves modest headroom.

A narrower hardware thesis survives only conditionally: train a tiny hidden-state
exit gate that approaches the ~1.91-step oracle without intermediate vocabulary
projections, then combine it with on-chip recurrent weights or hardware slack.
Kill that direction unless the cheap gate reaches <=2.6 mean steps at >=99%
depth-4 agreement (at least ~1.5x ideal recurrent-compute reduction) on reasoning
benchmarks, not just WikiText.
