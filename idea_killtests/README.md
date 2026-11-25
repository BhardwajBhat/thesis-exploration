# Software kill tests for the 2026-11 deep-search shortlist

## Results

| Idea | Test | Result | Verdict |
|---|---|---|---|
| Grammar-aware sparse output head | Exact XGrammar legal masks over 15 BFCL/tau2-shaped schemas, 500 token boundaries, KV260 timing projection | 45.45% mean rows at block 64; optimistic 1.135x; task speedup 1.063x at 50% structured tokens | **Kill** |
| Fused FPGA grammar filter + argmax | Charge measured XGrammar mask generation against 58 ms KV260 token | removing all host grammar work gives 1.00021x mean, 1.00103x worst | **Kill as performance thesis** |
| Spark unified-memory governor | Portable PSI/process/cgroup/NVML logger implemented and smoke-tested | Requires the DGX Spark and a real vLLM + agent workload; no relevant machine is attached to this session | **Pending hardware run; not honestly killable locally** |
| Visual-delta front end | Pixel/patch delta on 176 transitions from 20 real GUI-CC trajectories | 64.9% mean and 84.4% median patches change exactly; p90 is 100%; ideal encoder-only gain 1.54x, only 1.36x with refresh every 4 frames | **Kill**: weak/irregular headroom, no accuracy proof, and GUI-KV already covers reuse |
| Static-NPU re-arrival support | Inspected and partially ran Escapement's public CA25 NPU artifact | Existing work reports 5.7-17.8% bucket penalties and 9.72% tuning gain; local descriptor/reuse-cliff checks passed | **Kill for novelty** |
| Hardware action-safety monitor | 10,000 mutated robot actions: JSON Schema/ranges versus context-aware policy | schema catches 50.98% of unsafe actions; misses human proximity, fragile-object, closed-door, and emergency-stop context | **Kill as grammar-only safety thesis** |

## Commands

```bash
.venv/bin/python structured_head/kill_test.py
.venv/bin/python idea_killtests/fused_filter.py
.venv/bin/python idea_killtests/visual_delta.py --episodes 20
.venv/bin/python idea_killtests/safety_monitor.py
.venv/bin/python idea_killtests/uma_probe.py --seconds 600 --out uma_trace.jsonl
```

Dependencies added to the local venv: `xgrammar==0.1.29`, `requests`, `pillow`, and `jsonschema`. They were installed without changing the pinned CUDA PyTorch.

## Interpretation

Four newly proposed hardware directions are dead or substantially weakened. The static-NPU direction is directly preempted by stronger real-hardware work. The only item not resolved by software on this host is the Spark unified-memory governor because its premise is GB10 unified-memory accounting and whole-host failure behavior; simulating that on a discrete 4090 machine would not answer the question.
