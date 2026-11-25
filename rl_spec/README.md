# Exact speculation oracle screening bound

`oracle_bound.py` uses the public RollPacker KodCode trace (2,001 evaluator executions) to ask how much next-turn inference could ideally overlap code-environment execution.

Tool/evaluator duration:

- median: 30.8 ms
- p90: 78.4 ms
- p95: 155.7 ms
- p99: 9.14 s
- mean: 269 ms, dominated by the heavy tail

With 20 ms fork/verification overhead and perfect speculation:

- 0.5 s next LLM step: 1.051x aggregate speedup
- 1.0 s next LLM step: 1.042x
- 2.0 s next LLM step: 1.031x

At 90% gate accuracy the corresponding gains are 1.043x, 1.036x, and 1.027x. Only about 3-5% of tasks individually exceed 1.15x.

## Verdict

**Kill for this code-RL workload.** Typical evaluator calls are too short to hide meaningful next-turn inference, and realistic fork overhead consumes much of the median opportunity.

This is not a universal kill for multi-turn web/API agents because the public trace contains evaluator duration rather than complete multi-turn timelines. However, combined with prior no-gain math/SPORK results, it removes gated exact speculation from the recommended shortlist unless real target traces show much longer tool gaps on the straggler trajectory.
