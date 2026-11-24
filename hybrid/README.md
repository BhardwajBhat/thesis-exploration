# Direction #1, experiment 1: nested self-speculation and state snapshots on Qwen3.5-0.8B (GPU emulation)

Run: `.venv/bin/python hybrid/exp1_nested_spec.py --wfmt bp|rtn [--state-group 32]`, then `python3 hybrid/analyze_exp1.py hybrid/results/exp1_*.json`.
Nested weights are emulated round-to-nearest 8-bit codes (group 128), read at fewer bit-planes; not the user's calibrated KV260 codes.

Findings (17 prompt runs x 160 greedy tokens; tool, code, chat, math, with/without thinking):
1. Acceptance is high: a 4-bit nested draft starting from the 8-bit state accepts 3.0/4 (k=4) and 5.0/8 (k=8); a 6-bit draft accepts 3.7/4.
2. Projected KV260 speedup is at most 1.13x (4-bit draft, k=2), because the measured 4-bit step (38.6 ms) is 67% of the 8-bit step (58 ms).
   A 1.5x speedup at k=4 needs a draft step at or below ~22 ms (~0.39 of target).
3. Snapshot writes are cheap at batch 1: the DeltaNet state is 4.7 MB at 8 bit, so a 4-bit snapshot takes ~0.2 ms on the KV260 vs 58 ms per token.
4. Low-bit state snapshots are NOT free (teacher-forced 32 tokens after restoring, top-1 agreement vs unmodified state):
   group 128: 8-bit 0.97, 6-bit 0.90, 4-bit 0.76; group 32: 8-bit 0.98, 6-bit 0.95, 4-bit 0.90 (worst prompt 0.77).
5. Nested format matters: two's-complement bit-plane truncation of the state collapses at 4 bit (top-1 0.20 / 0.56 with group 32),
   sign-magnitude bit-planes with a zero bucket nearly match round-to-nearest (0.73 / 0.87). For weights the opposite holds:
   sign-magnitude collapses at 3 bit, two's-complement truncation works.
