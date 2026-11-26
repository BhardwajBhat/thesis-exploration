# Context (read fully)

You are helping a master's student (UC Irvine, advisor Sitao Huang: FPGA / heterogeneous computing) find a master's
thesis topic at the intersection of (1) computer architecture / FPGA / heterogeneous hardware and (2) AI systems
(LLM serving, agents, edge inference). Today is 2026-11-26. The student wants STRONG REAL DEMAND (users, engineers or
companies hitting the problem now: GitHub issues, forum threads, vendor docs, production reports), not a solution
looking for a problem. arXiv volume alone is not demand. Preprints from well-known groups are fine as prior work.
The student prefers blunt, short answers. Many ideas were already killed; do not re-propose them (list below).

Hardware the student can use for experiments:
- 4x A100 80GB (vLLM 0.19.1, driver 535 / CUDA 12.x only, no CUDA 13), plus 2x RTX 4090 on another box.
- KV260 FPGA with the student's own Qwen3.5-0.8B (Gated DeltaNet hybrid) decoder: nested-precision weights (one 8-bit
  code read at 8/6/4/3 bits per token), 58 ms/token at 8-bit, 0.388 J/token, idle 4.0 W (static power is 70-81% of energy).
- Possibly a VCK5000 (Versal VC1902, AIE + PL; Vitis 2022.x only); the lab's HeteroVLM work runs VLMs on it.
- Lab group papers (overlap risk): RaReCache (cross-model KV reuse), HeteroVLM (VLM on Versal), Chimera (HBM multi-die
  FPGA decode), TeLLMe v1/v2 (ternary LLM on KV260), Hummingbird.
- The student's own prior work: STRIDE (lossless look-ahead agent turn execution on A100s, hidden-state gates); SPORK replication.
- No DGX Spark access any more.

Rules: research only. Do NOT edit, create or delete any files outside /tmp. Do not run GPU jobs. Open every URL you cite
and only cite what you opened (an earlier reviewer fabricated a citation). Give arXiv IDs or URLs.

# Already dead or taken (with reasons) and still-alive ideas

{{DEAD}}

Also killed today: Inkling 2 Mini looped-MoE expert reuse across passes (14.1 unique experts over 3 passes, and already
published by noamshh.github.io/blog/inkling-loop-routing); edge STRIDE (lossless agent forks on KV260: ideal 1.10x
on tau2, 1.27x on BFCL; the algorithm is claimed by Speculative Actions 2510.04371).

# Your output (strict)

Return at most 3 candidate ideas from YOUR AREA (below), ranked best first, and nothing else. For each, at most 10 lines:
1. Name + one-sentence idea (hardware angle explicit).
2. Demand: 1-3 concrete links (issues/forums/vendor docs/production reports) with one line each.
3. Closest prior work: 2-4 opened links, with what they leave open.
4. Why it fits this student (hardware above, assets).
5. One-day kill test runnable on the hardware above, with a numeric kill criterion.
6. Verdict: strong / medium / weak, and the biggest risk.
If nothing in your area survives, say so in 3 lines and name the closest misses. Be blunt.

# YOUR AREA
