I found one idea worth a day of testing and one weak one. Everything else in edge agents is already taken by work from the last 3 months.

**1. Idle-aware FPGA LLM engine: keep the weights, cut the power between bursts** (medium)
1. **Idea:** design the KV260 accelerator for the idle gaps between agent or voice bursts, not for tokens per second. Keep the weights and state in DRAM and add a "warm standby" state (PL clock-gated or powered off, DRAM self-refresh) that resumes in milliseconds. Optimise energy per task including idle time. Your data supports this angle: 4.0 W idle, and static power is 70-81% of energy.
2. **Demand:**
   - Tandem (arXiv 2610.26660): a loaded model holds the phone NPU at top clocks, 0.76 W vs 0.25 W screen-off. Over a 5-minute keep-alive that wastes 120-150 J (12-15 agent steps); unloading means a 9 s cold start.
   - ollama #17958 (https://github.com/ollama/ollama/issues/17958): Strix Halo GPU pinned at about 30 W idle with 2 models resident, vs 6-8 W. Still open, no fix.
   - ollama PR #19788 (https://github.com/ollama/ollama/pull/19788): a loaded model keeps an A10 in P0, 58 W vs 16 W, about 12 kJ per default keep-alive.
3. **Prior work:**
   - Tandem only releases NPU power votes in software on a fixed SoC.
   - "Idle is the New Sleep" (arXiv 2407.12027) covers FPGA idle vs on-off for small DL accelerators. It cuts reconfiguration energy and has nothing on LLMs or resident weights and state.
   - AMD's Zynq US+ Power-Off Suspend wiki (https://xilinx-wiki.atlassian.net/wiki/spaces/A/pages/18842181/Zynq%2BUltraScale%2BMPSoC%2BPower%2BOff%2BSuspend): PL must be powered down, DRAM stays in self-refresh, resume is about Deep Sleep + 50 ms. It is off by default, built for ZCU102 only, tested on one board.
   - Open: no edge LLM accelerator treats resident-idle power as a design target or co-designs resume latency with agent and voice inter-arrival times.
4. **Fit:** uses your own KV260 decoder and power rig. Your static-power result becomes the motivation instead of a dead end. No GPU or Spark needed. No overlap with TeLLMe, Chimera or HeteroVLM.
5. **Kill test (1 day):** measure KV260 board power in five states:
   - bitstream loaded and idle (4.0 W)
   - PL clocks gated
   - bitstream unloaded (`xmutil unloadapp`)
   - Linux suspend-to-RAM
   - powered off

   Also time how long each takes to get back to the first token with weights still in DDR. **Dead if** the best weights-retained state saves less than about 1 W vs 4.0 W, or resumes in more than about 300 ms. For the energy math, replay the gaps between turns from your tau2/BFCL traces.
6. **Verdict: medium.** Biggest risk: the 4 W may be mostly PS, DDR and carrier board, leaving little PL power to cut. Second risk: it reads as power-management engineering unless the accelerator design changes too (for example, state laid out for fast save and restore, or keeping a small resident head).

**2. FPGA chunked Gated DeltaNet prefill that produces checkpoints where agent prompts diverge** (weak)
1. **Idea:** an edge-FPGA prefill engine for hybrid models whose chunked GDN dataflow can output the recurrent state at any token position cheaply. Agent prompts rewrite history, so the reuse point falls inside a chunk.
2. **Demand:**
   - llama.cpp #32316 (https://github.com/ggml-org/llama.cpp/issues/32316): Qwen3.5 re-prefills 20K tokens (42-48 s) per turn when checkpoints are off or too few.
   - llama.cpp discussion #30664 (https://github.com/ggml-org/llama.cpp/discussions/30664): 14.1K-token agent prefix recomputed every session (about 10 s); proposes divergence checkpoints.
3. **Prior work:**
   - Tandem (arXiv 2610.26660) already does divergence-point checkpoints plus a per-token tail on the phone NPU. Its rollback tax was 614 ms.
   - HA-NPU (arXiv 2609.32114) does hybrid prefill on NPUs.
   - Prasanna's group (arXiv 2603.05931) and arXiv 2611.18422 are decode-only GDN on FPGA.
   - The FPGA prefill side is open, but your own #1 result showed snapshots are cheap, so this is mostly a placement-policy problem.
4. **Fit:** your Qwen3.5-0.8B GDN decoder and agent traces. The risk is that it is the killed #1 under a new name.
5. **Kill test:** take your STRIDE tau2/BFCL traces and compute how many tokens get recomputed with checkpoints only at 64-token chunk boundaries vs exactly at the divergence point. **Dead if** the extra recompute is less than 5% of prefill tokens, or KV260 prefill is under 20% of step time.
6. **Verdict: weak.** Tandem covers the idea and the FPGA delta is small.

**Closest misses (taken):**
- **Thermal throttling of sustained decode:** ThermE (arXiv 2610.00267), CORE (MLSys'26) and HybridInfer (arXiv 2609.30270) cover it.
- **LLM interfering with perception and control on Jetson/Spark:** Ripple (arXiv 2611.01067), EdgePace (arXiv 2610.37307), MCSched and Dubot (arXiv 2611.30177) cover it.
- **On-device GUI-agent prefill, decode and screenshot reuse:** MobiInfer (arXiv 2611.08496), DroidPrefill (arXiv 2611.20772), GUIFlow (arXiv 2610.34959) and Tandem cover it.

I opened every link above. The only thing I couldn't fetch was the IEEE DOI page for the Prasanna paper, so I cite its arXiv version instead.
