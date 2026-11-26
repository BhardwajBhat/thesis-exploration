**Verdict: nothing in this area is strong.** The space filled up between Aug and Nov 2026. One medium idea survives, and it is a narrower version of your existing idea **M**. The other two are weak.

**1. Hard-isolated dual-system VLA on one Versal (a narrower M)** — *medium*
- **Idea:** the slow vision-language model (System 2) runs on the AIE with HeteroVLM's preemptible schedule. The fast flow-matching action head (System 1) runs on PL GEMV engines as a fixed-period, bounded-jitter loop. One chip does what Figure Helix needs two GPUs for, and the AIE's load cannot disturb the action head.
- **Demand:**
  - Helix puts System 2 and System 1 "each running on dedicated GPUs" to hold a 200 Hz loop: https://www.figure.ai/news/helix
  - DASH (2611.11854) runs System 2 and System 1 as concurrent processes: synchronous control drops from 9.7 to 4.1 Hz as the backbone grows from 0.8B to 8B.
  - Dubot (2611.30177): VLA and MPC serialize on Jetson Orin/Thor and miss control deadlines.
  - openpi #393: the robot "stops waiting until a new inference" at 86 ms: https://github.com/Physical-Intelligence/openpi/issues/393
- **Closest prior work:**
  - EdgeDAE (2610.00311): FPGA action head plus a separate GPU, Octo only, no co-scheduling.
  - 2611.24234: action-head accelerator in RTL simulation, no System 2 sharing the chip.
  - KerColle (2609.22335): GPU throughput only, simulation.
  - Dubot: GPU-only co-location.
  - None gives a jitter bound for System 1 while System 2 shares the chip.
- **Fit:** builds directly on HeteroVLM's AIE/PL split, preemption and DDR resume. Lab owns the VCK5000.
- **One-day kill test (2x RTX 4090, LeRobot pi0.5 or SmolVLA):**
  - Run the action-expert loop as a high-priority CUDA stream (and separately under MPS).
  - Run continuous VLM prefills next to it.
  - Measure System 1 p99 latency against running alone.
  - **Dead if** p99 is ≤ 1.2x alone with priority streams alone, i.e. software fixes it.
  - Also estimate the PL cost: action-head bytes per step over the VCK5000's DDR bandwidth. **Dead if** > 20 ms per chunk.
- **Biggest risks:** the 4090 is a weak stand-in for Jetson's shared DRAM. Dubot or Ripple-style software may be "good enough". It needs the advisor to agree.

**2. KV260 action head with precision rising over the flow steps** — *weak (scooped)*
- 2611.24234 (Nov 18) already shows this. Precision rising from 2 to 4 bits over the flow steps keeps closed-loop success. "One stored 8-bit copy replaces the copies of all widths" — that is your nested-precision code. Its RTL runs the GR00T N1.6 head in 8.02 ms (simulated).
- **What's left:** the first real-silicon prototype of that schedule, on the KV260, for the SmolVLA expert (~100M params, per the Cornell FCCM'26 SmolVLA-NPU paper).
- **Kill test:** run your nested GEMV at the 50-token action-head shapes for a 10-step 2→4-bit schedule. **Dead if** < 1.3x over 8-bit, which is likely given your 10-12 ms per pass of fixed cost and 70-81% static power.
- **Risk:** this is a replication, not a thesis.

**3. Sensor-to-token preprocessing in PL** — *weak*
- **Demand:** VLA-HWBench (2610.34717) finds CPU preprocessing is 39% of the optimized GR00T N1.7 call on Thor, mostly serial work on one Arm core.
- **Problem:** the paper itself blames host software and thread setup. Moving preprocessing onto the GPU or Jetson's PVA likely fixes it, so a hardware contribution is hard to defend.
- **Kill test:** time GR00T preprocessing on a 4090 host with all threads, then moved to GPU. **Dead if** it becomes < 10% of the call.

**Taken or crowded (don't revisit):**
- Action-head and VLA accelerators: Deltoris (2608.04428), EdgeVLA (ISLPED'26), a TCSI flow-predictive FPGA accelerator, 2611.11098.
- VLA measurement: VLA-HWBench, the ixian/ZETIC dashboard, VLA-Perf.
- GPU co-scheduling of VLA work: Dubot, Ripple, Linux AGX (SOSP'26), KerColle.
- World-action-model compression: SmolWAM, RealtimeWAM.

Real demand on the Jetson forum is mostly people asking for faster engines on smaller boards, e.g. π0.5 is ~5x slower on emulated T3000 than on T5000: https://forums.developer.nvidia.com/t/why-is-0-5-inference-about-5x-slower-under-t3000-software-emulation-on-jetson-thor/388625 . That pull is toward GPU kernels, not FPGAs. I opened every link cited here, except that for EdgeVLA and the TCSI accelerator I saw only search abstracts; IEEE Xplore blocked the fetch.
