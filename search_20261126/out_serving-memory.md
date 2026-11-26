Short version: one idea in my area is worth a one-day test, one is mostly taken, and nothing in my area is strong.

**1. A model and scheduler for the host-to-GPU KV path, built around copy-engine stalls**
- **Idea:** The CPU→GPU KV-load path is now a hardware bottleneck in its own right. The GPU's DMA copy engine can't process the descriptor queue fast enough, so one large prefix load blocks unrelated small copies. And whether DMA or an SM-driven copy (Triton/TMA kernel) is faster flips with page size, batch size and machine. Thesis: measure and model this (copy engine vs SM copy vs PCIe gen, on A100 and 4090 plus the published H100/B200/GB10 numbers), then build a scheduler in vLLM that picks the copy path, slices large loads across engine steps, ranks loads by priority, and recomputes instead of loading when the load is short.
- **Demand:**
  - vllm#65515 (opened 2026-11-03): a 6.6 GB load as 161k descriptors stalls the worker 110 ms. Slicing raises GLM-5.2 on 8xB200 from 706 to 859 tok/s and cuts mean TTFT from 23.8 to 17.4 s. https://github.com/vllm-project/vllm/issues/65515
  - vllm#57555 (open, active through 2026-11-15): the DMA/Triton cutoff was hard-coded from one H100 and is wrong on A30, GB10 and 5060 Ti, by up to 12x. On a 5060 Ti, the copy engine held a step's 64 KiB input copy behind a load: 74–209 ms per step against 5.7 ms alone. The calibration PR #57569 was closed without merging. https://github.com/vllm-project/vllm/issues/57555
  - vllm#65161: with EAGLE, 68 of 71 voice turns waited about 120 ms for a one-block CPU-tier load that would have been cheaper to recompute. https://github.com/vllm-project/vllm/issues/65161
- **Prior work:**
  - Strata (2508.18572, OSDI'26) handles layout fragmentation and cache-aware scheduling, but not copy-engine blocking of foreground copies or how the right path changes by machine.
  - GPUWeaver (APSys'26) covers RDMA vs host-device PCIe contention only.
  - HiPerfCC-KVCache (2611.28702) covers driver-lock stalls in confidential-computing mode only.
  - SGLang PR #40278 is a TMA copy kernel for Hopper and newer GPUs only.
  - Open: a measured, per-machine cost model (copy engine / SM copy / link / recompute) that drives decisions per step, under real load.
- **Fit:** This is computer architecture work (copy-engine queues, SM write ports, PCIe Gen4) that you can measure on your 4xA100 and 2x4090. PCIe Gen4 is where the defaults are most wrong. It links to STRIDE (agent traffic) and carries over to UMA/FPGA DMA later.
- **Kill test (1 day):** Standalone microbenchmark (the upstream `_swap_blocks_kernel` vs `cuMemcpyBatchAsync`, both on pinned memory). Run a 1–6 GB load of 16–64 KiB pages while a stream of decode-like kernels issues small H2D copies, on an A100 and a 4090. Then replay an agentic trace through the OffloadingConnector with and without slicing plus Triton loads.
  - Dead if foreground p99 step time rises by less than 1.3x, or if the fix gains less than 10% tok/s / TTFT on both cards.
- **Verdict: medium.**
  - Biggest risk: vLLM contributors are already patching it piece by piece (#65515, #57569, #66527), so it could turn into an engineering PR rather than a thesis.
  - Second risk: `cuMemcpyBatchAsync` needs a newer vLLM than 0.19.1 and CUDA 12.8 forward-compat libraries on driver 535. The A30 report in #57555 shows that setup works.

**2. Host-tier coupling of KV and recurrent state for hybrid GDN models**
- **Idea:** For hybrid models, the host tier has to budget full-attention KV and recurrent-state checkpoints together. Otherwise KV still sitting in host memory gets no hits.
- **Demand:**
  - sglang#45400: KV is still on host, but the 56 mamba-state slots run out, so hits go to 0 and TTFT goes from 0.29 to 23 s. Reproduced independently on an A100. https://github.com/sgl-project/sglang/issues/45400
  - vllm#52773: when the offloaded boundary state isn't consulted, a hit can collapse to zero. Loading that state would be about 80x cheaper than recomputing. https://github.com/vllm-project/vllm/issues/52773
- **Prior work:** HiState (2610.41508; checkpoints on SSD, unified eviction, vLLM, agent traces) covers most of it. GDNShift (2611.26153) covers migration.
- **Kill test:** Run #45400's repro on an A100 against HiState's policy. Dead if HiState recovers 90% or more of the hits.
- **Verdict: weak.** HiState already took the core; what's left is bug fixing.

**Closest misses (no survivor):** SmartNIC/FPGA KV fetch (ShadowServe 2509.16857 already did it), sparse-attention KV recall (PulseInfer 2609.34555, AVSG 2609.37538), and PCIe contention between RDMA and offload (GPUWeaver).

Only idea 1 is worth a day. Run its kill test on the 4090s first, since PCIe Gen4 is where the reported defaults are most wrong. If p99 step inflation is under 1.3x there, drop my area and go back to #2 (the Spark governor).
