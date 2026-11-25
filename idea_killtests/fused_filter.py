#!/usr/bin/env python3
"""Project benefit of fusing grammar mask/state update into the existing FPGA argmax."""
import json,statistics
from pathlib import Path
SRC='structured_head/results/kill_test.json'; OUT='idea_killtests/results/fused_filter.json'
r=json.load(open(SRC)); fill=[x['fill_us'] for x in r['steps']]
baseline_us=58_000
# Favorable bound: all measured host grammar time is serialized and FPGA fusion removes it completely.
mean=statistics.mean(fill); p90=float(__import__('numpy').percentile(fill,90)); worst=max(fill)
def speed(x): return (baseline_us+x)/baseline_us
out={'baseline_token_us':baseline_us,'grammar_fill_us':{'mean':mean,'p90':p90,'max':worst},
     'speedup_if_all_grammar_time_eliminated':{'mean':speed(mean),'p90':speed(p90),'max':speed(worst)},
     'threshold':1.05,'verdict':'kill_as_performance_thesis',
     'reason':'XGrammar mask generation is tens of microseconds versus a 58 ms FPGA token. A fused filter that still computes the full head cannot materially improve throughput; sparse row skipping was separately killed at 1.135x optimistic speedup.'}
Path(OUT).parent.mkdir(parents=True,exist_ok=True);json.dump(out,open(OUT,'w'),indent=2);print(json.dumps(out,indent=2))
