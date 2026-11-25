#!/usr/bin/env python3
"""Cheap oracle bound for hiding code-environment time with exact next-turn speculation.
Uses RollPacker's public KodCode evaluator-time trace. This is a bound, not a full agent trace.
"""
import argparse,json,statistics,urllib.request
from pathlib import Path
import numpy as np
URL='https://raw.githubusercontent.com/Farrrrland/RollPacker/main/data/code_KodCode_data_with_time.jsonl'

def load():
 data=urllib.request.urlopen(URL,timeout=60).read().decode(); xs=[]
 for l in data.splitlines():
  x=json.loads(l);t=x.get('reference_time')
  if isinstance(t,(int,float)) and t>=0:xs.append(float(t))
 return np.array(xs)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',default='rl_spec/results/oracle_bound.json');a=ap.parse_args();t=load()
 rows=[]
 # g = target next-turn LLM duration. Accuracy is probability speculation is useful/exact.
 for g in [.1,.25,.5,1,2,4]:
  for acc in [.5,.7,.9,1.0]:
   for overhead in [0,.02,.05]:
    baseline=t+g
    # Successful speculation overlaps min(tool,g); failed attempts give no overlap and pay fork/verification overhead.
    new=baseline-acc*np.minimum(t,g)+overhead
    rows.append({'next_llm_s':g,'gate_accuracy':acc,'overhead_s':overhead,'aggregate_speedup':float(baseline.sum()/new.sum()),
                 'median_task_speedup':float(np.median(baseline/new)),'p90_task_speedup':float(np.percentile(baseline/new,90)),
                 'fraction_tasks_1p15':float(np.mean(baseline/new>=1.15))})
 out={'source':URL,'n':len(t),'tool_s':{'mean':float(t.mean()),'median':float(np.median(t)),'p90':float(np.percentile(t,90)),'p95':float(np.percentile(t,95)),'p99':float(np.percentile(t,99)),'max':float(t.max())},'rows':rows,
      'interpretation':'Most evaluator calls are ~31 ms; gains come from a tiny heavy tail. This trace has one environment call per rollout, not multi-turn tool-agent timing, so it is a favorable screening bound only.'}
 Path(a.out).parent.mkdir(parents=True,exist_ok=True);json.dump(out,open(a.out,'w'),indent=2)
 print(json.dumps(out['tool_s'],indent=2))
 for r in rows:
  if r['gate_accuracy'] in (.7,.9,1) and r['overhead_s']==.02 and r['next_llm_s'] in (.5,1,2): print(r)
 print('# wrote',a.out)
if __name__=='__main__':main()
