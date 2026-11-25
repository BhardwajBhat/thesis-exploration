#!/usr/bin/env python3
"""Low-overhead host/UMA pressure logger for the Spark governor kill test.
Run beside vLLM + agent tools; it does not allocate memory or alter processes.
"""
import argparse,json,os,subprocess,time
from pathlib import Path

def read(path):
 try:return Path(path).read_text()
 except:return None

def kv(text):
 out={}
 for line in (text or '').splitlines():
  p=line.split()
  if len(p)>=2:
   try:out[p[0].rstrip(':')]=int(p[1])*(1024 if len(p)>2 and p[2]=='kB' else 1)
   except:pass
 return out

def cmd(x):
 try:return subprocess.run(x,capture_output=True,text=True,timeout=2).stdout.strip()
 except:return None

def sample():
 procs=[]
 for d in Path('/proc').iterdir():
  if not d.name.isdigit():continue
  st=kv(read(d/'status'))
  if st.get('VmRSS',0)>100*1024*1024:
   procs.append({'pid':int(d.name),'name':(read(d/'comm') or '').strip(),'rss':st.get('VmRSS',0),'swap':st.get('VmSwap',0)})
 procs.sort(key=lambda x:x['rss'],reverse=True)
 return {'ts':time.time(),'meminfo':kv(read('/proc/meminfo')),'psi_memory':read('/proc/pressure/memory'),'psi_io':read('/proc/pressure/io'),
         'cgroup_memory_current':read('/sys/fs/cgroup/memory.current'),'cgroup_memory_events':read('/sys/fs/cgroup/memory.events'),
         'top_processes':procs[:30],'nvidia_smi':cmd(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'])}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--interval',type=float,default=1);ap.add_argument('--seconds',type=float,default=600);ap.add_argument('--out',default='uma_trace.jsonl');a=ap.parse_args()
 end=time.time()+a.seconds
 with open(a.out,'a') as f:
  while time.time()<end:
   t=time.time();f.write(json.dumps(sample())+'\n');f.flush();time.sleep(max(0,a.interval-(time.time()-t)))
if __name__=='__main__':main()
