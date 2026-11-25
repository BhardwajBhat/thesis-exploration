#!/usr/bin/env python3
"""Kill test: does batch composition change Qwen3.5 agent trajectories on an RTX GPU?"""
import argparse,json,os,time
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM

PROMPTS=[
 "Respond with JSON only. Tools: get_weather(city, unit). User: What is the weather in Irvine in celsius?",
 "Solve step by step: A train travels 180 miles in 3 hours. What is its average speed?",
 "Write a Python function that validates a balanced-parentheses string.",
 "You can call run_sql(query). Count orders created in November 2026. Return one JSON tool call.",
 "Explain why LLM decode is memory-bandwidth bound in three concise sentences.",
]

def encode(tok,s): return tok(s,return_tensors='pt',add_special_tokens=True).input_ids[0]

def distractors(target,B,vocab,seed):
 g=torch.Generator().manual_seed(seed); L=target.numel(); rows=[target]
 # Use normal token IDs, avoiding padded model-only vocabulary rows.
 for _ in range(B-1): rows.append(torch.randint(0,vocab,(L,),generator=g))
 return torch.stack(rows).cuda()

@torch.inference_mode()
def run(m,ids,n):
 out=m(ids,use_cache=True); cache=out.past_key_values; logits=out.logits[:,-1].float(); rec=[]
 for step in range(n):
  vals,ix=logits[0].topk(2); nxt=logits.argmax(-1,keepdim=True)
  rec.append({'token':int(nxt[0]),'margin':float(vals[0]-vals[1]),'top_logit':float(vals[0])})
  out=m(nxt,past_key_values=cache,use_cache=True);cache=out.past_key_values;logits=out.logits[:,-1].float()
 return rec

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',default='Qwen/Qwen3.5-0.8B');ap.add_argument('--gen',type=int,default=64)
 ap.add_argument('--batches',default='2,4,8');ap.add_argument('--dtype',choices=['bf16','fp32'],default='bf16');ap.add_argument('--out',default='determinism/results/batch_invariance.json');a=ap.parse_args()
 torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=(a.dtype!='fp32')
 tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
 dtype=torch.bfloat16 if a.dtype=='bf16' else torch.float32
 m=AutoModelForCausalLM.from_pretrained(a.model,dtype=dtype,local_files_only=True).cuda().eval()
 rows=[];t0=time.time()
 for pi,p in enumerate(PROMPTS):
  ids=encode(tok,p).cuda();base=run(m,ids[None],a.gen)
  for B in [int(x) for x in a.batches.split(',')]:
   bat=distractors(ids.cpu(),B,len(tok),1000*pi+B); got=run(m,bat,a.gen)
   div=next((i for i,(x,y) in enumerate(zip(base,got)) if x['token']!=y['token']),None)
   same=sum(x['token']==y['token'] for x,y in zip(base,got))
   logit_delta=max(abs(x['top_logit']-y['top_logit']) for x,y in zip(base,got))
   rows.append({'prompt':pi,'batch':B,'same_tokens':same,'gen':a.gen,'first_divergence':div,'max_selected_logit_delta':logit_delta,
                'min_base_margin':min(x['margin'] for x in base),'base_tokens':[x['token'] for x in base],'batch_tokens':[x['token'] for x in got]})
   print(f'[{time.time()-t0:5.1f}s] p{pi} B{B}: same={same}/{a.gen} first={div} dlogit={logit_delta:.6g}',flush=True)
 agg={'comparisons':len(rows),'token_exact_comparisons':sum(r['same_tokens']==a.gen for r in rows),
      'trajectory_divergences':sum(r['same_tokens']<a.gen for r in rows),'total_tokens':len(rows)*a.gen,
      'different_tokens':sum(a.gen-r['same_tokens'] for r in rows),'max_logit_delta':max(r['max_selected_logit_delta'] for r in rows)}
 agg['verdict']='kill_hardware_determinism_motivation' if agg['trajectory_divergences']==0 else 'continue'
 out={'model':a.model,'gpu':torch.cuda.get_device_name(0),'dtype':a.dtype,'prompts':len(PROMPTS),'aggregate':agg,'rows':rows,
      'limitations':['Transformers eager model, not vLLM continuous batching.','Greedy decoding only. A second test must vary vLLM scheduling and sampled rollouts if this shows no divergence.']}
 Path(a.out).parent.mkdir(parents=True,exist_ok=True);json.dump(out,open(a.out,'w'),indent=2);print(json.dumps(agg,indent=2));print('# wrote',a.out)
if __name__=='__main__':main()
