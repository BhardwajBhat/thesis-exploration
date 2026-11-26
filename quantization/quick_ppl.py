#!/usr/bin/env python3
"""Quick identical-window WikiText-2 PPL comparison for Qwen3.5-0.8B-Base."""
import argparse, copy, json, math, time, urllib.request
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL="Qwen/Qwen3.5-0.8B"
URL="https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/test.txt"

def eval_ppl(model,tok,tokens=4096,seqlen=512):
 text=urllib.request.urlopen(URL).read().decode()
 ids=tok(text,return_tensors='pt',add_special_tokens=False).input_ids[0][:tokens+1]
 total=0.; n=0
 with torch.inference_mode():
  for s in range(0,len(ids)-1,seqlen):
   x=ids[s:min(s+seqlen,len(ids)-1)].unsqueeze(0).cuda(); y=ids[s+1:min(s+seqlen+1,len(ids))].cuda()
   out=model(x,use_cache=False,logits_to_keep=0).logits[0].float()
   total += torch.nn.functional.cross_entropy(out,y,reduction='sum').item(); n += y.numel()
 return total/n, math.exp(total/n), n

def quantize_linear_rtn(model,bits):
 count=0
 with torch.no_grad():
  for mod in model.modules():
   if isinstance(mod,torch.nn.Linear):
    w=mod.weight.data.float(); qmax=(1<<(bits-1))-1
    scale=w.abs().amax(dim=1,keepdim=True).clamp_min(1e-8)/qmax
    mod.weight.data.copy_((w/scale).round().clamp(-qmax,qmax).mul(scale).to(mod.weight.dtype)); count += w.numel()
 return count

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--modes',nargs='+',default=['bf16','int8','int4']); ap.add_argument('--tokens',type=int,default=4096); ap.add_argument('--output',default='quant_results.json'); a=ap.parse_args()
 tok=AutoTokenizer.from_pretrained(MODEL)
 results={}
 # Reload per mode so quantization is not cumulative.
 for mode in a.modes:
  torch.cuda.empty_cache(); t=time.time()
  m=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
  if mode.startswith('int'): quantize_linear_rtn(m,int(mode[3:]))
  nll,ppl,n=eval_ppl(m,tok,a.tokens)
  results[mode]={'nll':nll,'ppl':ppl,'tokens':n,'seconds':time.time()-t,'definition':'BF16' if mode=='bf16' else 'per-output-channel symmetric RTN, dequantized BF16 eval'}
  print(mode,results[mode],flush=True); del m
 open(a.output,'w').write(json.dumps(results,indent=2)+'\n')
if __name__=='__main__': main()
