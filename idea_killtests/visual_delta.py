#!/usr/bin/env python3
"""Software headroom test for visual-delta front ends on GUI agent trajectories."""
import argparse, io, json, os, statistics, time
from pathlib import Path
import numpy as np
import requests
from PIL import Image

API="https://huggingface.co/api/datasets/minuzero/GUI-CC/tree/main/offline_data/{:03d}?recursive=true&expand=false"
RAW="https://huggingface.co/datasets/minuzero/GUI-CC/resolve/main/{}"

def get(url):
    r=requests.get(url,timeout=60); r.raise_for_status(); return r

def patch_fraction(a,b,patch,threshold):
    h=min(a.shape[0],b.shape[0]); w=min(a.shape[1],b.shape[1]); h-=h%patch; w-=w%patch
    d=np.abs(a[:h,:w].astype(np.float32)-b[:h,:w].astype(np.float32)).mean(2)/255
    ph,pw=h//patch,w//patch
    scores=d.reshape(ph,patch,pw,patch).mean((1,3))
    return float((scores>threshold).mean()), float(scores.mean())

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--episodes',type=int,default=20); ap.add_argument('--patch',type=int,default=16)
    ap.add_argument('--thresholds',default='0,0.002,0.005,0.01,0.02'); ap.add_argument('--out',default='idea_killtests/results/visual_delta.json'); a=ap.parse_args()
    ths=[float(x) for x in a.thresholds.split(',')]; rows=[]
    for ep in range(1,a.episodes+1):
        files=get(API.format(ep)).json(); imgs=sorted(x['path'] for x in files if x['path'].endswith('.png'))
        prev=None
        for j,p in enumerate(imgs):
            im=np.asarray(Image.open(io.BytesIO(get(RAW.format(p)).content)).convert('RGB'))
            if prev is not None and im.shape==prev.shape:
                z={'episode':ep,'transition':j-1,'shape':list(im.shape),'thresholds':{}}
                for t in ths:
                    f,mae=patch_fraction(prev,im,a.patch,t); z['thresholds'][str(t)]={'changed_fraction':f,'mean_pixel_delta':mae}
                rows.append(z)
            prev=im
        print(f'episode {ep:03}: {len(imgs)} frames',flush=True)
    agg={}
    for t in ths:
        xs=[r['thresholds'][str(t)]['changed_fraction'] for r in rows]
        agg[str(t)]={'n':len(xs),'mean':statistics.mean(xs),'median':statistics.median(xs),'p75':float(np.percentile(xs,75)),'p90':float(np.percentile(xs,90)),
                     'ideal_encoder_speedup':1/statistics.mean(xs) if statistics.mean(xs)>0 else None}
    # Practical reuse must occasionally refresh a full frame. Charge one full frame every R steps.
    refresh={}
    for R in [4,8,16]:
        refresh[str(R)]={}
        for t in ths:
            f=agg[str(t)]['mean']; effective=1/R+(1-1/R)*f
            refresh[str(R)][str(t)]={'effective_fraction':effective,'ideal_speedup':1/effective}
    out={'dataset':'minuzero/GUI-CC offline','episodes':a.episodes,'patch':a.patch,'aggregate':agg,'periodic_refresh':refresh,'rows':rows,
         'limitations':['Pixel-delta patch reuse is an optimistic compute bound; VLM token equivalence/accuracy is not tested.','Download sample is first N GUI-CC episodes.','Existing GUI-KV work already covers spatiotemporal KV reuse, so headroom alone does not establish novelty.']}
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); json.dump(out,open(a.out,'w'),indent=2)
    print(json.dumps({'aggregate':agg,'periodic_refresh':refresh},indent=2)); print('# wrote',a.out)
if __name__=='__main__': main()
