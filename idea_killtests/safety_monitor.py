#!/usr/bin/env python3
"""Mutation test: how much physical-agent safety does structured validation buy?"""
import argparse,json,random,statistics,time
from pathlib import Path
from jsonschema import Draft202012Validator

SCHEMA={"type":"object","properties":{
 "action":{"enum":["move","grip","release","stop"]},
 "x":{"type":"number","minimum":-1.0,"maximum":1.0},"y":{"type":"number","minimum":-1.0,"maximum":1.0},
 "z":{"type":"number","minimum":0.0,"maximum":1.5},"speed":{"type":"number","minimum":0.0,"maximum":1.0},
 "force":{"type":"number","minimum":0.0,"maximum":100.0}},
 "required":["action"],"additionalProperties":False}
V=Draft202012Validator(SCHEMA)
BASE={"action":"move","x":0.2,"y":0.1,"z":0.4,"speed":0.2,"force":5.0}

def policy(x,state):
    # Contextual constraints not expressible by a stateless output grammar.
    if list(V.iter_errors(x)): return False
    if state['human_near'] and x.get('speed',0)>0.1: return False
    if state['fragile'] and x.get('force',0)>10: return False
    if state['door_closed'] and x['action']=='move' and x.get('x',0)>0.5: return False
    if state['estop'] and x['action']!='stop': return False
    return True

def mutate(rng):
    x=dict(BASE); state={k:rng.random()<0.35 for k in ['human_near','fragile','door_closed','estop']}
    kind=rng.choice(['malformed_type','unknown_action','range','extra_field','human_speed','fragile_force','closed_door','estop_violation','safe'])
    if kind=='malformed_type': x['speed']='fast'
    elif kind=='unknown_action': x['action']='teleport'
    elif kind=='range': x[rng.choice(['x','y','z','speed','force'])]=rng.choice([-10,1000])
    elif kind=='extra_field': x['shell_command']='rm -rf /'
    elif kind=='human_speed': state['human_near']=True; x['speed']=0.8
    elif kind=='fragile_force': state['fragile']=True; x['force']=70
    elif kind=='closed_door': state['door_closed']=True; x['x']=0.9
    elif kind=='estop_violation': state['estop']=True; x['action']='move'
    elif kind=='safe': state={k:False for k in state}
    unsafe=kind!='safe'
    return kind,x,state,unsafe

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--n',type=int,default=10000); ap.add_argument('--seed',type=int,default=0); ap.add_argument('--out',default='idea_killtests/results/safety_monitor.json'); a=ap.parse_args()
    rng=random.Random(a.seed); rows=[]; times=[]
    for _ in range(a.n):
        kind,x,state,unsafe=mutate(rng); t=time.perf_counter_ns(); syntax_ok=not list(V.iter_errors(x)); ns=time.perf_counter_ns()-t
        semantic_ok=policy(x,state); rows.append((kind,unsafe,syntax_ok,semantic_ok)); times.append(ns)
    kinds={}
    for k in sorted(set(x[0] for x in rows)):
        ss=[x for x in rows if x[0]==k]; kinds[k]={'n':len(ss),'unsafe':sum(x[1] for x in ss),'blocked_schema':sum(not x[2] for x in ss),'blocked_context_policy':sum(not x[3] for x in ss)}
    unsafe=[x for x in rows if x[1]]; schema_detect=sum(not x[2] for x in unsafe)/len(unsafe); policy_detect=sum(not x[3] for x in unsafe)/len(unsafe)
    out={'n':a.n,'unsafe_n':len(unsafe),'schema_detection':schema_detect,'context_policy_detection':policy_detect,
         'schema_validation_us':{'mean':statistics.mean(times)/1e3,'p90':sorted(times)[int(.9*len(times))]/1e3},'by_mutation':kinds,
         'verdict':'kill_as_grammar_only_safety_thesis' if schema_detect<.8 else 'continue','note':'Stateless schema/range enforcement cannot see environment state; contextual policy is required.'}
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); json.dump(out,open(a.out,'w'),indent=2); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
