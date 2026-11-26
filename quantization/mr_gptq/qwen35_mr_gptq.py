#!/usr/bin/env python3
"""MR-GPTQ (Micro-Rotated GPTQ, arXiv:2509.23202) for Qwen/Qwen3.5-0.8B, evaluated on the repo's
raw-WikiText 4096-prediction protocol (quantization/advanced_ppl.py::eval_ppl).

Recipe follows IST-DASLab/FP-Quant gptq_quantization() (commit d2e3092) step by step:
  1. block-diagonal Hadamard transforms per *input group* (inputs shared by several linears share
     one transform + one activation quantizer, like upstream qkv / gate_up);
  2. linears wrapped so the forward is  y = Qa(H x) @ (W H)^T   (rotation online on activations,
     fused into weights; H symmetric orthogonal so it is exact at full precision);
  3. NVFP4 weight global scale initialised from the rotated weight (optionally fused per group);
  4. block forward over calibration inputs: Hessians of the *rotated* inputs are collected while
     activation quantization is live (NVFP4 activation global scale tracked = min over calibration,
     then frozen -> static);
  5. weights rotated, GPTQ on fixed (MSE or absmax) FP4 grid with optional static act-order;
  6. quantized block re-run to produce the inputs of the next block (sequential propagation).

Qwen3.5 hybrid port (24 layers, 18 Gated-DeltaNet + 6 gated full-attention, all with SwiGLU MLP):
  GDN  'gdn_in'  : linear_attn.in_proj_qkv, in_proj_z, in_proj_b, in_proj_a   (input 1024)
       'gdn_out' : linear_attn.out_proj                                         (input 2048)
  attn 'qkv'     : self_attn.q_proj (incl. output gate rows), k_proj, v_proj    (input 1024)
       'o'       : self_attn.o_proj                                             (input 2048)
  mlp  'gate_up' : mlp.gate_proj, mlp.up_proj                                   (input 1024)
       'down'    : mlp.down_proj                                                (input 3584 = 28*128)
Not quantized (same as advanced_ppl defaults and upstream): embeddings, tied lm_head, conv1d,
norms, A_log/dt_bias, GDN recurrence, attention softmax / KV.

Deviations from upstream, all documented in README.md:
  * no fp16 autocast (--amp) during calibration: Qwen3.5 is run in bf16 throughout;
  * Hadamard via explicit fp32 Sylvester matmul (no fast_hadamard_transform build needed);
  * module wrapping instead of re-implementing Qwen3 attention classes (Qwen3.5 has no upstream class).

Usage (single GPU):
  .venv/bin/python quantization/mr_gptq/qwen35_mr_gptq.py --format nvfp --hadamard 16 \
      --w-observer mse --order activation --nsamples 128 --seqlen 2048 --output out.json
"""
import argparse, json, math, os, random, re, sys, time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import fpquant_port as fq  # noqa: E402

MODEL = "Qwen/Qwen3.5-0.8B"

# (regex on name inside a decoder layer) -> input group id
GROUP_RULES = [
    (r"^linear_attn\.in_proj_(qkv|z|b|a)$", "gdn_in"),
    (r"^linear_attn\.out_proj$", "gdn_out"),
    (r"^self_attn\.(q|k|v)_proj$", "qkv"),
    (r"^self_attn\.o_proj$", "o"),
    (r"^mlp\.(gate|up)_proj$", "gate_up"),
    (r"^mlp\.down_proj$", "down"),
]


def group_of(name):
    for rx, g in GROUP_RULES:
        if re.match(rx, name):
            return g
    return None


class MRQLinear(nn.Module):
    """y = act_quant(T x) @ W'^T (+b).  W' holds the rotated (and after GPTQ: dequantized) weight.
    Transform and activation quantizer objects are shared within an input group."""

    def __init__(self, lin: nn.Linear, transform, act_quantizer, name, group):
        super().__init__()
        self.in_features, self.out_features = lin.in_features, lin.out_features
        self.weight = nn.Parameter(lin.weight.detach().clone(), requires_grad=False)
        self.bias = None if lin.bias is None else nn.Parameter(lin.bias.detach().clone(), requires_grad=False)
        self.transform, self.act_quantizer = transform, act_quantizer
        self.name, self.group = name, group
        self.rotated = False      # weight already W H ?
        self.calls = 0

    def act(self, x):
        x = self.transform(x)
        if self.act_quantizer is not None:
            dt = x.dtype
            xf = x.float()   # quantize in fp32 (upstream: model/amp dtype), cast result back
            s, z = self.act_quantizer.get_quantization_params(xf)
            x = self.act_quantizer(xf, s, z).to(dt)
        return x

    def forward(self, x):
        self.calls += 1
        w = self.weight if self.rotated else self.transform(self.weight)
        return F.linear(self.act(x), w, self.bias)

    def extra_repr(self):
        return f"{self.in_features}->{self.out_features}, group={self.group}, rotated={self.rotated}"


def wrap_block(block, args, device):
    """Replace target linears in one decoder layer. Returns {name: MRQLinear}."""
    transforms, aq = {}, {}
    wrapped = {}
    for name, mod in list(block.named_modules()):
        if not isinstance(mod, nn.Linear):
            continue
        g = group_of(name)
        if g is None:
            raise RuntimeError(f"unmapped linear in decoder layer: {name}")
        if args.exclude_gdn and g.startswith("gdn"):
            continue
        if g not in transforms:
            transforms[g] = fq.build_transform(args.transform, args.hadamard)
            aq[g] = fq.make_quantizer(args.format, "minmax") if args.a_bits < 16 else None
        q = MRQLinear(mod, transforms[g], aq[g], name, g).to(device)
        parent, _, child = name.rpartition(".")
        setattr(block.get_submodule(parent), child, q)
        wrapped[name] = q
    return wrapped


# ----------------------------------------------------------------------------- calibration data
def get_calibration(tok, name, nsamples, seqlen, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    from datasets import load_dataset
    out = []
    if name == "fineweb-edu":   # identical to FP-Quant get_fineweb_edu
        ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
        ds = ds.shuffle(seed=seed, buffer_size=1_000)
        for sample in ds:
            ids = tok(sample["text"], return_tensors="pt").input_ids
            if ids.shape[1] < seqlen:
                continue
            i = random.randint(0, ids.shape[1] - seqlen)
            out.append(ids[:, i:i + seqlen])
            if len(out) >= nsamples:
                break
    elif name == "wikitext2-train":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        ids = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids
        for _ in range(nsamples):
            i = random.randint(0, ids.shape[1] - seqlen - 1)
            out.append(ids[:, i:i + seqlen])
    else:
        raise ValueError(name)
    return out


class _Stop(Exception):
    pass


@torch.no_grad()
def collect_layer0_inputs(model, samples, device):
    layers = model.model.layers
    store = []

    def hook(mod, args, kwargs):
        kw = {k: v for k, v in kwargs.items() if k not in ("past_key_values",)}
        store.append((args[0].detach(), kw))
        raise _Stop

    h = layers[0].register_forward_pre_hook(hook, with_kwargs=True)
    for s in samples:
        try:
            model(s.to(device), use_cache=False)
        except _Stop:
            pass
    h.remove()
    return store


# ----------------------------------------------------------------------------- main pipeline
@torch.no_grad()
def mr_gptq(model, samples, args, device, log):
    layers = model.model.layers
    inps = collect_layer0_inputs(model, samples, device)
    stats = []
    export = {}
    for li, block in enumerate(layers):
        t0 = time.time()
        wrapped = wrap_block(block, args, device)
        groups = defaultdict(list)
        for n, q in wrapped.items():
            groups[q.group].append(n)
        handles = {}
        hooks = []
        for n, q in wrapped.items():
            wq = fq.make_quantizer(args.format, args.w_observer)
            handles[n] = fq.GPTQ(q, wq, quantization_order=args.order, block_size=128, rel_damp=args.rel_damp)
            if wq.scale_precision == "e4m3":   # NVFP: freeze weight global scale from rotated weight
                wq.get_quantization_params(q.transform(q.weight).float())
                wq._track_global_scale = False

            def mk(name):
                def _h(mod, inp, out):
                    handles[name].update(mod.transform(inp[0]))   # Hessian of rotated input (pre act-quant)
                return _h
            hooks.append(q.register_forward_hook(mk(n)))
        if args.fuse_global_scale and args.format == "nvfp":
            for g, names in groups.items():
                gs = torch.stack([handles[n].quantizer.global_scale.to(device) for n in names]).min(0).values
                for n in names:
                    handles[n].quantizer.global_scale = gs.clone()
        # 5. calibration forward (BF16 weights, live activation quant; tracks act global scales)
        for x, kw in inps:
            block(x, **kw)
        for h in hooks:
            h.remove()
        # 6. rotate weights, freeze activation global scale
        for n, q in wrapped.items():
            q.weight.data = q.transform(q.weight.data)
            q.rotated = True
            if q.act_quantizer is not None:
                q.act_quantizer._track_global_scale = False
        # 7. GPTQ (or RTN on the same grid)
        for n, hd in handles.items():
            orig = hd.layer.weight.data.clone()
            if args.method == "gptq":
                dq, codes, scales = hd.quantize(export_codes=bool(args.save_dir))
            else:
                hd.W = hd.layer.weight
                dq, codes, scales = fq.rtn_quantize(hd.layer.weight.data, hd.quantizer)
            hinv_fb = getattr(hd, "hinv_fallback", False)
            # H is rebuilt with damping inside quantize(); rel-MSE uses the (damped) H as upstream
            H = hd.H if hd.H is not None else torch.eye(orig.shape[1], device=device)
            rel = fq.get_relative_mse_error(dq.float(), orig.float(), H).item()
            hd.layer.weight.data = dq.to(orig.dtype)
            stats.append({"layer": li, "name": n, "group": hd.layer.group, "rel_mse": rel,
                          "shape": list(orig.shape), "hinv_fallback": hinv_fb,
                          "w_global_scale": float(hd.quantizer.global_scale.reshape(-1)[0]),
                          "a_global_scale": (float(hd.layer.act_quantizer.global_scale.reshape(-1)[0])
                                             if hd.layer.act_quantizer is not None else None)})
            if args.save_dir:
                key = f"model.layers.{li}.{n}"
                export[key + ".qweight"] = fq.pack_fp4_to_uint8(codes.float()).cpu()
                export[key + ".scales"] = scales.float().cpu()
                export[key + ".weight_global_scale"] = hd.quantizer.global_scale.reshape(1).float().cpu()
                if hd.layer.act_quantizer is not None:
                    export[key + ".act_global_scale"] = hd.layer.act_quantizer.global_scale.reshape(1).float().cpu()
            del hd.H
        # 8. propagate quantized outputs
        inps = [(block(x, **kw), kw) for x, kw in inps]
        del handles
        torch.cuda.empty_cache()
        ls = [s for s in stats if s["layer"] == li]
        log(f"layer {li:2d} ({block.block_type:16s}) {len(ls)} linears  mean relMSE="
            f"{np.mean([s['rel_mse'] for s in ls]):.3e}  {time.time()-t0:.1f}s")
    return stats, export


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--format", choices=["nvfp", "mxfp"], default="nvfp")
    ap.add_argument("--a-bits", type=int, choices=[4, 16], default=4)
    ap.add_argument("--transform", choices=["hadamard", "identity"], default="hadamard")
    ap.add_argument("--hadamard", type=int, default=None, help="Hadamard block size (default: 16 nvfp / 128 mxfp)")
    ap.add_argument("--method", choices=["gptq", "rtn"], default="gptq")
    ap.add_argument("--w-observer", choices=["minmax", "mse"], default="mse")
    ap.add_argument("--order", choices=["default", "activation"], default="activation")
    ap.add_argument("--rel-damp", type=float, default=1e-2)
    ap.add_argument("--fuse-global-scale", action="store_true")
    ap.add_argument("--exclude-gdn", action="store_true")
    ap.add_argument("--calib", choices=["fineweb-edu", "wikitext2-train"], default="fineweb-edu")
    ap.add_argument("--nsamples", type=int, default=128)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-tokens", type=int, default=4096)
    ap.add_argument("--eval-seqlen", type=int, default=512)
    ap.add_argument("--save-dir", default=None, help="write packed FP4 codes/scales (safetensors)")
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    if a.hadamard is None:
        a.hadamard = 16 if a.format == "nvfp" else 128

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from advanced_ppl import eval_ppl, URL
    device = "cuda"
    lines = []

    def log(s):
        print(s, flush=True); lines.append(s)

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
    model.requires_grad_(False)
    samples = get_calibration(tok, a.calib, a.nsamples, a.seqlen, a.seed)
    log(f"calibration: {len(samples)} x {a.seqlen} from {a.calib} ({time.time()-t0:.0f}s)")
    stats, export = mr_gptq(model, samples, a, device, log)
    tq = time.time() - t0
    nll, ppl, n = eval_ppl(model, tok, a.eval_tokens, a.eval_seqlen, device)
    calls = [m.calls for m in model.modules() if isinstance(m, MRQLinear)]
    nparams = sum(s["shape"][0] * s["shape"][1] for s in stats)
    res = {"model": MODEL, "dataset": URL,
           "protocol": f"first {a.eval_tokens+1} tokenizer tokens, independent {a.eval_seqlen}-token windows, "
                       f"{a.eval_tokens} next-token predictions, FP32 cross entropy",
           "recipe": "MR-GPTQ port of IST-DASLab/FP-Quant@d2e3092", "args": vars(a),
           "nll": nll, "ppl": ppl, "tokens": n,
           "quantized_linears": len(stats), "quantized_weight_params": nparams,
           "min_calls_during_eval_and_calib": min(calls) if calls else 0,
           "hinv_fallbacks": sum(s["hinv_fallback"] for s in stats),
           "quant_seconds": tq, "total_seconds": time.time() - t0, "layer_stats": stats, "log": lines}
    log(f"RESULT {a.format} W4A{a.a_bits} {a.method} T={a.transform}{a.hadamard} obs={a.w_observer} "
        f"order={a.order} n={a.nsamples}x{a.seqlen}: ppl={ppl:.3f}  ({len(stats)} linears, {nparams/1e6:.1f}M w)")
    res["log"] = lines
    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
    open(a.output, "w").write(json.dumps(res, indent=2) + "\n")
    if a.save_dir:
        from safetensors.torch import save_file
        os.makedirs(a.save_dir, exist_ok=True)
        save_file(export, os.path.join(a.save_dir, "mr_gptq_fp4.safetensors"),
                  metadata={"format": a.format, "hadamard": str(a.hadamard), "transform": a.transform})
        log(f"saved {len(export)} tensors to {a.save_dir}")


if __name__ == "__main__":
    main()
