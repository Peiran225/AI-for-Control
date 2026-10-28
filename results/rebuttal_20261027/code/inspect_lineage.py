"""Read-only inspection of recorded checkpoint ancestry and stage metadata."""
import argparse
import hashlib
import json
from pathlib import Path
import torch

OMIT = {"model_state", "model_state_dict", "state_dict", "optimizer_state", "optimizer_state_dict", "nominal_reference", "history"}

def scalar(v):
    if isinstance(v, torch.Tensor):
        return {"tensor_shape": list(v.shape), "dtype": str(v.dtype)}
    if isinstance(v, dict):
        return {str(k): scalar(x) for k,x in v.items()}
    if isinstance(v, (list,tuple)):
        return [scalar(x) for x in v]
    if isinstance(v, Path): return str(v)
    if isinstance(v, (str,int,float,bool)) or v is None: return v
    return str(v)

def digest(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""): h.update(block)
    return h.hexdigest()

def resolve_recorded(s, root):
    p=Path(s)
    candidates=[p, root/p]
    for marker in ("/repo/", "/AI-for-Control-direct-guided-20260727/"):
        if marker in str(p): candidates.append(root/str(p).split(marker,1)[1])
    return next((p.resolve() for p in candidates if p.is_file()), None)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("checkpoints",nargs="+")
    args=ap.parse_args()
    queue=[(x,"requested") for x in args.checkpoints]
    records={}; missing=[]
    while queue:
        raw,origin=queue.pop(0)
        p=resolve_recorded(raw,args.root)
        if p is None:
            missing.append({"recorded_path":raw,"referred_by":origin});continue
        if str(p) in records: continue
        if len(records)>120: raise RuntimeError("ancestry exceeds inspection limit")
        d=torch.load(p,map_location="cpu",weights_only=False)
        meta={k:scalar(v) for k,v in d.items() if k not in OMIT}
        rec={"path":str(p),"sha256":digest(p),"metadata":meta,"parents":[]}
        for key in ("source_checkpoint","initialization_checkpoint"):
            if isinstance(d.get(key),str) and d[key].endswith(".pt"):
                rec["parents"].append({"key":key,"path":d[key]})
        for scope in ("args","lm_args","refinement"):
            for k,v in d.get(scope,{}).items():
                if isinstance(v,str) and v.endswith(".pt") and ("checkpoint" in k or k in ("init","start")):
                    rec["parents"].append({"key":scope+"."+k,"path":v})
        summary=p.parent/"summary.json"
        if summary.exists():
            j=json.loads(summary.read_text())
            rec["summary"]={k:v for k,v in j.items() if k not in ("history","initial_validation_metrics_by_state","selected_train_metrics_by_state","selected_validation_metrics_by_state")}
            rec["summary_path"]=str(summary)
            hist=j.get("history")
            if isinstance(hist,list): rec["history_count"]=len(hist)
        records[str(p)]=rec
        queue.extend((e["path"],str(p)+":"+e["key"]) for e in rec["parents"])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({"records":records,"missing":missing},indent=2,default=scalar)+"\n")
    print(json.dumps({"records":len(records),"missing":len(missing),"output":str(args.output)}))

if __name__=="__main__":main()
