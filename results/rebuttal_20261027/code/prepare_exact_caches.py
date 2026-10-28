"""Cache the same fixed-support queries using verified support K/V reuse."""
import json,os,sys,hashlib
from pathlib import Path
os.environ.setdefault('CUDA_VISIBLE_DEVICES','5')
os.environ['PMP_AUDIT_SOURCE_ROOT']='/home/mzhang/AI-for-Control-direct-guided-20260727'
ROOT=Path(os.environ['PMP_AUDIT_SOURCE_ROOT']);OUT=Path('/home/mzhang/pmp_rebuttal_sheet_20261027')
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np
import torch
from fixed_support_cache import CachedSupport,model_fingerprint
from generate_offgrid_policy_switching_diagnostics import load_time_model,raw_time_logits,fixed_support_query_logits
from evaluate_feedback_section5 import load_feedback_checkpoint

def main():
    torch.set_num_threads(1);torch.backends.mha.set_fastpath_enabled(False)
    records=json.loads((OUT/'inputs/source_checkpoint_hashes.json').read_text())['checkpoints']
    (OUT/'inputs/exact_caches').mkdir(exist_ok=True)
    evidence=[];known={};support=torch.linspace(0,1,801,dtype=torch.float64);query=torch.linspace(0,1,25601,dtype=torch.float64)
    for rec in records:
        p=Path(rec['checkpoint_path']);actual=hashlib.sha256(p.read_bytes()).hexdigest();assert actual==rec['declared_sha256']
        if rec['method']=='time':model,_,_=load_time_model(p)
        else:fb,_,_=load_feedback_checkpoint(p);model=fb.time_branch
        model=model.cpu().double().eval();fingerprint=model_fingerprint(model)
        if fingerprint not in known:
            cached=CachedSupport(model,support)
            with torch.inference_mode():
                raw=cached.query(query,chunk=512).numpy();supp=raw_time_logits(model,support).numpy()
                idx=torch.tensor([0,1,31,32,56,1643,9981,22000,25599,25600]);direct=fixed_support_query_logits(model,support,query[idx],batch_size=2).numpy()
            err=float(np.max(np.abs(raw[idx.numpy()]-direct)));err_support=float(np.max(np.abs(raw[::32]-supp)))
            assert max(err,err_support)<1e-10,(err,err_support)
            raw[::32]=supp;known[fingerprint]=(raw,supp,err,err_support)
        raw,supp,err,err_support=known[fingerprint]
        dest=OUT/'inputs/exact_caches'/f'{rec["method"]}_{rec["replicate"]}.npz'
        np.savez_compressed(dest,checkpoint_sha256=np.asarray(actual),dense_time=np.linspace(0,10,25601),dense_raw_logits=raw,support_raw_logits=supp)
        evidence.append({'checkpoint':str(p),'sha256':actual,'time_fingerprint':fingerprint,'cache':str(dest),'cached_vs_original_max_abs':err,'cached_vs_support_max_abs':err_support})
        print(dest.name,err,flush=True)
    (OUT/'evidence/exact_cache_verification.json').write_text(json.dumps(evidence,indent=2)+'\n')

if __name__=='__main__':main()
