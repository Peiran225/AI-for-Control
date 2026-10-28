"""Evaluate the four mandated before/after snapshots once they are saved."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
ROOT=Path('/home/mzhang/AI-for-Control-direct-guided-20260727');OUT=Path('/home/mzhang/pmp_rebuttal_sheet_20261027');RUN=OUT/'noncf_seed4_fb31'
os.environ['PMP_AUDIT_SOURCE_ROOT']=str(ROOT)
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np
import torch
from fixed_support_cache import CachedSupport
from generate_offgrid_policy_switching_diagnostics import load_time_model,raw_time_logits,fixed_support_query_logits,time_action_from_raw
from evaluate_feedback_section5 import load_feedback_checkpoint
from evaluate_table1_heldout_common import Problem,load_policy,rollout_controls

def main():
    torch.set_num_threads(1);torch.backends.mha.set_fastpath_enabled(False)
    jobs=[('time_before','time',RUN/'temporal_supervised.pt'),
          ('time_after','time',RUN/'temporal_lm_equal/selected_checkpoint.pt'),
          ('der_before','der',RUN/'feedback_supervised/feedback_direct_initialization.pt'),
          ('der_after','der',RUN/'der_m8/best_feedback_section5.pt')]
    (OUT/'new_evaluation').mkdir(exist_ok=True);summaries=[]
    for label,method,checkpoint in jobs:
        start=time.monotonic()
        while not checkpoint.exists():
            if (RUN/'run.json').exists():
                meta=json.loads((RUN/'run.json').read_text())
                if any(x.get('status')=='FAILED' for x in meta.get('stages',[])):
                    raise RuntimeError('training FAILED; snapshot unavailable: '+label)
            if time.monotonic()-start>8*3600:raise RuntimeError('snapshot wait exceeded 8h: '+label)
            time.sleep(10)
        if method=='time':model,cfg,wrapper=load_time_model(checkpoint)
        else:fb,cfg,_=load_feedback_checkpoint(checkpoint);model=fb.time_branch;wrapper={}
        model=model.cpu().double().eval();support=torch.linspace(0,1,801,dtype=torch.float64);times=torch.linspace(0,1,25601,dtype=torch.float64)
        with torch.inference_mode():
            cache=CachedSupport(model,support);raw=cache.query(times,chunk=512).numpy();supp=raw_time_logits(model,support).numpy()
            idx=torch.tensor([1,29,1609,7281,21001,25599]);direct=fixed_support_query_logits(model,support,times[idx],batch_size=2).numpy()
        error=float(np.max(np.abs(raw[idx.numpy()]-direct)));assert error<1e-10;raw[::32]=supp
        cpath=OUT/'new_evaluation'/f'{label}_cache.npz'
        np.savez_compressed(cpath,checkpoint_sha256=np.asarray(hashlib.sha256(checkpoint.read_bytes()).hexdigest()),dense_time=np.linspace(0,10,25601),dense_raw_logits=raw,support_raw_logits=supp)
        dest=OUT/'new_evaluation'/label
        if not (dest/'summary.json').exists():
            cmd=[sys.executable,'-u',str(OUT/'code/evaluate_frozen_s4.py'),'--source-root',str(ROOT),'--checkpoint',str(checkpoint),'--method',method,'--cache',str(cpath),'--out',str(dest)]
            with (OUT/'logs'/f'{label}_evaluation.log').open('w') as log:subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT)
        p=Problem();initial=np.stack([np.full(21,10.),np.linspace(8,12,21)])
        query=None;close=lambda:None;u=None
        if method=='time':u=time_action_from_raw(supp,cfg,wrapper)[:-1]
        else:query,close,_=load_policy('pmp_kkt_der',checkpoint,hjb_tau=10.)
        _,J=rollout_controls(initial,p,p.vectors(),fixed_control=u,query=query,objective_substeps=4);close()
        rows=json.loads((dest/'summary.json').read_text())['rows']
        for index,row in enumerate(rows):row['J_original_n800']=float(J[index]);row['snapshot']=label
        summaries.extend(rows)
        (OUT/'new_evaluation/before_after.json').write_text(json.dumps(summaries,indent=2)+'\n')
        print(label,'completed',[(r['J_original_n800'],r['epsilon']) for r in rows],flush=True)

if __name__=='__main__':main()
