"""Return additional diagnostics from the unchanged original S4 integrator."""
from __future__ import annotations
import argparse
import csv
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source-root',type=Path,required=True)
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--method',choices=['time','cf','der'],required=True)
    ap.add_argument('--cache',type=Path)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--directions',type=Path)
    ap.add_argument('--steps',type=int,default=25600)
    ap.add_argument('--batch-size',type=int,default=16)
    ap.add_argument('--threads',type=int,default=1)
    a=ap.parse_args();sys.path[:0]=[str(a.source_root),str(a.source_root/'scripts')]
    import numpy as np
    import evaluate_table1_feedback_continuous_strict_fixed_rk4 as original
    from evaluate_table1_heldout_continuous_pmp import actions_batch
    from evaluate_table1_baseline_continuous_strict import strict_quantities_batch,common_parameters
    from evaluate_table1_heldout_common import Problem,load_directions
    from build_final_der_phenotype_3d_figure import switching_records
    p=Problem();q=np.linspace(0,p.T,25601);mask=(q>=1.5)&(q<8)&(np.arange(25601)%32!=0)
    assert int(mask.sum())==16120
    # Instrument only the returned result; all original arithmetic is intact.
    code=inspect.getsource(original.fixed_rk4_evaluate)
    code=code.replace('    return results, timing','    results["_arrays"] = {"t": dense_time, "N": dense_states, "lam": dense_costates}\n    return results, timing')
    ns=dict(original.__dict__);exec(compile(code,'<S4-return-only-instrumentation>','exec'),ns)
    evaluate=ns['fixed_rk4_evaluate']
    a.out.mkdir(parents=True,exist_ok=False)
    policy,meta,cache=original.load_proposed_dense_policy(a.checkpoint,'neural_pmp_'+a.method,q,p,
        query_batch_size=16,torch_threads=a.threads,policy_cache=a.cache)
    if cache is not None:np.savez_compressed(a.out/'policy_cache.npz',**cache)
    # This vectorized adapter is an existing repository implementation of the
    # same state branch. Verify against the original scalar adapter first.
    class Batch:
        def __call__(self,t,N):return actions_batch(policy.policy,float(t),N)
        def query_grid(self,t,N):return np.stack([self(x,y) for x,y in zip(t,N)])
    adapter=Batch()
    samples=np.stack([np.ones(21)*10,np.linspace(8,12,21),np.linspace(9,11,21)])
    err=max(float(np.max(np.abs(adapter(t,samples)-policy(t,samples)))) for t in (0,.47,1.5,4.,8.,8.34,10.))
    if err>1e-12:raise RuntimeError('batch action mismatch '+str(err))
    if a.directions:
        directions=load_directions(a.directions,128,21);initial=10*(1+.2*directions);ids=list(range(128))
    else:initial=np.stack([np.ones(21)*10,np.linspace(8,12,21)]);ids=['NOM','S20']
    params=common_parameters(p);rows=[];started=time.perf_counter()
    for start in range(0,len(initial),a.batch_size):
        end=min(len(initial),start+a.batch_size)
        results,timing=evaluate(adapter,initial[start:end],q,mask,p,fixed_steps=a.steps)
        arrays=results.pop('_arrays');N=arrays['N'];lam=arrays['lam'];u=adapter.query_grid(q,N)
        psi,dot,ddot=strict_quantities_batch(N,lam,u,params)
        safe=np.maximum(N,1e-12);C=params['phi'][None,None,:];beta=params['beta'][None,None,:]
        rho=np.sum(params['M'][None,None,:]*lam*safe,axis=2)/(21+safe.sum(axis=2))
        B=-np.sum(C*C*beta*safe,axis=2)+rho*np.sum(C*C*safe,axis=2)-rho*np.sum(C*safe,axis=2)**2/(21+safe.sum(axis=2))
        for i,sid in enumerate(ids[start:end]):
            components=[psi[:,i],dot[:,i],ddot[:,i]]
            rr=[float(np.sqrt(np.mean(x[mask]**2))) for x in components]
            mm=[float(np.max(np.abs(x[mask]))) for x in components]
            eps=float(np.linalg.norm(rr))
            assert abs(eps-float(results['epsilon'][i]))<1e-9
            early,late=switching_records(q,u[:,i]);tin=early['profile_time'];tout=late['profile_time']
            interior=(q>tin)&(q<tout);near=(q>=tout-.25)&(q<=tout+.25)
            indices=np.flatnonzero(near);peak=int(indices[np.argmax(np.abs(ddot[near,i]))])
            ip=int(np.searchsorted(q,tin,side='right'));im=int(np.searchsorted(q,tout,side='left')-1)
            row={'method':a.method,'state_id':sid,'R0':rr[0],'R1':rr[1],'R2':rr[2],
                'M0':mm[0],'M1':mm[1],'M2':mm[2],'epsilon':eps,'max_B':float(B[mask,i].max()),
                'v0':'UNKNOWN: original action-bound tolerance not recorded',
                'v3':'UNKNOWN: original action-bound tolerance not recorded',
                'max_control_bound_violation':float(np.maximum.reduce([np.zeros(len(q)),-u[:,i],u[:,i]-3]).max()),
                'tin_marker':tin,'tout_marker':tout,'t_entry_sample':float(q[ip]),'entry_psi':float(psi[ip,i]),'entry_dot':float(dot[ip,i]),
                't_exit_sample':float(q[im]),'exit_psi':float(psi[im,i]),'exit_dot':float(dot[im,i]),
                'tpeak':float(q[peak]),'ddot_peak_signed':float(ddot[peak,i]),
                'interior_max_abs_ddot':float(np.max(np.abs(ddot[interior,i]))),
                'peak_side':'BEFORE' if q[peak]<tout else 'AFTER' if q[peak]>tout else 'AT',
                'interval_class':'U','detector_type':'control-change markers only; singular/control-interior interval detector UNVERIFIED',
                'dense_J_NOT_paper_J':float(results['J'][i]),'steps':a.steps}
            rows.append(row)
        np.savez_compressed(a.out/f'arrays_{start:03d}_{end:03d}.npz',time=q,state_ids=np.asarray(ids[start:end]),
            controls=u,psi=psi,dot_psi=dot,ddot_psi=ddot,B=B,initial_states=initial[start:end])
        print(f'{a.method}: {start}:{end}/{len(initial)}, {timing["elapsed_seconds"]:.2f}s',flush=True)
    with (a.out/'rows.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    payload={'rows':rows,'policy_metadata':meta,'batch_adapter_max_abs_error':err,'seconds':time.perf_counter()-started,
        'protocol':{'point_count':16120,'mask':'(T32 minus T1) intersect [1.5,8)',
        'solver':'original S4 fixed RK4, open-loop adjoint, Hermite midpoint states',
        'steps':a.steps,'query_rule':'original fixed 801-token support; PCHIP extension of T32 raw logits at internal ODE stages',
        'derivative_scale':'physical objective weights (1,40,8000); physical time',
        'detector':'build_final_der_phenotype_3d_figure.switching_records; largest adjacent action change in [0,2], [7,10]; not an interval detector',
        'boundary_tolerance':'UNKNOWN; do not substitute the 20261024 audit tolerance'},
        'source_sha256':hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
        'original_evaluator_sha256':hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest()}
    (a.out/'summary.json').write_text(json.dumps(payload,indent=2)+'\n')

if __name__=='__main__':main()
