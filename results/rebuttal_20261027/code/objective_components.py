"""Decompose the original n=800, four-substep RK4 task objective."""
import argparse,csv,json,sys
from pathlib import Path
import numpy as np

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();sys.path[:0]=[str(a.root),str(a.root/'scripts')]
    from evaluate_table1_heldout_common import Problem,load_control,load_policy,dynamics,rollout_controls
    import torch
    torch.set_num_threads(1)
    p=Problem();v=p.vectors();rows=[]
    time=a.root/'output/final_paper_recompute_20260727_clean_v2/tables/time_only_solution_n800.npz'
    direct_nom=a.root/'outputs/new_objective_alpha1_grid_20260721_v2/a1_b40_g8000/direct_nominal/n800_strict_final/scale_1_direct_solution.npz'
    direct_s20=a.root/'output/r020_state_specific_direct_20260729/final/solution.npz'
    for method in ('Direct','Time','CF','DER'):
        for sid,initial in [('NOM',np.full(21,10.)),('S20',np.linspace(8,12,21))]:
            if method=='Direct':source=direct_nom if sid=='NOM' else direct_s20
            elif method=='Time':source=time
            else:source=a.root/f'tmp/table1_checkpoints_20260728/{method.lower()}_final.pt'
            u=load_control(source,p) if method in ('Direct','Time') else None
            query=None;close=lambda:None
            if u is None:query,close,_=load_policy('pmp_kkt_'+method.lower(),source,hjb_tau=10.)
            state=initial[None,:].copy();cn=0.;cu=0.;dt=p.T/p.intervals;h=dt/4
            for k in range(800):
                action=np.asarray([u[k] if u is not None else query(k,k*dt,state[0].copy())]);action=np.clip(action,0,3)
                for _ in range(4):
                    k1=dynamics(state,action,p,v);s2=state+h*k1/2;k2=dynamics(s2,action,p,v)
                    s3=state+h*k2/2;k3=dynamics(s3,action,p,v);s4=state+h*k3;k4=dynamics(s4,action,p,v)
                    cn+=float((h/6)*((state@v['beta'])+2*(s2@v['beta'])+2*(s3@v['beta'])+(s4@v['beta']))[0])
                    cu+=float(h*p.gamma*action[0]);state+=h/6*(k1+2*k2+2*k3+k4)
            ct=float((state@v['alpha'])[0])
            _,J=rollout_controls(initial[None,:],p,v,fixed_control=u,query=query,objective_substeps=4)
            close();gap=abs(ct+cn+cu-float(J[0]))
            if gap>2e-7:raise RuntimeError('objective decomposition mismatch')
            rows.append({'method':method,'state':sid,'C_T':ct,'C_N':cn,'C_u':cu,'J':float(J[0]),'reconstruction_error':gap,'source':str(source)})
    a.out.mkdir(parents=True,exist_ok=True)
    with (a.out/'objective_components.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (a.out/'objective_components.json').write_text(json.dumps({'rows':rows,'protocol':'original float64 n=800 left-endpoint control execution, four RK4 substeps per interval; matching RK4 quadrature'},indent=2)+'\n')
    print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
