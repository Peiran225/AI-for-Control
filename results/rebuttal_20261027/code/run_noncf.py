"""One recorded-budget Direct-only initialization and DER chain; no searches.

Budget/optimizer arguments are transcribed from the locked metadata and the
archived table1 replication launcher. Fresh seed-4 parameters replace the
CF-trained ancestor, the sole necessary initialization change.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source-root',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--gpu',default='4')
    a=ap.parse_args(); root=a.source_root.resolve(); out=a.out.resolve()
    out.mkdir(parents=True,exist_ok=False); (out/'logs').mkdir()
    os.environ['CUDA_VISIBLE_DEVICES']=a.gpu
    os.environ['OMP_NUM_THREADS']='4'; os.environ['OPENBLAS_NUM_THREADS']='1'
    sys.path[:0]=[str(root),str(root/'scripts')]
    import numpy as np
    import torch
    from train_paper_pmp_kkt import ProblemConfig,set_seed
    from scripts.refine_time_only_singular_plateau import build_model
    from supervised_initialization import supervised_stage
    from supervised_initialization import AffineBoundaryProjectedControl
    torch.set_num_threads(4)
    if hasattr(torch.backends,'mha'):torch.backends.mha.set_fastpath_enabled(False)
    set_seed(4)
    cfg=ProblemConfig(T=10,n=800,m=21,umax=3,beta=.1,alpha=.0025,gamma=20,n0=10,m_suppression=.5)
    architecture={'model':'transformer','d_model':64,'heads':4,'layers':2,'init_u':1.5,'seed':4}
    base=build_model(architecture,cfg).double()
    model=AffineBoundaryProjectedControl(base,umax=3,scale=1.02,offset=0).double().cuda()
    model.train()
    direct=root/'outputs/new_objective_alpha1_grid_20260721_v2/a1_b40_g8000/direct_nominal/n800_strict_final/scale_1_direct_solution.npz'
    teacher=root/'inputs/direct_feedback_init_20260727/direct__seed_selected__resistant_heavy.npz'
    opts=argparse.Namespace(supervised_epochs=1500,supervised_lr=5e-5,
        supervised_switch_point_weight=30.,supervised_switch_difference_weight=50.,
        supervised_difference_loss_weight=10.,supervised_global_point_weight=100000.,
        grad_clip=5.,print_every=100,fixed_validation_every=0)
    record={'source_root':str(root),'stage_seeds':{'temporal_initialization':4,'temporal_LM':2607,'feedback_initialization':31},
        'fresh_initialization':True,'CF_loss_used':False,'CF_ancestor_loaded':False,
        'direct_target':str(direct),'feedback_teacher':str(teacher),
        'budgets':{'temporal_AdamW':1500,'temporal_LM':[20,12,6],'feedback_AdamW':[1200]*9,'feedback_LBFGS_outer':[2,8,10,6]},
        'training_states':['NOM','structured_r010'],'S20_training_used':False,'stages':[]}
    def save(): (out/'run.json').write_text(json.dumps(record,indent=2)+'\n')
    raw0={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    torch.save({'model_state':raw0,'args':architecture,'base_model_args':architecture,'problem':cfg.__dict__,
        'initialization':'fresh random parameters; seed 4; no checkpoint loaded'},out/'fresh_initialization.pt')
    t=torch.linspace(0,1,801,device='cuda',dtype=torch.float64)
    target=torch.as_tensor(np.load(direct)['u'],device='cuda',dtype=torch.float64)
    if target.numel()==800: target=torch.cat([target,target[-1:]])
    started=time.perf_counter()
    selected,hist=supervised_stage(model,t,(cfg.T*t).detach().cpu().numpy(),target,cfg,opts)
    model.load_state_dict(selected)
    payload={'model_state':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
        'args':vars(opts),'base_model_args':architecture,'problem':cfg.__dict__,
        'wrapper':{'class':'AffineBoundaryProjectedControl','scale':1.02,'offset':0.},
        'source_checkpoint':str(out/'fresh_initialization.pt'),'method':'Direct-only supervised initialization',
        'direct_or_manual_solution_used':True,'teacher_free':False,'CF_loss_used':False}
    checkpoint=out/'temporal_supervised.pt';torch.save(payload,checkpoint)
    with (out/'temporal_supervised_history.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(hist[0]));w.writeheader();w.writerows(hist)
    record['stages'].append({'stage':'temporal_supervised','seconds':time.perf_counter()-started,'checkpoint':str(checkpoint),'status':'completed'})
    save();del model,base;torch.cuda.empty_cache()
    def run(name,script,args,expected):
        cmd=[sys.executable,'-u',str(root/'scripts'/script),*map(str,args)]
        entry={'stage':name,'command':cmd,'status':'running'};record['stages'].append(entry);save()
        st=time.perf_counter()
        with (out/'logs'/f'{name}.log').open('w') as log:
            status=subprocess.run(cmd,cwd=root,env=os.environ,stdout=log,stderr=subprocess.STDOUT).returncode
        entry.update(seconds=time.perf_counter()-st,returncode=status,checkpoint=str(expected))
        entry['status']='completed' if status==0 and expected.exists() else 'FAILED'
        save()
        if entry['status']=='FAILED':raise RuntimeError('FAILED '+name)
        print(json.dumps(entry),flush=True)
        return expected
    for name,steps,mult,weights,damp,cap,batch in [
        ('temporal_lm_m2',20,2,(.01,.03,1),.001,2,16),
        ('temporal_lm_m8',12,8,(.01,.03,1),.001,2,16),
        ('temporal_lm_equal',6,8,(1,1,1),1e-5,1,32)]:
        dest=out/name
        checkpoint=run(name,'refine_direct_initialized_output_head_lm.py',[
            '--start-checkpoint',checkpoint,'--out-dir',dest,'--device','cuda','--seed',2607,
            '--iterations',steps,'--trajectory-multiplier',mult,'--query-batch-size',batch,
            '--trainable-scope','output-head','--interior-start',1.5,'--interior-end',8.,
            '--boundary-margin',.05,'--w0',weights[0],'--w1',weights[1],'--w2',weights[2],
            '--boundary-weight',1,'--initial-damping',damp,'--maximum-step-norm',cap,
            '--jacobian-mode','forward','--linear-solver','explicit'],dest/'selected_checkpoint.pt')
    dest=out/'feedback_supervised'
    checkpoint=run('feedback_supervised','initialize_feedback_from_direct_trajectories.py',[
        '--time-checkpoint',checkpoint,'--teacher',teacher,'--out-dir',dest,'--device','cuda','--float64',
        '--seed',31,'--epochs-per-round',1200,'--dagger-rounds',9,'--batch-size',1600,
        '--eval-every',25,'--state-feature-mode','burden_composition','--interior-weight',50,
        '--reset-optimizer-each-round','--nominal-reference-method','dop853','--nominal-reference-multiplier',16,
        '--nominal-reference-query-batch-size',16],dest/'feedback_direct_initialization.pt')
    for name,steps,mult,inner,w2 in [('der_m2_initial',2,2,2,1),('der_m2_continuation',8,2,3,1),('der_m4',10,4,3,4),('der_m8',6,8,3,4)]:
        dest=out/name
        args=['--checkpoint',checkpoint,'--out-dir',dest,'--option','der','--trajectory-mode','continuous-policy',
            '--continuous-collocation','nodes','--device','cuda','--threads',4,'--query-batch-size',128,
            '--interval-start',1.5,'--interval-end',8.,'--no-refresh-nominal-reference','--exclude-nominal-from-training',
            '--train-random-states',0,'--validation-random-states',0,'--train-scope','all','--lr',.5,
            '--selection-metric','scalar','--residual-p',2,'--report-scale-factor',400,
            '--multiplier',mult,'--outer-steps',steps,'--inner-iterations',inner,'--max-eval',2*inner,
            '--w0',1,'--w1',1,'--w2',w2]
        if name!='der_m2_initial':args+=['--reference-multiplier',16,'--reference-method','dop853']
        checkpoint=run(name,'refine_feedback_offgrid_scalar.py',args,dest/'best_feedback_section5.pt')
    record['status']='completed';save()

if __name__=='__main__':main()
