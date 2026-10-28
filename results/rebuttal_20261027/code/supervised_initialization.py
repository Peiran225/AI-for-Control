"""Portable arithmetic of the original supervised_stage; no legacy imports."""
import math,time
import numpy as np
import torch
from torch import nn

class AffineBoundaryProjectedControl(nn.Module):
    def __init__(self,base,*,umax,scale,offset):
        super().__init__();self.base=base;self.umax=float(umax)
        self.register_buffer('scale',torch.tensor(float(scale),dtype=torch.float64))
        self.register_buffer('offset',torch.tensor(float(offset),dtype=torch.float64))
    def forward(self,t):return torch.clamp(self.scale*self.base(t)-self.offset,0.0,self.umax)

def clone_state(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}

def supervised_stage(model,normalized_t,physical_t,target,cfg,args):
    point_mask=(((physical_t>=.25)&(physical_t<=.70))|((physical_t>=8.60)&(physical_t<=9.40)))
    point_weights=torch.ones_like(target);point_weights[torch.as_tensor(point_mask,device=target.device)]=args.supervised_switch_point_weight
    difference_time=physical_t[1:]
    difference_mask=(((difference_time>=.25)&(difference_time<=.70))|((difference_time>=8.60)&(difference_time<=9.40)))
    difference_weights=torch.ones(cfg.n,dtype=target.dtype,device=target.device)
    difference_weights[torch.as_tensor(difference_mask,device=target.device)]=args.supervised_switch_difference_weight
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.supervised_lr,weight_decay=0.0)
    best_loss=math.inf;best_state=clone_state(model);history=[];start=time.perf_counter()
    for epoch in range(1,args.supervised_epochs+1):
        optimizer.zero_grad(set_to_none=True);prediction=model(normalized_t)
        point_error=(prediction-target)/cfg.umax
        point_loss=(point_weights*point_error.square()).sum()/point_weights.sum()
        difference_error=(torch.diff(prediction)-torch.diff(target))/cfg.umax
        difference_loss=(difference_weights*difference_error.square()).sum()/difference_weights.sum()
        global_point_loss=point_error.square().mean()
        loss=point_loss+args.supervised_difference_loss_weight*difference_loss+args.supervised_global_point_weight*global_point_loss
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),args.grad_clip);optimizer.step()
        row={'stage':'supervised','epoch':float(epoch),'loss':float(loss.detach().cpu()),'point_loss':float(point_loss.detach().cpu()),
             'global_point_loss':float(global_point_loss.detach().cpu()),'difference_loss':float(difference_loss.detach().cpu()),'elapsed_seconds':float(time.perf_counter()-start)}
        history.append(row)
        if row['loss']<best_loss:best_loss=row['loss'];best_state=clone_state(model)
        if epoch==1 or epoch%args.print_every==0 or epoch==args.supervised_epochs:
            with torch.no_grad():rmse=float((prediction-target).square().mean().sqrt().cpu())
            print(f'[supervised {epoch:04d}] loss={row["loss"]:.6g} control_rmse={rmse:.6g}',flush=True)
    return best_state,history
