from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Sequence
import torch

NamedParam=tuple[str,torch.nn.Parameter]
GradList=list[torch.Tensor]
TaskGrads=list[GradList]

def named_trainable(model:torch.nn.Module)->list[NamedParam]: return [(n,p) for n,p in model.named_parameters() if p.requires_grad]

def task_gradients(losses:Sequence[torch.Tensor], params:Sequence[NamedParam], *, create_graph:bool=False)->TaskGrads:
    raw_params=[p for _,p in params]; out=[]
    for i,loss in enumerate(losses):
        grads=torch.autograd.grad(loss,raw_params,retain_graph=True,create_graph=create_graph,allow_unused=True)
        out.append([torch.zeros_like(p) if g is None else g for g,p in zip(grads,raw_params)])
    return out

def blend_grads(task_grads:TaskGrads,w:torch.Tensor)->GradList:
    return [sum(w[i]*task_grads[i][j] for i in range(len(task_grads))) for j in range(len(task_grads[0]))]

def inner_lists(a:Sequence[torch.Tensor],b:Sequence[torch.Tensor])->torch.Tensor:
    return sum((x*y).sum() for x,y in zip(a,b))

def task_scores(direction:GradList,task_grads:TaskGrads)->torch.Tensor:
    return torch.stack([inner_lists(direction,g) for g in task_grads])

def assign_grads(params:Sequence[NamedParam], grads:Sequence[torch.Tensor])->None:
    for (_,p),g in zip(params,grads): p.grad=g.detach().clone()

def apply_direction(params:Sequence[NamedParam], direction:Sequence[torch.Tensor], gamma:float)->None:
    with torch.no_grad():
        for (_,p),d in zip(params,direction): p.add_(d,alpha=float(gamma))

def simplex_projection(y:torch.Tensor)->torch.Tensor:
    if y.ndim!=1: raise ValueError("simplex_projection expects a vector")
    if y.numel()==1: return torch.ones_like(y)
    u=torch.sort(y,descending=True).values; cssv=torch.cumsum(u,0)-1
    ind=torch.arange(1,y.numel()+1,device=y.device,dtype=y.dtype)
    cond=u-cssv/ind>0
    if not bool(cond.any()): return torch.full_like(y,1.0/y.numel())
    rho=torch.nonzero(cond,as_tuple=False)[-1,0]; theta=cssv[rho]/ind[rho]
    w=torch.clamp(y-theta,min=0); return w/w.sum().clamp_min(torch.finfo(w.dtype).eps)

class GradientMomentum:
    def __init__(self,alpha:float|None): self.alpha=None if alpha is None else float(alpha); self.state:TaskGrads|None=None
    def update(self,grads:TaskGrads)->TaskGrads:
        if self.alpha is None: return [[g.detach() for g in task] for task in grads]
        a=self.alpha
        if not (0<a<=1): raise ValueError("gradient momentum alpha must be in (0,1]")
        if self.state is None: self.state=[[torch.zeros_like(g) for g in task] for task in grads]
        elif self.state and self.state[0] and self.state[0][0].device != grads[0][0].device:
            self.state=[[x.to(grads[i][j].device) for j,x in enumerate(task)] for i,task in enumerate(self.state)]
        with torch.no_grad():
            for i in range(len(grads)):
                for j in range(len(grads[i])): self.state[i][j].mul_(1-a).add_(grads[i][j].detach(),alpha=a)
        return [[g.clone() for g in task] for task in self.state]
    def state_dict(self):
        state=None if self.state is None else [[x.detach().cpu() for x in task] for task in self.state]
        return {"alpha":self.alpha,"state":state}
    def load_state_dict(self,d): self.alpha=d.get("alpha"); self.state=d.get("state")
