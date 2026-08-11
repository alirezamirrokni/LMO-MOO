from __future__ import annotations
import torch
import torch.nn.functional as F

def semantic_loss(pred:torch.Tensor,target:torch.Tensor)->torch.Tensor:
    return F.nll_loss(pred,target.long(),ignore_index=-1)
def depth_loss(pred:torch.Tensor,target:torch.Tensor)->torch.Tensor:
    mask=(target!=0).float(); denom=mask.sum().clamp_min(1.0)
    return ((pred-target).abs()*mask).sum()/denom
def normal_loss(pred:torch.Tensor,target:torch.Tensor)->torch.Tensor:
    mask=(target.norm(dim=1,keepdim=True)>0).float(); denom=mask.sum().clamp_min(1.0)
    return 1.0-((pred*target).sum(dim=1,keepdim=True)*mask).sum()/denom
def task_losses(pred:dict[str,torch.Tensor],batch:dict[str,torch.Tensor],tasks:tuple[str,...])->dict[str,torch.Tensor]:
    out={}
    if "semantic" in tasks: out["semantic"]=semantic_loss(pred["semantic"],batch["semantic"])
    if "depth" in tasks: out["depth"]=depth_loss(pred["depth"],batch["depth"])
    if "normal" in tasks: out["normal"]=normal_loss(pred["normal"],batch["normal"])
    return out
