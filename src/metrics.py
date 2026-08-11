from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import numpy as np
import torch

class Confusion:
    def __init__(self,n:int): self.n=n; self.mat=torch.zeros(n,n,dtype=torch.float64)
    def update(self,pred:torch.Tensor,target:torch.Tensor):
        pred=pred.detach().view(-1).cpu(); target=target.detach().view(-1).cpu(); valid=(target>=0)&(target<self.n)
        ids=self.n*target[valid]+pred[valid]; self.mat += torch.bincount(ids,minlength=self.n*self.n).reshape(self.n,self.n)
    def get(self):
        diag=self.mat.diag(); denom=self.mat.sum(1)+self.mat.sum(0)-diag
        iou=(diag/denom.clamp_min(1)).mean().item(); acc=(diag.sum()/self.mat.sum().clamp_min(1)).item()
        return iou*100.0,acc*100.0

class DenseMetricAccumulator:
    def __init__(self,dataset:str):
        self.dataset=dataset; self.conf=Confusion(7 if dataset=="cityscapes" else 13)
        self.depth_abs=0.; self.depth_rel=0.; self.depth_n=0
        self.normal_angles=[]
    @torch.no_grad()
    def update(self,pred,batch):
        self.conf.update(pred["semantic"].argmax(1),batch["semantic"])
        mask=(batch["depth"]!=0); n=int(mask.sum())
        if n:
            diff=(pred["depth"]-batch["depth"]).abs(); self.depth_abs += float(diff[mask].sum())
            self.depth_rel += float((diff[mask]/batch["depth"][mask].abs().clamp_min(1e-8)).sum()); self.depth_n+=n
        if self.dataset=="nyuv2":
            maskn=batch["normal"].norm(dim=1)>0
            if maskn.any():
                dot=(pred["normal"]*batch["normal"]).sum(1).clamp(-1,1)
                ang=torch.rad2deg(torch.acos(dot[maskn])).detach().cpu().numpy(); self.normal_angles.append(ang)
    def compute(self):
        miou,acc=self.conf.get(); rel=self.depth_rel/max(self.depth_n,1)
        out={"miou":miou,"pix_acc":acc,"abs_err":self.depth_abs/max(self.depth_n,1),"rel_err":rel}
        if self.dataset=="nyuv2":
            a=np.concatenate(self.normal_angles) if self.normal_angles else np.array([np.nan])
            out.update(normal_mean=float(np.nanmean(a)),normal_median=float(np.nanmedian(a)),normal_11_25=float(np.nanmean(a<11.25)*100),normal_22_5=float(np.nanmean(a<22.5)*100),normal_30=float(np.nanmean(a<30)*100))
        return out

METRIC_DIRECTIONS={"miou":1,"pix_acc":1,"abs_err":-1,"rel_err":-1,"normal_mean":-1,"normal_median":-1,"normal_11_25":1,"normal_22_5":1,"normal_30":1}
def delta_m(method:dict[str,float],stl:dict[str,float])->float:
    vals=[]
    for k,b in stl.items():
        if k not in METRIC_DIRECTIONS or k not in method or not np.isfinite(b) or abs(b)<1e-12: continue
        # paper convention: (-1)^l (M_m-M_b)/M_b, l=1 if higher is better.
        sign=-1 if METRIC_DIRECTIONS[k]>0 else 1
        vals.append(sign*(method[k]-b)/b*100.)
    return float(np.mean(vals)) if vals else float("nan")
def method_ranks(rows:list[dict])->dict[str,float]:
    if not rows: return {}
    names=[r["method"] for r in rows]; metrics=[k for k in METRIC_DIRECTIONS if any(k in r for r in rows)]
    ranks={n:[] for n in names}
    for k in metrics:
        vals=np.array([r.get(k,np.nan) for r in rows],float); order=vals if METRIC_DIRECTIONS[k]<0 else -vals
        idx=np.argsort(order); rr=np.empty(len(vals),float); rr[idx]=np.arange(1,len(vals)+1)
        for n,v in zip(names,rr): ranks[n].append(v)
    return {n:float(np.mean(v)) for n,v in ranks.items() if v}
