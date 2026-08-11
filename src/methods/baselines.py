from __future__ import annotations
import copy, math, random
from typing import Sequence
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize
from src.methods.common import named_trainable,task_gradients,blend_grads,inner_lists,simplex_projection,assign_grads,apply_direction


def _gram(grads):
    K=len(grads); G=torch.empty(K,K,device=grads[0][0].device,dtype=grads[0][0].dtype)
    for i in range(K):
        for j in range(i,K): G[i,j]=G[j,i]=inner_lists(grads[i],grads[j])
    return G

def min_norm_weights(grads,steps:int=250)->torch.Tensor:
    gram=_gram(grads); K=gram.shape[0]; w=torch.full((K,),1/K,device=gram.device,dtype=gram.dtype)
    # Convex PGD with safe spectral step; K is 2/3 in the dense benchmarks.
    L=float(torch.linalg.norm(gram.float(),ord=2).detach().cpu())+1e-12; lr=1.0/L
    for _ in range(steps):
        new=simplex_projection(w-lr*(gram@w))
        if torch.norm(new-w)<1e-8: w=new; break
        w=new
    return w

def cagrad_direction(grads,alpha:float=0.4):
    K=len(grads); gram=_gram(grads).detach().float().cpu().numpy(); x0=np.ones(K)/K; g0_norm=math.sqrt(max(float(gram.mean()),0)+1e-8); c=alpha*g0_norm+1e-8
    def obj(x): return float((x.reshape(1,K)@gram@x0.reshape(K,1)+c*np.sqrt(x.reshape(1,K)@gram@x.reshape(K,1)+1e-8)).item())
    res=minimize(obj,x0,bounds=[(0,1)]*K,constraints={"type":"eq","fun":lambda x:1-x.sum()},method="SLSQP",options={"maxiter":100,"ftol":1e-10})
    x=res.x if res.success else x0; ww=torch.tensor(x,device=grads[0][0].device,dtype=grads[0][0].dtype)
    avg=[sum(g[j] for g in grads)/K for j in range(len(grads[0]))]; gw=blend_grads(grads,ww)
    gw_norm=torch.sqrt(sum((g*g).sum() for g in gw)); lam=c/(float(gw_norm.detach().cpu())+1e-8)
    return [(a+lam*b)/(1+alpha**2) for a,b in zip(avg,gw)],ww

def nash_weights(grads)->torch.Tensor:
    gram=_gram(grads).detach().double().cpu().numpy(); K=gram.shape[0]
    scale=max(np.linalg.norm(gram),1e-12); A=gram/scale
    def obj(a): return 0.5*float(a@A@a)-float(np.log(a).sum())
    def jac(a): return A@a-1.0/a
    res=minimize(obj,np.ones(K),jac=jac,bounds=[(1e-6,1e6)]*K,method="L-BFGS-B",options={"maxiter":100,"ftol":1e-12})
    # The Nash bargaining first-order condition is A @ alpha = 1/alpha.
    # Solving 0.5*alpha^T A alpha - sum(log alpha) gives that condition directly;
    # unlike MGDA/LS, Nash weights are not normalized to sum to K.
    a=res.x if res.success and np.isfinite(res.x).all() else np.ones(K)
    return torch.tensor(a,device=grads[0][0].device,dtype=grads[0][0].dtype)

class BaselineController:
    """Runnable baselines used in the draft's comparison list.

    Gradient-manipulation formulas follow the public FAMO/LibMTL implementations; MoCo and MoDo
    are implemented from their published recurrences. `step` performs a single SGD-style model update.
    """
    def __init__(self,name:str,model:torch.nn.Module,tasks:tuple[str,...],cfg:dict,device:torch.device):
        self.name=name.lower(); self.model=model; self.tasks=tasks; self.cfg=cfg; self.device=device; self.params=named_trainable(model); self.K=len(tasks)
        self.shared_params=[(n,p) for n,p in self.params if not (n.startswith("pred_") or ".pred_" in n)]
        self.task_params=[(n,p) for n,p in self.params if (n.startswith("pred_") or ".pred_" in n)]
        self.lr=float(cfg.get("gamma",cfg.get("lr",5e-4))); self.steps=0
        self.w=torch.full((self.K,),1/self.K,device=device)
        self.epoch_loss_history=[]
        self.dwa_window=int(cfg.get("iteration_window",25)) if self.name=="dwa" else 25
        self.dwa_costs=np.ones((2*self.dwa_window,self.K),dtype=np.float32) if self.name=="dwa" else None
        self.dwa_running_iterations=0
        self.dwa_weights=np.ones(self.K,dtype=np.float32) if self.name=="dwa" else None
        self.logsigma=torch.zeros(self.K,device=device,requires_grad=True) if self.name=="uw" else None
        self.uw_opt=torch.optim.SGD([self.logsigma],lr=float(cfg.get("weight_lr",self.lr))) if self.logsigma is not None else None
        if self.name=="famo":
            self.famo_logits=torch.zeros(self.K,device=device,requires_grad=True); self.famo_min=torch.zeros(self.K,device=device)
            self.famo_opt=torch.optim.Adam([self.famo_logits],lr=float(cfg.get("weight_lr",0.025)),weight_decay=float(cfg.get("weight_decay",1e-5)))
            self.prev_losses=None
        else: self.famo_logits=None
        self.tracking=None
        self.warm_started=False
    @property
    def required_batches(self)->int: return 3 if self.name=="modo" else 1
    def begin_epoch(self)->None: pass
    def end_epoch(self,mean_losses:Sequence[float])->None: self.epoch_loss_history.append(list(map(float,mean_losses)))
    def _dwa_weighted_loss(self, losses:torch.Tensor):
        # FAMO public baseline: maintain 2*window recent per-iteration losses;
        # compare the newer and older halves once enough observations exist.
        assert self.dwa_costs is not None and self.dwa_weights is not None
        self.dwa_costs[:-1,:]=self.dwa_costs[1:,:]
        self.dwa_costs[-1,:]=losses.detach().cpu().numpy()
        if self.dwa_running_iterations>self.dwa_window:
            newer=self.dwa_costs[self.dwa_window:,:].mean(0)
            older=np.maximum(self.dwa_costs[:self.dwa_window,:].mean(0),1e-12)
            ratio=newer/older; temp=float(self.cfg.get("temperature",2.0))
            logits=ratio/temp; logits=logits-logits.max()
            e=np.exp(logits); self.dwa_weights=self.K*e/e.sum()
        weights=torch.from_numpy(self.dwa_weights.astype(np.float32)).to(device=self.device,dtype=losses.dtype)
        self.dwa_running_iterations+=1
        # The reference DWA weights sum to K and then averages over tasks.
        return (weights*losses).mean(),weights
    def _update_with_grads(self,grads,weights=None):
        if weights is None: weights=torch.full((self.K,),1/self.K,device=self.device)
        direction=blend_grads(grads,weights); apply_direction(self.params,[-g for g in direction],self.lr)
        return direction
    def step(self,loss_dict:dict[str,torch.Tensor],*, extra_loss_dicts:list[dict[str,torch.Tensor]]|None=None)->dict:
        losses=torch.stack([loss_dict[t] for t in self.tasks]); name=self.name; info={}
        if name in ("ls","si","rlw","dwa","uw","famo"):
            if name=="ls": weights=torch.ones(self.K,device=self.device); scalar=(losses*weights).sum()
            elif name=="si": weights=torch.ones(self.K,device=self.device); scalar=torch.log(losses.clamp_min(1e-8)).sum()
            elif name=="rlw": weights=torch.softmax(torch.randn(self.K,device=self.device),0); scalar=(losses*weights).sum()
            elif name=="dwa": scalar,weights=self._dwa_weighted_loss(losses)
            elif name=="uw":
                assert self.logsigma is not None; weights=torch.exp(-self.logsigma); scalar=sum(0.5*(torch.exp(-s)*l+s) for l,s in zip(losses,self.logsigma)); self.uw_opt.zero_grad(set_to_none=True)
            else:
                z=torch.softmax(self.famo_logits,-1); D=losses-self.famo_min+1e-8; c=(z/D).sum().detach(); scalar=(D.log()*z/c).sum(); weights=z; self.prev_losses=losses.detach()
            grads=torch.autograd.grad(scalar,[p for _,p in self.params]+(([self.logsigma] if name=="uw" else [])),retain_graph=True,allow_unused=True)
            model_grads=[torch.zeros_like(p) if g is None else g for g,(_,p) in zip(grads[:len(self.params)],self.params)]
            apply_direction(self.params,[-g for g in model_grads],self.lr)
            if name=="uw":
                self.logsigma.grad=grads[-1].detach().clone(); self.uw_opt.step()
            self.w=(weights.detach()/weights.detach().sum().clamp_min(1e-12))
            info["weights"]=self.w.cpu().tolist()
        else:
            if name=="mgda":
                shared_grads=task_gradients(list(losses),self.shared_params); weights=min_norm_weights(shared_grads)
                scalar=sum(losses[i]*(weights[i]*self.K) for i in range(self.K)); full=torch.autograd.grad(scalar,[p for _,p in self.params],retain_graph=True,allow_unused=True)
                apply_direction(self.params,[-(torch.zeros_like(p) if g is None else g) for g,(_,p) in zip(full,self.params)],self.lr); self.w=weights
            elif name=="pcgrad":
                grads=task_gradients(list(losses),self.shared_params); projected=[[g.clone() for g in task] for task in grads]
                order=list(range(self.K))
                for i in range(self.K):
                    random.shuffle(order)
                    for j in order:
                        if i==j: continue
                        dot=inner_lists(projected[i],grads[j]); norm=inner_lists(grads[j],grads[j]).clamp_min(1e-12)
                        if float(dot)<0:
                            projected[i]=[gi-(dot/norm)*gj for gi,gj in zip(projected[i],grads[j])]
                merged=[sum(task[p] for task in projected) for p in range(len(self.shared_params))]; apply_direction(self.shared_params,[-g for g in merged],self.lr)
                if self.task_params:
                    tg=torch.autograd.grad(losses.sum(),[p for _,p in self.task_params],retain_graph=True,allow_unused=True); apply_direction(self.task_params,[-(torch.zeros_like(p) if g is None else g) for g,(_,p) in zip(tg,self.task_params)],self.lr)
            elif name=="graddrop":
                grads=task_gradients(list(losses),self.shared_params); flats=torch.stack([torch.cat([g.reshape(-1) for g in task]) for task in grads],dim=1)
                P=0.5*(1+flats.sum(1)/(flats.abs().sum(1)+1e-8)); U=torch.rand_like(P); M=(P.gt(U)[:,None]&flats.gt(0)) | (P.lt(U)[:,None]&flats.lt(0)); merged=(flats*M.float()).mean(1)
                out=[]; pos=0
                for _,p in self.shared_params: n=p.numel(); out.append((merged[pos:pos+n]*self.K).view_as(p)); pos+=n
                apply_direction(self.shared_params,[-g for g in out],self.lr)
                if self.task_params:
                    tg=torch.autograd.grad(losses.sum(),[p for _,p in self.task_params],retain_graph=True,allow_unused=True); apply_direction(self.task_params,[-(torch.zeros_like(p) if g is None else g) for g,(_,p) in zip(tg,self.task_params)],self.lr)
            elif name=="cagrad":
                grads=task_gradients(list(losses),self.shared_params); direction,weights=cagrad_direction(grads,float(self.cfg.get("c",0.4))); apply_direction(self.shared_params,[-self.K*g for g in direction],self.lr); self.w=weights
                if self.task_params:
                    tg=torch.autograd.grad(losses.sum(),[p for _,p in self.task_params],retain_graph=True,allow_unused=True); apply_direction(self.task_params,[-(torch.zeros_like(p) if g is None else g) for g,(_,p) in zip(tg,self.task_params)],self.lr)
            elif name in ("nash-mtl","nash_mtl","nashmtl"):
                shared_grads=task_gradients(list(losses),self.shared_params); weights=nash_weights(shared_grads); scalar=sum(losses[i]*weights[i] for i in range(self.K)); full=torch.autograd.grad(scalar,[p for _,p in self.params],retain_graph=True,allow_unused=True); apply_direction(self.params,[-(torch.zeros_like(p) if g is None else g) for g,(_,p) in zip(full,self.params)],self.lr); self.w=weights/weights.sum()
            elif name=="mgda-warm":
                grads=task_gradients(list(losses),self.params); rho=float(self.cfg.get("rho",0.5)); beta=float(self.cfg.get("beta",0.5))
                if not self.warm_started:
                    gram=_gram(grads); beta0=float(self.cfg.get("warm_beta",beta));
                    for _ in range(int(self.cfg.get("warm_steps",40))): self.w=simplex_projection(self.w-beta0*(gram@self.w+rho*self.w))
                    self.warm_started=True
                gram=_gram(grads); self.w=simplex_projection(self.w-beta*(gram@self.w+rho*self.w)); self._update_with_grads(grads,self.w)
            elif name=="moco":
                # LibMTL's author-shared MoCo recurrence: normalize each stochastic
                # task gradient, track y_t with polynomially decaying beta_t, then
                # update lambda with decaying gamma_t and a softmax simplex map.
                grads=task_gradients(list(losses),self.params)
                beta=float(self.cfg.get("tracking_beta",0.5)); beta_sigma=float(self.cfg.get("tracking_beta_sigma",0.5))
                gamma=float(self.cfg.get("weight_lr",0.1)); gamma_sigma=float(self.cfg.get("weight_lr_sigma",0.5)); rho=float(self.cfg.get("rho",0.0))
                if self.tracking is None: self.tracking=[[torch.zeros_like(g) for g in task] for task in grads]
                elif self.tracking[0][0].device != grads[0][0].device: self.tracking=[[x.to(self.device) for x in task] for task in self.tracking]
                t=max(self.steps+1,1); bt=beta/(t**beta_sigma); gt=gamma/(t**gamma_sigma)
                with torch.no_grad():
                    normalized=[]
                    for i,task in enumerate(grads):
                        n=torch.sqrt(sum((g.float()*g.float()).sum() for g in task)).clamp_min(1e-8)
                        normalized.append([(g.detach()/n)*losses[i].detach() for g in task])
                    for i in range(self.K):
                        for j in range(len(self.params)): self.tracking[i][j].mul_(1-bt).add_(normalized[i][j],alpha=bt)
                gram=_gram(self.tracking); self.w=torch.softmax(self.w-gt*(gram@self.w+rho*self.w),dim=0)
                self._update_with_grads(self.tracking,self.w)
            elif name=="modo":
                grads=task_gradients(list(losses),self.params)
                if not extra_loss_dicts or len(extra_loss_dicts)!=2: raise ValueError("MoDo requires two extra independent mini-batches")
                g1=grads; g2=task_gradients([extra_loss_dicts[0][t] for t in self.tasks],self.params); g3=task_gradients([extra_loss_dicts[1][t] for t in self.tasks],self.params)
                gamma=float(self.cfg.get("weight_lr",0.001)); rho=float(self.cfg.get("rho",0.1))
                cross=torch.empty(self.K,self.K,device=self.device)
                for i in range(self.K):
                    for j in range(self.K): cross[i,j]=inner_lists(g1[i],g2[j])
                self.w=simplex_projection(self.w-gamma*(cross@self.w+rho*self.w)); self._update_with_grads(g3,self.w)
            else: raise ValueError(f"unknown runnable baseline {self.name}")
            info["weights"]=self.w.detach().cpu().tolist()
        self.steps+=1; return info
    def famo_post_step(self,new_loss_dict:dict[str,torch.Tensor])->None:
        if self.name!="famo" or self.prev_losses is None: return
        curr=torch.stack([new_loss_dict[t] for t in self.tasks]).detach(); delta=torch.log(self.prev_losses-self.famo_min+1e-8)-torch.log(curr-self.famo_min+1e-8)
        with torch.enable_grad(): d=torch.autograd.grad(torch.softmax(self.famo_logits,-1),self.famo_logits,grad_outputs=delta)[0]
        self.famo_opt.zero_grad(set_to_none=True); self.famo_logits.grad=d; self.famo_opt.step()
    def state_dict(self):
        tracking=None if self.tracking is None else [[x.detach().cpu() for x in task] for task in self.tracking]
        d={"w":self.w.detach().cpu(),"steps":self.steps,"history":self.epoch_loss_history,"warm_started":self.warm_started,"tracking":tracking}
        if self.name=="dwa":
            d["dwa_costs"]=self.dwa_costs.copy(); d["dwa_running_iterations"]=self.dwa_running_iterations; d["dwa_weights"]=self.dwa_weights.copy()
        if self.logsigma is not None: d["logsigma"]=self.logsigma.detach().cpu()
        if self.famo_logits is not None: d["famo_logits"]=self.famo_logits.detach().cpu(); d["famo_opt"]=self.famo_opt.state_dict()
        return d

    def load_state_dict(self,d):
        self.w=d.get("w",self.w).to(self.device); self.steps=int(d.get("steps",0)); self.epoch_loss_history=list(d.get("history",[])); self.warm_started=bool(d.get("warm_started",False)); self.tracking=d.get("tracking")
        if self.name=="dwa":
            if "dwa_costs" in d: self.dwa_costs=np.array(d["dwa_costs"],dtype=np.float32,copy=True)
            self.dwa_running_iterations=int(d.get("dwa_running_iterations",self.dwa_running_iterations))
            if "dwa_weights" in d: self.dwa_weights=np.array(d["dwa_weights"],dtype=np.float32,copy=True)
        if self.logsigma is not None and "logsigma" in d:
            with torch.no_grad(): self.logsigma.copy_(d["logsigma"].to(self.device))
        if self.famo_logits is not None and "famo_logits" in d:
            with torch.no_grad(): self.famo_logits.copy_(d["famo_logits"].to(self.device))
            if "famo_opt" in d:
                try: self.famo_opt.load_state_dict(d["famo_opt"])
                except Exception: pass
