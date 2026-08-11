from __future__ import annotations
import time
import torch
from src.methods.common import named_trainable,task_gradients,blend_grads,task_scores,apply_direction,simplex_projection,GradientMomentum
from src.methods.lmo import build_lmo_direction

def quadratic_weight_gradient(w:torch.Tensor,scores:torch.Tensor,dual_norm:torch.Tensor|float,radius:float,rho:float)->torch.Tensor:
    """Draft Algorithm 4 gradient: -||G||_* g/R + rho (w - 1/K)."""
    R=float(radius)
    if R<=0: raise ValueError("LMO radius must be positive")
    uniform=torch.full_like(w,1/w.numel())
    return -(torch.as_tensor(dual_norm,device=w.device,dtype=w.dtype)*scores/R)+float(rho)*(w-uniform)

def quadratic_weight_update(w:torch.Tensor,scores:torch.Tensor,dual_norm:torch.Tensor|float,radius:float,beta:float,rho:float)->tuple[torch.Tensor,torch.Tensor]:
    grad=quadratic_weight_gradient(w,scores,dual_norm,radius,rho)
    return simplex_projection(w-float(beta)*grad),grad

class QuadraticLMOMGDA:
    """Algorithm 4 from the project draft: LMO model step + PGD task-weight step with centered quadratic regularization."""
    def __init__(self,model:torch.nn.Module,tasks:tuple[str,...],config:dict,device:torch.device):
        self.model=model; self.tasks=tasks; self.cfg=config; self.device=device; self.params=named_trainable(model)
        self.w=torch.full((len(tasks),),1/len(tasks),device=device); self.momentum=GradientMomentum(config.get("momentum_alpha")); self.steps=0
    def step(self,loss_dict:dict[str,torch.Tensor])->dict:
        def sync():
            if self.device.type=="cuda": torch.cuda.synchronize(self.device)
        losses=[loss_dict[t] for t in self.tasks]; sync(); t0=time.perf_counter(); grads=task_gradients(losses,self.params); sync(); t1=time.perf_counter(); used=self.momentum.update(grads)
        G=blend_grads(used,self.w); sync(); t2=time.perf_counter()
        lmo=build_lmo_direction(G,self.params,geometry=self.cfg.get("lmo","l2"),radius=self.cfg.get("radius",1.0),
            spectral_backend=self.cfg.get("spectral_backend","newton_schulz"),ns_steps=self.cfg.get("ns_steps",5),
            width_scaling=self.cfg.get("width_scaling","none"),output_parameter_names=getattr(self.model,"output_parameter_names",set())); sync(); t3=time.perf_counter()
        scores=task_scores(lmo.direction,used); R=float(self.cfg.get("radius",1.0)); rho=float(self.cfg.get("rho",0.1)); beta=float(self.cfg.get("beta",0.5))
        # Draft Alg.4: dJ/dw_i = -||G||_* <D,grad_i>/R + rho (w_i - 1/K).
        self.w,weight_grad=quadratic_weight_update(self.w,scores,lmo.normalized_dual_proxy,R,beta,rho); sync(); t4=time.perf_counter()
        apply_direction(self.params,lmo.direction,float(self.cfg.get("gamma",5e-4))); sync(); t5=time.perf_counter(); self.steps+=1
        return {"timing":{"gradient_seconds":t1-t0,"blend_seconds":t2-t1,"lmo_seconds":t3-t2,"weight_seconds":t4-t3,"parameter_seconds":t5-t4},"weights":self.w.detach().cpu().tolist(),"scores":scores.detach().cpu().tolist(),"weight_grad":weight_grad.detach().cpu().tolist(),"lmo_support":float(lmo.support.detach()),"dual_proxy":float(lmo.normalized_dual_proxy.detach())}
    def state_dict(self): return {"w":self.w.detach().cpu(),"momentum":self.momentum.state_dict(),"steps":self.steps}
    def load_state_dict(self,d): self.w=d["w"].to(self.device); self.momentum.load_state_dict(d.get("momentum",{})); self.steps=int(d.get("steps",0))
