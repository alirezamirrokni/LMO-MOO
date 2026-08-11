from __future__ import annotations
import copy, time
from pathlib import Path
from typing import Any
import numpy as np
import torch
from tqdm.auto import tqdm
from src.data.dense import build_loaders,TASKS
from src.models.segnet import MultiTaskSegNet
from src.losses import task_losses
from src.metrics import DenseMetricAccumulator,delta_m
from src.methods.alg1 import EntropicLMOMGDA
from src.methods.alg4 import QuadraticLMOMGDA
from src.methods.baselines import BaselineController
from src.util import seed_everything,synchronize
from src.config import resolve_method_config

PROPOSED_KINDS={"alg1":"alg1","algorithm1":"alg1","alg4":"alg4","algorithm4":"alg4"}

def resolve_device(spec:str|None)->torch.device:
    if spec and spec!="auto":
        d=torch.device(spec)
        if d.type=="cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
        return d
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def move_batch(batch:dict[str,torch.Tensor],device:torch.device): return {k:v.to(device,non_blocking=True) for k,v in batch.items()}

def build_model(config:dict,dataset:str,width:float|None=None)->MultiTaskSegNet:
    width=float(config.get("model",{}).get("width",1.0) if width is None else width)
    return MultiTaskSegNet(dataset,width=width)

def build_controller(method_name:str,model,tasks,cfg,device):
    kind=str(cfg.get("kind",method_name)).lower()
    if kind in ("alg1","algorithm1"): return EntropicLMOMGDA(model,tasks,cfg,device)
    if kind in ("alg4","algorithm4"): return QuadraticLMOMGDA(model,tasks,cfg,device)
    return BaselineController(kind,model,tasks,cfg,device)

@torch.no_grad()
def evaluate(model,loader,dataset,device):
    model.eval(); acc=DenseMetricAccumulator(dataset)
    for batch in loader:
        batch=move_batch(batch,device); pred=model(batch["image"]); acc.update(pred,batch)
    return acc.compute()

def _next_batch(it,loader):
    try: return next(it),it
    except StopIteration:
        it=iter(loader); return next(it),it

def train_method(config:dict[str,Any],dataset:str,method_name:str,seed:int,*,width:float|None=None,
                 resume_state:dict[str,Any]|None=None,checkpoint_callback=None,max_epochs:int|None=None,
                 progress:bool=True)->dict[str,Any]:
    seed_everything(seed,bool(config.get("experiment",{}).get("deterministic",True)))
    device=resolve_device(config.get("experiment",{}).get("device","auto")); train_loader,val_loader=build_loaders(config,dataset,seed)
    model=build_model(config,dataset,width).to(device); tasks=TASKS[dataset]; mcfg=resolve_method_config(config,method_name,dataset); controller=build_controller(method_name,model,tasks,mcfg,device)
    start_epoch=0; history=[]; weight_history=[]; timings=[]
    if resume_state:
        model.load_state_dict(resume_state["model"]); start_epoch=int(resume_state.get("epoch",0)); history=list(resume_state.get("history",[])); weight_history=list(resume_state.get("weight_history",[])); timings=list(resume_state.get("timings",[]))
        if "controller" in resume_state: controller.load_state_dict(resume_state["controller"])
    epochs=int(max_epochs if max_epochs is not None else config.get("experiment",{}).get("epochs",200)); checkpoint_interval=int(config.get("experiment",{}).get("checkpoint_interval",1))
    iterator=range(start_epoch,epochs); iterator=tqdm(iterator,desc=f"{dataset}:{method_name}:seed{seed}",unit="epoch",dynamic_ncols=True) if progress else iterator
    for epoch in iterator:
        model.train();
        if hasattr(controller,"begin_epoch"): controller.begin_epoch()
        sums=np.zeros(len(tasks),float); n_batches=0
        epoch_start=time.perf_counter(); backward_time=0.; post_time=0.; extra_it=iter(train_loader); components={k:0.0 for k in ("gradient_seconds","blend_seconds","lmo_seconds","weight_seconds","parameter_seconds")}
        for batch in train_loader:
            batch=move_batch(batch,device); model.zero_grad(set_to_none=True)
            t0=time.perf_counter(); pred=model(batch["image"]); losses=task_losses(pred,batch,tasks)
            extra_losses=[]
            if getattr(controller,"required_batches",1)==3:
                for _ in range(2):
                    xb,extra_it=_next_batch(extra_it,train_loader); xb=move_batch(xb,device); xp=model(xb["image"]); extra_losses.append(task_losses(xp,xb,tasks))
            synchronize(device); t1=time.perf_counter()
            if hasattr(controller,"required_batches"): info=controller.step(losses,extra_loss_dicts=extra_losses or None)
            else: info=controller.step(losses)
            synchronize(device); t2=time.perf_counter(); backward_time += t2-t1
            for k,v in info.get("timing",{}).items(): components[k]=components.get(k,0.0)+float(v)
            if getattr(controller,"name","")=="famo":
                with torch.no_grad(): npred=model(batch["image"]); nloss=task_losses(npred,batch,tasks)
                controller.famo_post_step(nloss)
            synchronize(device); post_time += time.perf_counter()-t2
            vals=[float(losses[t].detach()) for t in tasks]; sums+=np.array(vals); n_batches+=1
            if "weights" in info: weight_history.append({"epoch":epoch,"step":len(weight_history),"weights":info["weights"]})
        epoch_minutes=(time.perf_counter()-epoch_start)/60.0; metrics=evaluate(model,val_loader,dataset,device); mean_losses=(sums/max(n_batches,1)).tolist()
        if hasattr(controller,"end_epoch"): controller.end_epoch(mean_losses)
        row={"epoch":epoch+1,"train_losses":dict(zip(tasks,mean_losses)),**metrics}; history.append(row)
        timings.append({"epoch":epoch+1,"minutes":epoch_minutes,"method_update_seconds":backward_time,"post_seconds":post_time,**components})
        if progress and hasattr(iterator,"set_postfix_str"): iterator.set_postfix_str(" ".join(f"{k}={v:.4g}" for k,v in list(metrics.items())[:3]))
        if checkpoint_callback and ((epoch+1)%checkpoint_interval==0 or epoch+1==epochs):
            checkpoint_callback({"epoch":epoch+1,"model":{k:v.detach().cpu() for k,v in model.state_dict().items()},"controller":controller.state_dict(),"history":history,"weight_history":weight_history,"timings":timings})
    last_n=int(config.get("experiment",{}).get("average_last_epochs",10)); tail=history[-min(last_n,len(history)):]
    metric_keys=[k for k in tail[-1].keys() if k not in ("epoch","train_losses")] if tail else []
    final={k:float(np.mean([r[k] for r in tail])) for k in metric_keys}
    return {"complete":True,"epoch":epochs,"model":{k:v.detach().cpu() for k,v in model.state_dict().items()},"controller":controller.state_dict(),"history":history,"weight_history":weight_history,"timings":timings,"final":final,"dataset":dataset,"method":method_name,"seed":seed,"width":float(width if width is not None else config.get("model",{}).get("width",1.0))}
