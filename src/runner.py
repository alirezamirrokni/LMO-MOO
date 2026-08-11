from __future__ import annotations
import copy, math, shutil, sys, time
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from src.cache import CacheManager
from src.config import load_config,save_config,resolve_method_config
from src.experiments.common import train_method,build_model,move_batch,resolve_device
from src.data.dense import build_loaders,TASKS
from src.losses import task_losses
from src.methods.common import named_trainable,task_gradients,inner_lists,apply_direction
from src.report import ensure_dirs,aggregate_results,add_ranks,save_table,plot_metric,write_json
from src.reported import ZHANG_REPORTED
from src.metrics import delta_m
from src.util import seed_everything

PROBLEMS=("main_comparison","wall_clock","effect_k","lmo_geometry","local_smoothness","hyperparameter_transfer")

def _selected_methods(config,override):
    if override: return override
    return list(config.get("experiment",{}).get("methods",config.get("methods",{}).keys()))

def _selected_datasets(config,override=None):
    if override: return [override]
    return list(config.get("experiment",{}).get("datasets",["cityscapes","nyuv2"]))

def _seeds(config):
    e=config.get("experiment",{}); seeds=e.get("seeds")
    if seeds is not None:return [int(s) for s in seeds]
    n=int(e.get("num_seeds",3)); base=int(e.get("seed",2025)); return [base+i for i in range(n)]

def _run_cached(cache:CacheManager,config,dataset,method,seed,*,plot_only=False,width=None,max_epochs=None,progress=True):
    from src.methods.alg1 import EntropicLMOMGDA
    from src.methods.alg4 import QuadraticLMOMGDA
    from src.methods.baselines import BaselineController
    from src.methods.lmo import build_lmo_direction
    from src.methods.common import simplex_projection,task_gradients
    from src.models.segnet import MultiTaskSegNet
    from src.losses import task_losses
    from src.metrics import DenseMetricAccumulator
    from src.data.dense import build_loaders
    kind=str(resolve_method_config(config,method,dataset).get("kind",method)).lower()
    common_sources=(train_method,MultiTaskSegNet,task_losses,DenseMetricAccumulator,build_loaders,task_gradients)
    if kind in ("alg1","algorithm1"):
        sources=common_sources+(EntropicLMOMGDA,build_lmo_direction,simplex_projection)
    elif kind in ("alg4","algorithm4"):
        sources=common_sources+(QuadraticLMOMGDA,build_lmo_direction,simplex_projection)
    else:
        sources=common_sources+(BaselineController,simplex_projection)
    fp=cache.fingerprint(method,dataset,seed,sources)
    payload=cache.load(method,dataset,seed,fp)
    if payload and payload.get("complete"):
        return payload["state"]
    if plot_only: return None
    resume=payload.get("state") if payload else None
    def checkpoint(state): cache.save(method,dataset,seed,fp,state,complete=False)
    state=train_method(config,dataset,method,seed,width=width,resume_state=resume,checkpoint_callback=checkpoint,max_epochs=max_epochs,progress=progress)
    cache.save(method,dataset,seed,fp,state,complete=True); return state

def _attach_delta(frame:pd.DataFrame)->pd.DataFrame:
    out=frame.copy()
    if "delta_m" not in out.columns: out["delta_m"]=np.nan
    for idx,row in out.iterrows():
        if pd.notna(row.get("delta_m",np.nan)): continue
        stl=ZHANG_REPORTED.get(row["dataset"],{}).get("STL")
        if stl:
            metrics={k:row[k] for k in stl if k in row and pd.notna(row[k])}
            out.at[idx,"delta_m"]=delta_m(metrics,stl)
    return out

def _run_grid(config,output,problem,methods,datasets,*,reset_cache=False,plot_only=False,width=None,max_epochs=None):
    cache=CacheManager(output,problem,config)
    if reset_cache: cache.clear_all_methods()
    results=[]
    tasks=[(d,m,s) for d in datasets for m in methods for s in _seeds(config)]
    for d,m,s in tasks:
        if m not in config.get("methods",{}):
            print(f"[skip] method {m!r} has no config entry",file=sys.stderr); continue
        r=_run_cached(cache,config,d,m,s,plot_only=plot_only,width=width,max_epochs=max_epochs)
        if r is not None: results.append(r)
    return results

def run_main_comparison(config,output,methods,reset_cache,plot_only,dataset_override):
    datasets=_selected_datasets(config,dataset_override); results=_run_grid(config,output,"main_comparison",methods,datasets,reset_cache=reset_cache,plot_only=plot_only)
    reported={d:ZHANG_REPORTED[d] for d in datasets if d in ZHANG_REPORTED} if bool(config.get("experiment",{}).get("include_reported_baselines",True)) else None
    frame=aggregate_results(results,reported); frame=_attach_delta(frame); frame=add_ranks(frame); save_table(frame,output/"tables"/"main_comparison")
    for d in datasets:
        for metric in ("delta_m","miou","abs_err"): 
            if metric in frame: plot_metric(frame,d,metric,output/"figures"/f"{d}_{metric}.png")
    return frame

def run_wall_clock(config,output,methods,reset_cache,plot_only,dataset_override):
    datasets=_selected_datasets(config,dataset_override); results=_run_grid(config,output,"wall_clock",methods,datasets,reset_cache=reset_cache,plot_only=plot_only,max_epochs=int(config.get("experiment",{}).get("timing_epochs",10)))
    rows=[]; skip=int(config.get("experiment",{}).get("warmup_epochs",1))
    for r in results:
        vals=[x["minutes"] for x in r.get("timings",[])[skip:]]
        update=[x["method_update_seconds"] for x in r.get("timings",[])[skip:]]
        rec={"dataset":r["dataset"],"method":r["method"],"seed":r["seed"],"avg_minutes_per_epoch":float(np.mean(vals)) if vals else np.nan,"avg_method_update_seconds":float(np.mean(update)) if update else np.nan}
        for key in ("gradient_seconds","lmo_seconds","weight_seconds","parameter_seconds"):
            arr=[x.get(key,np.nan) for x in r.get("timings",[])[skip:]]; rec["avg_"+key]=float(np.nanmean(arr)) if arr else np.nan
        rows.append(rec)
    raw=pd.DataFrame(rows); save_table(raw,output/"tables"/"wall_clock_per_seed")
    if not raw.empty:
        agg=raw.groupby(["dataset","method"],as_index=False).agg(avg_minutes_per_epoch=("avg_minutes_per_epoch","mean"),std_minutes=("avg_minutes_per_epoch","std"),avg_method_update_seconds=("avg_method_update_seconds","mean"),avg_gradient_seconds=("avg_gradient_seconds","mean"),avg_lmo_seconds=("avg_lmo_seconds","mean"),avg_weight_seconds=("avg_weight_seconds","mean"),avg_parameter_seconds=("avg_parameter_seconds","mean")); save_table(agg,output/"tables"/"wall_clock")
    else: agg=raw
    return agg

def run_effect_k(config,output,methods,reset_cache,plot_only,dataset_override):
    datasets=_selected_datasets(config,dataset_override); results=_run_grid(config,output,"effect_k",methods,datasets,reset_cache=reset_cache,plot_only=plot_only)
    frame=_attach_delta(aggregate_results(results,None)); frame["K"]=frame["dataset"].map({"cityscapes":2,"nyuv2":3}); save_table(frame,output/"tables"/"effect_k")
    if not frame.empty:
        figdata=frame[["K","method","delta_m"]].dropna();
        if not figdata.empty:
            import matplotlib.pyplot as plt
            fig,ax=plt.subplots(figsize=(7.5,5.2))
            for m,g in figdata.groupby("method"): ax.plot(g.K,g.delta_m,marker="o",label=m)
            ax.set_xlabel("number of tasks K"); ax.set_ylabel("Δm% (lower is better)"); ax.set_xticks(sorted(figdata.K.unique())); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(output/"figures"/"effect_k.png",dpi=220); plt.close(fig)
    return frame

def run_lmo_geometry(config,output,methods,reset_cache,plot_only,dataset_override):
    datasets=_selected_datasets(config,dataset_override); results=_run_grid(config,output,"lmo_geometry",methods,datasets,reset_cache=reset_cache,plot_only=plot_only)
    frame=_attach_delta(aggregate_results(results,None)); frame["algorithm"]=frame.method.str.extract(r"^(Alg[14])",expand=False); frame["lmo"]=frame.method.str.replace(r"^Alg[14]-","",regex=True); save_table(frame,output/"tables"/"lmo_geometry")
    for d in datasets:
        if "delta_m" in frame: plot_metric(frame,d,"delta_m",output/"figures"/f"{d}_lmo_geometry.png")
    return frame

def _flat_norm(grads): return torch.sqrt(sum((g.float()**2).sum() for g in grads))

def _single_task_smoothness(config,dataset,task,seed,output,max_points):
    seed_everything(seed); device=resolve_device(config.get("experiment",{}).get("device","auto")); train_loader,_=build_loaders(config,dataset,seed); model=build_model(config,dataset).to(device); params=named_trainable(model); lr=float(config.get("smoothness",{}).get("lr",5e-4)); sample_every=int(config.get("smoothness",{}).get("sample_every",10)); rows=[]; step=0
    for epoch in range(int(config.get("smoothness",{}).get("epochs",1))):
        for batch in train_loader:
            batch=move_batch(batch,device); model.train(); pred=model(batch["image"]); loss=task_losses(pred,batch,TASKS[dataset])[task]; g=task_gradients([loss],params)[0]; gn=float(_flat_norm(g).detach().cpu()); before=[p.detach().clone() for _,p in params]
            apply_direction(params,[-x for x in g],lr)
            if step%sample_every==0:
                pred2=model(batch["image"]); loss2=task_losses(pred2,batch,TASKS[dataset])[task]; g2=task_gradients([loss2],params)[0]; num=float(_flat_norm([b-a for a,b in zip(g,g2)]).detach().cpu()); den=math.sqrt(sum(float(((p.detach()-b).float()**2).sum().cpu()) for (_,p),b in zip(params,before)))
                rows.append({"dataset":dataset,"task":task,"seed":seed,"step":step,"gradient_norm":gn,"local_smoothness":num/max(den,1e-12)})
                if len(rows)>=max_points:return rows
            step+=1
    return rows

def run_local_smoothness(config,output,methods,reset_cache,plot_only,dataset_override):
    # This experiment is intentionally single-task SGD, matching Zhang's generalized-smoothness illustration.
    import matplotlib.pyplot as plt
    rows=[]; max_points=int(config.get("smoothness",{}).get("max_points_per_task",500)); table_path=output/"tables"/"local_smoothness.csv"
    if plot_only:
        if table_path.exists():
            rows=pd.read_csv(table_path).to_dict("records")
        else:
            print(f"[plot-only] no saved local-smoothness table at {table_path}; nothing to plot",file=sys.stderr)
    else:
        for d in _selected_datasets(config,dataset_override):
            for task in TASKS[d]: rows.extend(_single_task_smoothness(config,d,task,_seeds(config)[0],output,max_points))
    frame=pd.DataFrame(rows,columns=None if rows else ["dataset","task","seed","step","gradient_norm","local_smoothness"]); save_table(frame,output/"tables"/"local_smoothness")
    for (d,t),g in frame.groupby(["dataset","task"]):
        fig,ax=plt.subplots(figsize=(6.3,5.2)); ax.scatter(g.gradient_norm,g.local_smoothness,s=13,alpha=.55); ax.set_xlabel("gradient norm"); ax.set_ylabel("local smoothness estimate"); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(output/"figures"/f"local_smoothness_{d}_{t}.png",dpi=220); plt.close(fig)
    return frame

def run_hyperparameter_transfer(config,output,methods,reset_cache,plot_only,dataset_override):
    dataset=(dataset_override or config.get("experiment",{}).get("datasets",["nyuv2"])[0]); base_method=methods[0]; gammas=[float(x) for x in config.get("transfer",{}).get("gammas",[1e-4,2e-4,5e-4,1e-3])]; widths=[float(x) for x in config.get("transfer",{}).get("widths",[0.5,1.0,2.0])]; seeds=_seeds(config); rows=[]
    # Proxy-width sweep.
    for gamma in gammas:
        cfg=copy.deepcopy(config); alias=f"{base_method}-g{gamma:g}-w{widths[0]:g}"; cfg["methods"][alias]=resolve_method_config(cfg,base_method,dataset); cfg["methods"][alias]["gamma"]=gamma
        cache=CacheManager(output/"transfer_cache","hyperparameter_transfer",cfg)
        vals=[]
        for s in seeds:
            r=_run_cached(cache,cfg,dataset,alias,s,plot_only=plot_only,width=widths[0]);
            if r: vals.append(delta_m(r["final"],ZHANG_REPORTED[dataset]["STL"]))
        if vals: rows.append({"phase":"proxy_search","dataset":dataset,"width":widths[0],"gamma":gamma,"delta_m":float(np.mean(vals)),"delta_m_std":float(np.std(vals))})
    if not rows: frame=pd.DataFrame(); save_table(frame,output/"tables"/"hyperparameter_transfer"); return frame
    best=min((r for r in rows if r["phase"]=="proxy_search"),key=lambda r:r["delta_m"])["gamma"]
    for width in widths:
        cfg=copy.deepcopy(config); alias=f"{base_method}-g{best:g}-w{width:g}"; cfg["methods"][alias]=resolve_method_config(cfg,base_method,dataset); cfg["methods"][alias]["gamma"]=best
        cache=CacheManager(output/"transfer_cache","hyperparameter_transfer",cfg); vals=[]
        for s in seeds:
            r=_run_cached(cache,cfg,dataset,alias,s,plot_only=plot_only,width=width)
            if r: vals.append(delta_m(r["final"],ZHANG_REPORTED[dataset]["STL"]))
        if vals: rows.append({"phase":"transfer","dataset":dataset,"width":width,"gamma":best,"delta_m":float(np.mean(vals)),"delta_m_std":float(np.std(vals))})
    frame=pd.DataFrame(rows); save_table(frame,output/"tables"/"hyperparameter_transfer"); write_json({"selected_gamma":best,"proxy_width":widths[0]},output/"tables"/"selected_gamma.json")
    return frame

def run_experiment(problem:str,config_path:Path,output_override:Path|None=None,*,reset_cache=False,plot_only=False,methods_override=None,dataset_override=None):
    if problem not in PROBLEMS: raise ValueError(f"unknown problem {problem}; expected one of {PROBLEMS}")
    config=load_config(config_path); output=Path(output_override or config.get("output_dir",f"D:/LMO-MOO-runs/{problem}")); ensure_dirs(output); save_config(config,output/"configs"/config_path.name)
    methods=_selected_methods(config,methods_override)
    dispatch={"main_comparison":run_main_comparison,"wall_clock":run_wall_clock,"effect_k":run_effect_k,"lmo_geometry":run_lmo_geometry,"local_smoothness":run_local_smoothness,"hyperparameter_transfer":run_hyperparameter_transfer}
    frame=dispatch[problem](config,output,methods,reset_cache,plot_only,dataset_override)
    print(f"\nDone: {problem}\nOutputs: {output}")
    if isinstance(frame,pd.DataFrame) and not frame.empty: print(frame.to_string(index=False,max_rows=40))
    return frame
