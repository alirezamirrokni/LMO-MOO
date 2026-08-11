from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.metrics import METRIC_DIRECTIONS,method_ranks

def ensure_dirs(root:Path):
    for sub in ("tables","figures","logs","configs","cache"): (root/sub).mkdir(parents=True,exist_ok=True)

def write_json(value:Any,path:Path):
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(value,indent=2,default=float),encoding="utf-8")

def aggregate_results(results:list[dict],reported:dict[str,dict]|None=None)->pd.DataFrame:
    rows=[]
    groups={}
    for r in results: groups.setdefault((r["dataset"],r["method"]),[]).append(r["final"])
    for (dataset,method),vals in groups.items():
        row={"dataset":dataset,"method":method,"source":"run","seeds":len(vals)}
        keys=sorted(set().union(*[v.keys() for v in vals]))
        for k in keys:
            arr=np.array([v.get(k,np.nan) for v in vals],float); row[k]=float(np.nanmean(arr)); row[k+"_std"]=float(np.nanstd(arr,ddof=1)) if np.sum(np.isfinite(arr))>1 else 0.0
        rows.append(row)
    if reported:
        existing={(r["dataset"],r["method"]) for r in rows}
        for dataset,methods in reported.items():
            for method,metrics in methods.items():
                if (dataset,method) not in existing: rows.append({"dataset":dataset,"method":method,"source":"Zhang et al. (reported)","seeds":0,**metrics})
    if not rows:
        return pd.DataFrame(columns=["dataset","method","source","seeds"])
    return pd.DataFrame(rows)

def add_ranks(frame:pd.DataFrame)->pd.DataFrame:
    out=frame.copy()
    # Zhang's published MR was computed on the paper's original comparison set.
    # Once new LMO methods are added, MR must be recomputed over the expanded
    # table. Preserve the paper value separately for traceability.
    if "mr" in out.columns:
        out["mr_reported"]=out["mr"]
    else:
        out["mr_reported"]=np.nan
    out["mr"]=np.nan
    for dataset,g in out.groupby("dataset"):
        rows=[{"method":r["method"],**{k:r[k] for k in METRIC_DIRECTIONS if k in r and pd.notna(r[k])}} for _,r in g.iterrows()]
        ranks=method_ranks(rows)
        for idx,r in g.iterrows():
            out.at[idx,"mr"]=ranks.get(r["method"],np.nan)
    return out

def save_table(frame:pd.DataFrame,path:Path):
    path.parent.mkdir(parents=True,exist_ok=True); frame.to_csv(path.with_suffix(".csv"),index=False)
    try: path.with_suffix(".md").write_text(frame.to_markdown(index=False),encoding="utf-8")
    except ImportError: pass

def plot_metric(frame:pd.DataFrame,dataset:str,metric:str,path:Path):
    g=frame[(frame.dataset==dataset)&frame[metric].notna()].copy()
    if g.empty:return
    g=g.sort_values(metric,ascending=METRIC_DIRECTIONS.get(metric,-1)<0); fig,ax=plt.subplots(figsize=(max(8,0.65*len(g)),5.2)); ax.bar(np.arange(len(g)),g[metric].to_numpy()); ax.set_xticks(np.arange(len(g)),g.method,rotation=50,ha="right"); ax.set_ylabel(metric); ax.grid(axis="y",alpha=.25); fig.tight_layout(); path.parent.mkdir(parents=True,exist_ok=True); fig.savefig(path,dpi=220,bbox_inches="tight"); plt.close(fig)
