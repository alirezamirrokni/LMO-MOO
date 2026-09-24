"""Aggregate completed 3-seed runs and generate CSV + the paper's LaTeX table."""
import argparse
import csv
import json
from pathlib import Path
import statistics

BASELINES=[('MGDA','mgda',(95.58,94.10,94.84)),('MGDA + Muon','mgda_muon',(94.61,93.65,94.13)),
           ('FAMO','famo',(95.89,94.88,95.39)),('FAMO + Muon','famo_muon',(95.61,94.33,94.97)),
           ('Muon (LS)','muon_ls',(95.39,94.82,95.11)),('MOON','moon',(95.99,95.31,95.65))]
GROUPS=[('Oracle','oracle',[('l2',r'$\ell_2$'),('sign','sign'),('spectral','spectral')]),
        ('Weights','weights',[('entropic','entropic'),('projected','projected')]),
        ('Momentum','momentum',[('blended','blended'),('per-task','per-task'),('none','none')]),
        ('Step $\\eta$','eta',[(str(x),f'{x:g}') for x in [1e-5,5e-5,1e-4,5e-4,1e-3]])]


def aggregate(root,tag,method,seeds):
    values=[]
    for seed in seeds:
        p=root/tag/method/f'seed{seed}'/'summary.json'
        if not p.exists(): return None
        x=json.loads(p.read_text())
        if x.get('smoke') or not x.get('completed'): return None
        if x.get('selection', 'test') != 'test':
            raise ValueError(f'Validation results cannot be reported as test results: {p}')
        if x['seed']!=seed or x['method']!=method or x['tag']!=tag: raise ValueError(f'Identity mismatch: {p}')
        values.append(x)
    sigs=[{k:v for k,v in x['signature'].items() if k not in {'seed'}} for x in values]
    if any(s!=sigs[0] for s in sigs): raise ValueError(f'Mixed configurations/data for {tag}/{method}')
    out={'tag':tag,'method':method,'n_seeds':len(values),'data_sha256':sigs[0]['data_sha256'],'metric':values[0]['metric']}
    for key in ['left','right','avg','gap']:
        vals=[x[key] for x in values]
        out[key]=None if any(v is None for v in vals) else statistics.mean(vals)
        out[key+'_std']=None if any(v is None for v in vals) else statistics.stdev(vals) if len(vals)>1 else 0.0
    return out


def fmt(x,key): return '--' if x is None or x.get(key) is None else f'{x[key]:.2f}' if key!='gap' else f'{x[key]:.3g}'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('results/multimnist'))
    p.add_argument('--seeds',nargs='+',type=int,default=[42,43,44])
    p.add_argument('--baseline-source',choices=['reported','reproduced'],default='reported')
    p.add_argument('--out',type=Path,default=Path('results/multimnist/report'))
    a=p.parse_args()
    if len(set(a.seeds))!=len(a.seeds): p.error('Seeds must be unique')
    entries=[]; main={}
    for _,method,_ in BASELINES+[('Entropic LMO-MGDA','ours',None)]:
        main[method]=aggregate(a.root,'main',method,a.seeds)
        if main[method]: entries.append(main[method])
    ablations={}
    for _,name,variants in GROUPS:
        for value,_ in variants:
            tag=f'{name}-{value}'; ablations[tag]=aggregate(a.root,tag,'ours',a.seeds)
            if ablations[tag]: entries.append(ablations[tag])
    # One comparison must use the same underlying generated data and metric convention.
    if len({(x['data_sha256'],x['metric']) for x in entries})>1:
        raise ValueError('Different datasets or metric conventions in one table')
    a.out.mkdir(parents=True,exist_ok=True)
    with (a.out/'results.csv').open('w',newline='') as f:
        fields=['tag','method','n_seeds','left','left_std','right','right_std','avg','avg_std','gap','gap_std','metric','data_sha256']
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(entries)
    lines=[r'\begin{table}[t]\centering\scriptsize\setlength{\tabcolsep}{3.5pt}',
           r'\begin{minipage}[t]{0.46\textwidth}\centering',r'\begin{tabular}{lccc}',
           r'\toprule',r'Method & Left$\uparrow$ & Right$\uparrow$ & Avg.$\uparrow$\\',r'\midrule']
    for label,method,reported in BASELINES:
        cells=[f'{x:.2f}' for x in reported] if a.baseline_source=='reported' else [fmt(main[method],k) for k in ['left','right','avg']]
        lines.append(label+' & '+' & '.join(cells)+r'\\')
    lines += [r'\midrule',r'\textbf{Entropic LMO-MGDA} & '+' & '.join(fmt(main['ours'],k) for k in ['left','right','avg'])+r'\\',
              r'\bottomrule',r'\end{tabular}',r'\end{minipage}\hfill',r'\begin{minipage}[t]{0.52\textwidth}\centering',
              r'\begin{tabular}{llcc}',r'\toprule',r'Ablation & Variant & Avg.$\uparrow$ & Gap$\downarrow$\\',r'\midrule']
    # Expand the slash-separated variants into rows, so each measured value is unambiguous.
    for label,name,variants in GROUPS:
        for i,(value,display) in enumerate(variants):
            x=ablations[f'{name}-{value}']
            lines.append((label if i==0 else '')+' & '+display+' & '+fmt(x,'avg')+' & '+fmt(x,'gap')+r'\\')
    caption=f'MultiMNIST with the released MOON ViT backbone, test accuracy (\\%), {len(a.seeds)} seeds. '
    caption+=r'Baseline rows are reported by \citet{moon2026}; ours and ablations are local measurements. ' if a.baseline_source=='reported' else 'All rows are local measurements. '
    caption+='Final-epoch metrics; missing or incomplete runs are shown as --. Gap is the simplex Frank--Wolfe gap on a fixed training probe, using the exact geometry oracle.'
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{minipage}',r'\caption{'+caption+'}',r'\label{tab:mnist}',r'\end{table}']
    (a.out/'table.tex').write_text('\n'.join(lines)+'\n')
    print(f'Wrote {a.out}/results.csv and table.tex; complete measured configurations: {len(entries)}')

if __name__=='__main__': main()
