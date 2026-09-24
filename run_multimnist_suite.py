"""Sequential launcher: defaults to ONLY ours, three seeds; resume-safe."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
CONFIG=json.loads((ROOT/'experiments/multimnist/configs.json').read_text())


def jobs(suite,methods):
    result=[]
    if suite in {'main','all'}:
        result.extend(('main',method,[]) for method in methods)
    if suite in {'ablations','all'}:
        for name,flag,values in [('oracle','--oracle',['l2','sign','spectral']),
                                 ('weights','--weights',['entropic','projected']),
                                 ('momentum','--momentum',['blended','per-task','none']),
                                 ('eta','--eta',CONFIG['eta_grid'])]:
            for value in values:
                result.append((f'{name}-{value}', 'ours', [flag,str(value)]))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',choices=['main','ablations','all'],default='main')
    p.add_argument('--methods',type=lambda x: {'entropic-lmo-mgda':'ours','entropic_lmo_mgda':'ours'}.get(x,x),nargs='+',default=['ours'],choices=['ours','mgda','mgda_muon','famo','famo_muon','muon_ls','moon'])
    p.add_argument('--seeds',nargs='+',type=int,default=CONFIG['seeds'])
    p.add_argument('--data-path',type=Path,default=ROOT/'datasets/multimnist')
    p.add_argument('--output-root',type=Path,default=ROOT/'results/multimnist')
    p.add_argument('--device',default=None)
    p.add_argument('--dry-run',action='store_true')
    a,extra=p.parse_known_args()
    # Prevent accidental redefinition of identities controlled by the suite.
    if any(x.split('=')[0] in {'--tag','--method','--seed','--resume'} for x in extra):
        p.error('tag, method, seed, and resume are controlled by the suite')
    for tag,method,flags in jobs(a.suite,a.methods):
        for seed in a.seeds:
            cmd=[sys.executable,str(ROOT/'run_multimnist.py'),'--method',method,'--seed',str(seed),
                 '--tag',tag,'--data-path',str(a.data_path.resolve()),'--output-root',str(a.output_root.resolve()),'--resume']
            if a.device: cmd+=['--device',a.device]
            cmd+=extra+flags
            print(' '.join(cmd),flush=True)
            if not a.dry_run: subprocess.run(cmd,check=True,cwd=ROOT)

if __name__=='__main__': main()
