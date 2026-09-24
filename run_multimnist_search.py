"""Staged Ours-only search: training-split validation, then three-seed confirmation.

Never reads test data or baseline scores. Existing reference-schedule validation
checkpoints can be reused. No tuning checkpoint is used for final test training.
"""
import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
LR_GRID = [0.001, 0.003, 0.006, 0.01, 0.015, 0.022, 0.03, 0.045]
ALPHA_GRID = [0.02, 0.05, 0.1, 0.2, 0.5, 0.8]
ETA_GRID = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 3e-3, 1e-2]


def candidate(lr, alpha, eta=1e-4, lr_schedule='reference', min_lr_ratio=0.05):
    return dict(lr=float(lr), alpha=float(alpha), eta=float(eta), lr_schedule=lr_schedule,
                min_lr_ratio=float(min_lr_ratio), oracle='spectral', weights='entropic', momentum='blended')


def tag_for(c):
    tag = f'lr-{c["lr"]:.12g}-alpha-{c["alpha"]:.12g}'
    if c['lr_schedule'] != 'reference' or c['eta'] != 1e-4:
        tag += f'-eta-{c["eta"]:.12g}-{c["lr_schedule"]}'
        if c['lr_schedule'] == 'cosine':
            tag += f'-floor-{c["min_lr_ratio"]:.12g}'
    return tag


def unique(candidates):
    return list({tag_for(c): c for c in candidates}.values())


def ranking(rows):
    return sorted(rows, key=lambda r: (-r['score'], r.get('score_std', 0), r['tag']))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-path', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--incumbent-selection', type=Path, required=True)
    p.add_argument('--reuse-root', type=Path)
    p.add_argument('--through', choices=['schedule', 'eta', 'confirm'], default='confirm')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--threads', type=int, default=4)
    a = p.parse_args()
    from experiments.multimnist.trainer import data_fingerprint, parser, run_signature, write_json
    previous = json.loads(a.incumbent_selection.read_text())
    if previous.get('test_used') is not False or previous['settings']['selection'] != 'validation':
        raise ValueError('Incumbent must come from a training-only validation search')
    selected = previous['selected']
    if selected['epochs'] != 100 or selected['batch_size'] != 256:
        raise ValueError('Keep the existing 100-epoch, batch-256 comparison budget')
    if any(selected[k] != v for k, v in [('oracle','spectral'),('weights','entropic'),('momentum','blended')]):
        raise ValueError('This search preserves the final Entropic LMO-MGDA method')
    incumbent = candidate(**{k: selected[k] for k in ['lr','alpha','eta']},
                          lr_schedule=selected.get('lr_schedule','reference'),
                          min_lr_ratio=selected.get('min_lr_ratio',0.05))
    train_hash = data_fingerprint(a.data_path, splits=('train',))
    if previous['settings']['train_sha256'] != train_hash:
        raise ValueError('Incumbent used different training images')
    split_seed = previous['settings'].get('split_seed', 2026)
    val_fraction = previous['settings'].get('val_fraction', 0.1)
    # Freeze source code and the full search before starting. No adaptive test-score input.
    code_hash = hashlib.sha256()
    for file in [Path(__file__), *sorted((ROOT/'experiments/multimnist').glob('*.py')),
                 *sorted((ROOT/'methods').glob('*.py'))]:
        code_hash.update(file.relative_to(ROOT).as_posix().encode()); code_hash.update(file.read_bytes())
    protocol = dict(version=1, selection='validation', metric='sample', train_sha256=train_hash,
                    split_seed=split_seed, val_fraction=val_fraction, epochs=100, batch_size=256,
                    lr_grid=LR_GRID, alpha_grid=ALPHA_GRID, eta_grid=ETA_GRID,
                    schedules=['reference','cosine'], min_lr_ratio=0.05,
                    screening_seed=42, confirmation_seeds=[42,43,44],
                    schedule_stages=[[25,96],[60,24],[100,8]], eta_families=4,
                    eta_screen_epochs=60, eta_finalists=8, confirmation_finalists=4,
                    protected_incumbent=incumbent, score_window=5,
                    source_sha256=code_hash.hexdigest())
    a.output_root.mkdir(parents=True, exist_ok=True)
    manifest = a.output_root/'search.json'
    if manifest.exists() and json.loads(manifest.read_text()) != protocol:
        raise ValueError('Search protocol/data/code changed. Use a new search directory.')
    write_json(manifest, protocol)
    print('Training-only validation search; no baseline or test-score inputs.', flush=True)
    env = dict(os.environ, PYTHONUNBUFFERED='1')

    def args_for(c, seed, stop):
        flags = [part for k, v in c.items() for part in ('--'+k.replace('_','-'),str(v))]
        return ['--method','ours','--selection','validation','--metric','sample',
                '--seed',str(seed),'--tag',tag_for(c),'--data-path',str(a.data_path.resolve()),
                '--output-root',str(a.output_root.resolve()),'--epochs','100','--batch-size','256',
                '--split-seed',str(split_seed),'--val-fraction',str(val_fraction),
                '--stop-after-epoch',str(stop),'--device',a.device,'--threads',str(a.threads),
                '--workers',str(a.workers),'--cache-data','--resume',*flags]

    def run(c, seed, stop):
        folder = a.output_root/tag_for(c)/'ours'/f'seed{seed}'
        args = args_for(c, seed, stop)
        expected = run_signature(parser().parse_args(args), train_hash)
        legacy = None if not a.reuse_root else a.reuse_root/tag_for(c)/'ours'/f'seed{seed}'
        if not folder.exists() and legacy and (legacy/'config.json').exists():
            old = json.loads((legacy/'config.json').read_text())
            if old.get('signature') == expected and (legacy/'checkpoint.pt').exists():
                staging = folder.with_name(folder.name+'.importing')
                if staging.exists(): shutil.rmtree(staging)
                shutil.copytree(legacy, staging)
                staging.replace(folder)
                print(f'Reused matching validation checkpoint: {legacy}', flush=True)
        for name in ['config.json','summary.json']:
            if (folder/name).exists() and json.loads((folder/name).read_text()).get('signature') != expected:
                raise ValueError(f'Validation checkpoint mismatch: {folder/name}')
        rows = json.loads((folder/'history.json').read_text()) if (folder/'history.json').exists() else []
        if not any(r['epoch'] == stop for r in rows):
            command = [sys.executable,'-u',str(ROOT/'run_multimnist.py'),*args]
            print(f'\nTraining {tag_for(c)}, seed={seed}, through epoch {stop}', flush=True)
            subprocess.run(command, cwd=ROOT, env=env, check=True)
            rows = json.loads((folder/'history.json').read_text())
        window = [next((r for r in rows if r['epoch']==ep), None) for ep in range(stop-4,stop+1)]
        if any(r is None for r in window): raise RuntimeError(f'Missing validation epochs: {folder}')
        if not all(math.isfinite(r['avg']) for r in window): raise RuntimeError(f'Nonfinite validation score: {folder}')
        score = statistics.mean(r['avg'] for r in window)
        return dict(tag=tag_for(c), config=c, seed=seed, epoch=stop, score=score,
                    final_avg=window[-1]['avg'], left=window[-1]['left'], right=window[-1]['right'])

    def evaluate(candidates, epoch, label):
        records = []
        for i, c in enumerate(unique(candidates),1):
            print(f'\n{label} {i}/{len(unique(candidates))}: {tag_for(c)}', flush=True)
            records.append(run(c,42,epoch))
        records = ranking(records)
        write_json(a.output_root/(label+'.json'), records)
        print(f'{label}: best validation score={records[0]["score"]:.4f}', flush=True)
        return records

    def shortlist(rows, count):
        # Keep the old configuration through every stage as an explicit control.
        return unique([r['config'] for r in ranking(rows)[:count]] + [incumbent])

    initial = unique([candidate(lr,alpha,eta=incumbent['eta'],lr_schedule=schedule)
                      for lr,alpha,schedule in itertools.product(LR_GRID,ALPHA_GRID,['reference','cosine'])] + [incumbent])
    screen = evaluate(initial,25,'schedule_screen')
    middle = evaluate(shortlist(screen,24),60,'schedule_middle')
    full = evaluate(shortlist(middle,8),100,'schedule_final')
    if a.through == 'schedule': return
    families = [r['config'] for r in ranking(full)[:4]]
    eta_candidates = unique([dict(c,eta=eta) for c in families for eta in ETA_GRID] + [incumbent])
    eta_screen = evaluate(eta_candidates,60,'eta_screen')
    eta_full = evaluate(shortlist(eta_screen,8),100,'eta_final')
    if a.through == 'eta': return
    pool = {r['tag']:r for r in full+eta_full}
    confirmation_candidates = shortlist(list(pool.values()),4)
    confirmation = []
    for c in confirmation_candidates:
        rows = [run(c,seed,100) for seed in protocol['confirmation_seeds']]
        confirmation.append(dict(tag=tag_for(c),config=c,score=statistics.mean(r['score'] for r in rows),
                                 score_std=statistics.stdev(r['score'] for r in rows),
                                 final_avg=statistics.mean(r['final_avg'] for r in rows),seeds=rows))
        write_json(a.output_root/'confirmation_partial.json',confirmation)
    confirmation = ranking(confirmation)
    best = confirmation[0]
    old = next(r for r in confirmation if r['tag']==tag_for(incumbent))
    result = dict(settings=protocol,selected=dict(best['config'],epochs=100,batch_size=256),
                  validation=best,incumbent_validation=old,confirmation=confirmation,
                  improvement_over_incumbent=best['score']-old['score'],test_used=False,
                  criterion='Mean over seeds 42/43/44 of sample-weighted validation accuracy averaged over epochs 96-100; lower seed std then tag break ties.')
    write_json(a.output_root/'selection.json',result)
    print('\nSelected on validation only: '+json.dumps(result['selected']),flush=True)
    print(f'Validation mean: {best["score"]:.4f}; incumbent: {old["score"]:.4f}.',flush=True)
    print('Fresh full-training refits are required. No claim of test superiority is made.',flush=True)


if __name__=='__main__': main()
