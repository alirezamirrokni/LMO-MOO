"""Resume-safe model-LR/momentum selection using only a held-out part of TRAIN data.

Screen a declared grid at epoch 40, continue the top two to epoch 100, and
select by final sample-weighted validation accuracy. All trials and settings
are retained. Final test experiments must start fresh on the full training set.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from experiments.multimnist.trainer import data_fingerprint, write_json

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / 'experiments/multimnist/configs.json').read_text())


def tag_for(lr, alpha):
    return 'lr-' + format(lr, '.12g') + '-alpha-' + format(alpha, '.12g')


def main():
    defaults = CONFIG['tuning']
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-path', type=Path, default=ROOT / 'datasets/multimnist')
    p.add_argument('--output-root', type=Path, default=ROOT / 'results/multimnist_tuning')
    p.add_argument('--lr-grid', nargs='+', type=float, default=defaults['lr_grid'])
    p.add_argument('--seed', type=int, default=defaults['seed'])
    p.add_argument('--split-seed', type=int, default=defaults['split_seed'])
    p.add_argument('--val-fraction', type=float, default=defaults['val_fraction'])
    p.add_argument('--screen-epochs', type=int, default=defaults['screen_epochs'])
    p.add_argument('--epochs', type=int, default=defaults['epochs'])
    p.add_argument('--finalists', type=int, default=defaults['finalists'])
    p.add_argument('--batch-size', type=int, default=CONFIG['common']['batch_size'])
    p.add_argument('--eta', type=float, default=CONFIG['ours']['eta'])
    p.add_argument('--alpha-grid', nargs='+', type=float, default=defaults['alpha_grid'])
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--cache-data', action=argparse.BooleanOptionalAction, default=True)
    a = p.parse_args()
    if not 0 < a.screen_epochs <= a.epochs:
        p.error('Require 0 < screen-epochs <= epochs')
    if len(set(a.lr_grid)) != len(a.lr_grid) or any(not 0 < x < float('inf') for x in a.lr_grid):
        p.error('Learning rates must be unique, finite and positive')
    if len(set(a.alpha_grid)) != len(a.alpha_grid) or any(not 0 < x <= 1 for x in a.alpha_grid):
        p.error('Momentum injection values must be unique and in (0, 1]')
    candidates=[(lr, alpha) for lr in a.lr_grid for alpha in a.alpha_grid]
    if not 1 <= a.finalists <= len(candidates):
        p.error('Invalid number of finalists')
    a.output_root.mkdir(parents=True, exist_ok=True)
    settings = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(a).items()
                if k not in {'device', 'workers', 'threads', 'cache_data', 'data_path', 'output_root'}}
    settings.update(train_sha256=data_fingerprint(a.data_path, splits=('train',)),
                    implementation='multimnist-v2', selection='validation', metric='sample')
    manifest = a.output_root / 'search.json'
    if manifest.exists() and json.loads(manifest.read_text()) != settings:
        raise ValueError('Tuning settings/data changed. Choose a new output root.')
    write_json(manifest, settings)
    env = dict(os.environ, PYTHONUNBUFFERED='1')

    def run(lr, alpha, stop):
        cmd = [sys.executable, '-u', str(ROOT / 'run_multimnist.py'), '--method', 'ours',
               '--seed', str(a.seed), '--lr', str(lr), '--eta', str(a.eta), '--alpha', str(alpha),
               '--selection', 'validation', '--metric', 'sample', '--split-seed', str(a.split_seed),
               '--val-fraction', str(a.val_fraction), '--epochs', str(a.epochs),
               '--stop-after-epoch', str(stop), '--batch-size', str(a.batch_size),
               '--data-path', str(a.data_path.resolve()), '--output-root', str(a.output_root.resolve()),
               '--tag', tag_for(lr, alpha), '--resume', '--device', a.device, '--workers', str(a.workers),
               '--threads', str(a.threads), '--cache-data' if a.cache_data else '--no-cache-data']
        print('\n' + ' '.join(cmd), flush=True)
        subprocess.run(cmd, cwd=ROOT, env=env, check=True)

    def read_row(lr, alpha, epoch):
        folder = a.output_root / tag_for(lr, alpha) / 'ours' / f'seed{a.seed}'
        config = json.loads((folder / 'config.json').read_text())
        sig = config['signature']
        if sig['selection'] != 'validation' or sig['data_sha256'] != settings['train_sha256']:
            raise ValueError('Invalid validation trial identity')
        rows = json.loads((folder / 'history.json').read_text())
        row = next((x for x in rows if x['epoch'] == epoch), None)
        if row is None:
            raise RuntimeError(f'Missing epoch {epoch} for {lr}')
        return dict(lr=lr, alpha=alpha, epoch=epoch, left=row['left'], right=row['right'],
                    avg=row['avg'], gap=row['gap'], weights=row['weights'])

    screen = []
    for i, (lr, alpha) in enumerate(candidates, 1):
        print(f'\nScreen {i}/{len(candidates)}: lr={lr:g}, alpha={alpha:g}', flush=True)
        run(lr, alpha, a.screen_epochs)
        screen.append(read_row(lr, alpha, a.screen_epochs))
    # Stable ties favor smaller LR, then smaller momentum injection.
    shortlisted = sorted(screen, key=lambda x: (-x['avg'], x['lr'], x['alpha']))[:a.finalists]
    final = []
    for i, row in enumerate(shortlisted, 1):
        print(f'\nFinalist {i}/{len(shortlisted)}: lr={row["lr"]:g}', flush=True)
        run(row['lr'], row['alpha'], a.epochs)
        final.append(read_row(row['lr'], row['alpha'], a.epochs))
    best = sorted(final, key=lambda x: (-x['avg'], x['lr'], x['alpha']))[0]
    selection = dict(settings=settings, screening=screen, finalists=final,
                     criterion='Highest sample-weighted validation avg at the final epoch; smaller LR then smaller alpha break ties.',
                     selected=dict(lr=best['lr'], eta=a.eta, alpha=best['alpha'], epochs=a.epochs,
                                   batch_size=a.batch_size, oracle='spectral', weights='entropic', momentum='blended'),
                     validation=best, test_used=False)
    write_json(a.output_root / 'selection.json', selection)
    print('\nSelected on validation only: ' + json.dumps(selection['selected']), flush=True)
    print('Retrain from scratch on all training images; do not reuse tuning checkpoints for test reporting.', flush=True)


if __name__ == '__main__':
    main()
