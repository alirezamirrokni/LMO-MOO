"""Focused, validation-only late-LR search; preserves the 100-epoch budget.

Four candidates, three seeds each. The incumbent's existing completed
validation histories are audited and reused; baseline/test scores are not read.
The final method, geometry, alpha and eta stay fixed.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

from experiments.multimnist.trainer import parser, run_signature, data_fingerprint, write_json

ROOT = Path(__file__).resolve().parent
SEEDS = [42, 43, 44]


def flags(settings):
    return [v for k, value in settings.items() for v in ('--'+k.replace('_', '-'), str(value))]


def validation_record(folder, expected):
    config = json.loads((folder/'config.json').read_text())
    summary = json.loads((folder/'summary.json').read_text())
    if config.get('signature') != expected or summary.get('signature') != expected:
        raise ValueError(f'Signature mismatch: {folder}')
    if not summary.get('completed') or summary.get('selection') != 'validation' or summary.get('smoke'):
        raise ValueError(f'Expected completed validation run: {folder}')
    rows = json.loads((folder/'history.json').read_text())
    if [r['epoch'] for r in rows] != list(range(1, 101)):
        raise ValueError(f'Expected exactly 100 consecutive epochs: {folder}')
    values = [50*(r['test_sample'][1]+r['test_sample'][3]) for r in rows[-5:]]
    if not all(math.isfinite(v) for v in values):
        raise ValueError(f'Nonfinite validation metric: {folder}')
    return dict(seed=expected['seed'], score=statistics.mean(values), final_avg=values[-1])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-path', type=Path, required=True)
    p.add_argument('--previous-search', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--threads', type=int, default=4)
    a = p.parse_args()
    old = json.loads((a.previous_search/'selection.json').read_text())
    if old.get('test_used') is not False or old['settings']['selection'] != 'validation':
        raise ValueError('Expected a validation-only incumbent')
    selected = old['selected']
    for key, value in dict(epochs=100, batch_size=256, oracle='spectral', weights='entropic', momentum='blended', lr_schedule='reference').items():
        if selected[key] != value:
            raise ValueError(f'Unexpected incumbent {key}: {selected[key]}')
    train_hash = data_fingerprint(a.data_path, splits=('train',))
    if train_hash != old['settings']['train_sha256']:
        raise ValueError('Incumbent used different training data')
    common = ['--method', 'ours', '--selection', 'validation', '--metric', 'sample',
        '--data-path', str(a.data_path.resolve()), '--output-root', str(a.output_root.resolve()),
        '--split-seed', str(old['settings']['split_seed']), '--val-fraction', str(old['settings']['val_fraction']),
        '--device', a.device, '--threads', str(a.threads), '--workers', '0', '--cache-data', '--resume']
    incumbent_tag = old['validation']['tag']
    incumbent_rows = []
    for seed in SEEDS:
        expected = run_signature(parser().parse_args(common + flags(selected) + ['--seed', str(seed), '--tag', incumbent_tag]), train_hash)
        incumbent_rows.append(validation_record(a.previous_search/incumbent_tag/'ours'/f'seed{seed}', expected))
    def record(tag, settings, rows):
        return dict(tag=tag, config=settings, seeds=rows,
            score=statistics.mean(r['score'] for r in rows), score_std=statistics.stdev(r['score'] for r in rows))
    incumbent = record(incumbent_tag, selected, incumbent_rows)
    if not math.isclose(incumbent['score'], old['validation']['score'], abs_tol=1e-6):
        raise ValueError('Incumbent histories disagree with saved selection')
    candidates = [(f'late-{start}-floor-{floor:g}', dict(selected,
        lr_schedule='late-cosine', cooldown_start=start, min_lr_ratio=floor))
        for start in (60,80) for floor in (0.1,0.3)]
    digest = hashlib.sha256()
    for path in [Path(__file__), *sorted((ROOT/'experiments/multimnist').glob('*.py')), *sorted((ROOT/'methods').glob('*.py'))]:
        digest.update(path.relative_to(ROOT).as_posix().encode()); digest.update(path.read_bytes())
    protocol = dict(version=1, selection='validation', train_sha256=train_hash,
        split_seed=old['settings']['split_seed'], val_fraction=old['settings']['val_fraction'],
        seeds=SEEDS, epochs=100, batch_size=256, candidates=candidates,
        incumbent=incumbent, source_sha256=digest.hexdigest(), score_window=5,
        minimum_improvement_pp=0.1,
        criterion='Mean of each seed\'s sample-weighted validation average over epochs 96-100; lower seed std breaks ties. Retain incumbent unless improvement exceeds 0.1 percentage point.')
    # JSON normalizes tuples to lists before identity comparisons.
    protocol = json.loads(json.dumps(protocol))
    a.output_root.mkdir(parents=True, exist_ok=True)
    manifest = a.output_root/'refinement.json'
    if manifest.exists() and json.loads(manifest.read_text()) != protocol:
        raise ValueError('Refinement protocol/code/data changed. Use a new output directory.')
    write_json(manifest, protocol)
    records = [incumbent]
    for index, (tag, settings) in enumerate(candidates, 1):
        rows = []
        for seed in SEEDS:
            args = common + flags(settings) + ['--seed', str(seed), '--tag', tag]
            folder = a.output_root/tag/'ours'/f'seed{seed}'
            expected = run_signature(parser().parse_args(args), train_hash)
            print(f'Candidate {index}/4: {tag}, seed {seed}; validation only', flush=True)
            summary_path = folder/'summary.json'
            if not summary_path.exists() or not json.loads(summary_path.read_text()).get('completed'):
                subprocess.run([sys.executable, '-u', str(ROOT/'run_multimnist.py'), *args],
                    cwd=ROOT, env=dict(os.environ, PYTHONUNBUFFERED='1'), check=True)
            rows.append(validation_record(folder, expected))
        records.append(record(tag, settings, rows))
        write_json(a.output_root/'partial.json', records)
    ranked = sorted(records, key=lambda r: (-r['score'], r['score_std'], r['tag']))
    winner = ranked[0]
    if winner['score'] <= incumbent['score'] + protocol['minimum_improvement_pp']:
        winner = incumbent
    result = dict(settings=protocol, selected=winner['config'], validation=winner,
        incumbent_validation=incumbent, confirmation=ranked, test_used=False,
        changed=winner['tag'] != incumbent_tag, improvement_over_incumbent=winner['score']-incumbent['score'])
    write_json(a.output_root/'selection.json', result)
    print(json.dumps(result, indent=2), flush=True)
    print('Selection frozen. Fresh full-training runs are required only if changed=true.', flush=True)


if __name__ == '__main__':
    main()
