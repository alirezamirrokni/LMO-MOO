"""Read-only checkpoint/curve audit and CPU/GPU update-scale probes.

Only load your own trusted checkpoint.pt files (PyTorch pickle format).
Probes use training images only, restore a checkpoint before each batch, and
use the learning rate used in its last epoch, not the next scheduler rate.
Different methods have different trained weights: these are descriptive
measurements, not a controlled causal comparison or a new test result.
"""
import argparse
import copy
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from experiments.multimnist.trainer import (parser, resolve_args, make_method,
    load_method_state, train_step, evaluate, data_fingerprint, MultiMNISTViT,
    MultiMNISTDataset, CachedDataset)


def group(name, p):
    if name in ('cls_token', 'pos_emb'):
        return name
    if p.ndim < 2:
        return 'vectors_and_biases'
    if name.startswith('transformer_enc'):
        return 'encoder_matrices'
    if name.startswith('head'):
        return 'head_matrices'
    return 'patch_matrices'


def load_run(path, device):
    config = json.loads((path/'config.json').read_text())
    args = parser().parse_args([])
    for key, value in config.items():
        if hasattr(args, key):
            setattr(args, key, value)
    args.device = device
    resolve_args(args)
    ckpt = torch.load(path/'checkpoint.pt', map_location=device, weights_only=False)
    if ckpt['signature'] != config['signature']:
        raise ValueError(f'Configuration/checkpoint mismatch: {path}')
    if ckpt['epoch'] != len(ckpt['rows']):
        raise ValueError(f'Incomplete history: {path}')
    return args, ckpt


def restore(args, ckpt, device):
    model = MultiMNISTViT().to(device)
    method, optimizer = make_method(model, args, torch.device(device))
    # Deep copies are essential: loading method buffers on the same device
    # can otherwise alias and mutate the in-memory checkpoint.
    state = copy.deepcopy(ckpt)
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    load_method_state(method, state['method'])
    for pg in optimizer.param_groups:
        pg['lr'] = ckpt['rows'][-1].get('model_lr', args.lr)
    return model, method, optimizer


def write_csv(path, rows):
    if rows:
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', nargs='+', required=True, help='label=directory containing seed*/checkpoint.pt')
    p.add_argument('--data-path', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--probe-batches', type=int, default=4)
    a = p.parse_args()
    if a.probe_batches < 1:
        p.error('--probe-batches must be positive')
    a.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(a.threads)
    torch.manual_seed(2026)
    full = CachedDataset(MultiMNISTDataset(str(a.data_path), 'train'))
    train_hash = data_fingerprint(a.data_path, splits=('train',))
    hashes = {'validation': train_hash}
    summaries, scales, curves = [], [], {}
    for item in a.runs:
        label, root = item.split('=', 1)
        paths = sorted(Path(root).glob('seed*/checkpoint.pt'))
        if not paths:
            raise FileNotFoundError(root)
        curves[label] = []
        for checkpoint_path in paths:
            args, ckpt = load_run(checkpoint_path.parent, a.device)
            if args.selection not in hashes:
                hashes[args.selection] = data_fingerprint(a.data_path)
            if ckpt['signature']['data_sha256'] != hashes[args.selection]:
                raise ValueError(f'Dataset mismatch: {checkpoint_path}')
            rows = ckpt['rows']; curves[label].append((args.seed, rows))
            train = full
            if args.selection == 'validation':
                order = torch.randperm(len(full), generator=torch.Generator().manual_seed(args.split_seed))
                n = int(len(full)*args.val_fraction)
                train = Subset(full, order[n:].tolist())
            loader = DataLoader(train, batch_size=args.batch_size, shuffle=False)
            model, _, _ = restore(args, ckpt, a.device)
            _, final_train = evaluate(model, loader, torch.device(a.device))
            last = rows[-1]
            metric = lambda r: 50*(r['test_sample'][1]+r['test_sample'][3])
            summaries.append(dict(run=label, seed=args.seed, selection=args.selection,
                epochs=ckpt['epoch'], lr=args.lr, alpha=args.alpha, eta=args.eta,
                schedule=args.lr_schedule, train_samples=len(train),
                final_train_avg=50*(final_train[1]+final_train[3]),
                online_last_train_avg=50*(last['train'][1]+last['train'][3]),
                final_eval_avg=metric(last),
                last5_eval_avg=float(np.mean([metric(r) for r in rows[-5:]])),
                last20_eval_std=float(np.std([metric(r) for r in rows[-20:]], ddof=1)),
                final_gap=last.get('gap')))
            for batch_id, batch in enumerate(loader):
                if batch_id >= a.probe_batches:
                    break
                model, method, optimizer = restore(args, ckpt, a.device)
                before = {n:p.detach().clone() for n,p in model.named_parameters()}
                train_step(model, method, optimizer, tuple(v.to(a.device) for v in batch), args.method)
                for name, param in model.named_parameters():
                    delta = param.detach()-before[name]
                    scales.append(dict(run=label, seed=args.seed, batch=batch_id,
                        parameter=name, group=group(name,param), numel=param.numel(),
                        update_sq=float(delta.double().square().sum()),
                        parameter_sq=float(before[name].double().square().sum()),
                        update_rms=float(delta.double().square().mean().sqrt()),
                        relative_update=float(delta.norm()/before[name].norm().clamp_min(1e-12))))
            print(json.dumps(summaries[-1]), flush=True)
    write_csv(a.out/'checkpoint_metrics.csv', summaries)
    write_csv(a.out/'parameter_updates.csv', scales)
    grouped = []
    for label in curves:
        for category in sorted({r['group'] for r in scales}):
            part = [r for r in scales if r['run']==label and r['group']==category]
            grouped.append(dict(run=label, group=category,
                update_rms=float(np.sqrt(sum(r['update_sq'] for r in part)/sum(r['numel'] for r in part))),
                relative_update=float(np.sqrt(sum(r['update_sq'] for r in part)/max(1e-30,sum(r['parameter_sq'] for r in part))))))
    write_csv(a.out/'block_updates.csv', grouped)
    fig, axes = plt.subplots(len(curves), 2, figsize=(12, 3*len(curves)), squeeze=False)
    for (label, values), ax in zip(curves.items(), axes):
        for seed, rows in values:
            x = [r['epoch'] for r in rows]
            ax[0].plot(x, [50*(r['train'][1]+r['train'][3]) for r in rows], label=f'seed {seed}')
            ax[1].plot(x, [50*(r['test_sample'][1]+r['test_sample'][3]) for r in rows], label=f'seed {seed}')
        ax[0].set_title(label+' / online train')
        ax[1].set_title(label+' / '+next(r['selection'] for r in summaries if r['run']==label))
        for cell in ax:
            cell.set(xlabel='Epoch', ylabel='Average accuracy (%)'); cell.legend(); cell.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(a.out/'curves.png', dpi=150); plt.close(fig)
    (a.out/'audit.json').write_text(json.dumps(dict(data_sha256=hashes,
        probe_batches=a.probe_batches, device=a.device, torch_version=torch.__version__,
        note='Independent restored one-step training-only probes at last-epoch LR. No checkpoint files changed. Different methods have different trained weights.'), indent=2)+'\n')


if __name__ == '__main__':
    main()
