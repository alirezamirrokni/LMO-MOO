"""Two-epoch training-only continuation diagnostic, never a comparison-table run.

Use trusted validation checkpoints only. Original checkpoints are read-only.
All learning-rate branches start at the same checkpoint and use the same
minibatch order. This diagnoses sensitivity; it does not preserve the original
100-epoch budget and must not be substituted for a fresh final experiment.
"""
import argparse
import json
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader, Subset
from diagnose_multimnist import load_run, restore
from experiments.multimnist.trainer import CachedDataset, MultiMNISTDataset, data_fingerprint, train_step, evaluate, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--validation-root', type=Path, required=True, help='Directory containing seed*/checkpoint.pt')
    p.add_argument('--data-path', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=2)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    data = CachedDataset(MultiMNISTDataset(str(a.data_path), 'train'))
    fingerprint = data_fingerprint(a.data_path, splits=('train',))
    rows = []
    for seed in (42,43,44):
        args, ckpt = load_run(a.validation_root/f'seed{seed}', a.device)
        if args.selection != 'validation' or args.method != 'ours' or args.seed != seed or args.epochs != 100 or ckpt['epoch'] != 100:
            raise ValueError('Expected completed 100-epoch Ours validation checkpoint')
        if ckpt['signature']['data_sha256'] != fingerprint:
            raise ValueError('Training data mismatch')
        order = torch.randperm(len(data), generator=torch.Generator().manual_seed(args.split_seed))
        n = int(len(data)*args.val_fraction)
        train = Subset(data, order[n:].tolist())
        val = DataLoader(Subset(data, order[:n].tolist()), batch_size=args.batch_size)
        for lr in (.01, .003, .001):
            model, method, optimizer = restore(args, ckpt, a.device)
            for pg in optimizer.param_groups:
                pg['lr'] = lr
            loader = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                generator=torch.Generator().manual_seed(7000+seed))
            for epoch in (1,2):
                begin = time.monotonic()
                for batch in loader:
                    train_step(model, method, optimizer, tuple(v.to(a.device) for v in batch), 'ours')
                _, ev = evaluate(model, val, torch.device(a.device))
                row = dict(seed=seed, lr=lr, extra_epoch=epoch,
                    validation_avg=50*(ev[1]+ev[3]), seconds=time.monotonic()-begin)
                rows.append(row); print(json.dumps(row), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(a.out, dict(rows=rows, test_used=False, training_sha256=fingerprint,
        note='Two extra epochs: diagnostic only, not a 100-epoch benchmark result.'))


if __name__ == '__main__':
    main()
