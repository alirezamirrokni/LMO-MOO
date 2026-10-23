import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, default_collate

from experiments.data import MultiMNIST
from experiments.models import MultiMNISTViT
from experiments.utils import capture_rng_state, load_method_state, method_state, restore_rng_state, set_seed
from methods import FAMO, MGDA, MOON, EntropicLMOMGDA, LinearScalarization
from methods.muon import muon_optimizer, previous_update_scores


def loss_vector(pred, y0, y1):
    return torch.stack([F.cross_entropy(pred[0], y0), F.cross_entropy(pred[1], y1)])


def make_method(model, cfg, device):
    if cfg.method == "entropic_lmo_mgda":
        method = EntropicLMOMGDA(2, device, eta=cfg.eta, alpha=cfg.alpha, kappa=cfg.kappa,
                                 oracle=cfg.oracle, weights=cfg.weights, momentum=cfg.momentum)
        return method, torch.optim.SGD(model.parameters(), lr=cfg.lr)
    if cfg.method == "moon":
        method = MOON(2, device, w_lr=1e-4, gamma=1e-3)
    elif cfg.method.startswith("famo"):
        method = FAMO(2, device, w_lr=0.025, gamma=0.01)
    elif cfg.method.startswith("mgda"):
        method = MGDA(2, device)
    else:
        method = LinearScalarization(2, device)
    if cfg.method in {"moon", "mgda_muon", "famo_muon", "muon_ls"}:
        return method, muon_optimizer(model, cfg.lr, multimnist=True)
    return method, torch.optim.Adam([dict(params=model.parameters(), lr=cfg.lr), dict(params=method.parameters(), lr=0.025)])


def train_step(model, method, optimizer, batch, name):
    model.train()
    optimizer.zero_grad()
    x, y0, y1 = batch
    pred, features = model(x, return_representation=True)
    losses = loss_vector(pred, y0, y1)
    if name == "moon":
        method.update(previous_update_scores(losses, list(model.parameters()), optimizer))
    method.backward(losses, shared_parameters=list(model.shared_parameters()),
                    task_specific_parameters=list(model.task_specific_parameters()),
                    representation=features, model_parameters=list(model.parameters()))
    optimizer.step()
    if name.startswith("famo"):
        with torch.no_grad():
            method.update(loss_vector(model(x), y0, y1).detach())
    acc = [float((p.argmax(1) == y).float().mean()) for p, y in zip(pred, (y0, y1))]
    return [float(losses[0].detach()), acc[0], float(losses[1].detach()), acc[1]]


def evaluate(model, loader, device):
    model.eval()
    sums, batch_means, count = np.zeros(4), np.zeros(4), 0
    with torch.no_grad():
        for batch in loader:
            x, y0, y1 = (v.to(device) for v in batch)
            pred = model(x)
            losses = loss_vector(pred, y0, y1)
            row = np.array([float(losses[0]), float((pred[0].argmax(1) == y0).float().mean()),
                            float(losses[1]), float((pred[1].argmax(1) == y1).float().mean())])
            sums += row * len(x)
            batch_means += row
            count += len(x)
    return batch_means / len(loader), sums / count


def train(cfg, out):
    device = torch.device(cfg.device)
    set_seed(cfg.seed)
    model = MultiMNISTViT().to(device)
    method, optimizer = make_method(model, cfg, device)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    full_train = MultiMNIST(cfg.data_path, "train")
    if cfg.validation:
        order = torch.randperm(len(full_train), generator=torch.Generator().manual_seed(2026))
        n_val = int(len(full_train) * 0.1)
        test_set, train_set = Subset(full_train, order[:n_val].tolist()), Subset(full_train, order[n_val:].tolist())
    else:
        train_set, test_set = full_train, MultiMNIST(cfg.data_path, "test")
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False)

    out.mkdir(parents=True, exist_ok=True)
    checkpoint, rows, start = out / "checkpoint.pt", [], 0
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        load_method_state(method, state["method"])
        rows, start = state["rows"], state["epoch"]
        restore_rng_state(state["rng"])

    for epoch in range(start, cfg.epochs):
        begin = time.monotonic()
        train_cost = np.zeros(4)
        for batch in train_loader:
            batch = tuple(v.to(device) for v in batch)
            train_cost += np.array(train_step(model, method, optimizer, batch, cfg.method)) / len(train_loader)
        scheduler.step()
        test, test_sample = evaluate(model, test_loader, device)
        gap = None
        if cfg.method == "entropic_lmo_mgda":
            x, y0, y1 = (v.to(device) for v in default_collate([train_set[i] for i in range(min(256, len(train_set)))]))
            with torch.enable_grad():
                gap = method.inner_gap(loss_vector(model(x), y0, y1), list(model.parameters()))
        rows.append(dict(epoch=epoch + 1, left=100 * test[1], right=100 * test[3], avg=50 * (test[1] + test[3]), gap=gap,
                         weights=method.weights.cpu().tolist() if cfg.method == "entropic_lmo_mgda" else None,
                         train=train_cost.tolist(), test=test.tolist(), test_sample=test_sample.tolist(),
                         seconds=time.monotonic() - begin))
        torch.save(dict(epoch=epoch + 1, model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                        method=method_state(method), rng=capture_rng_state(), rows=rows), checkpoint)
        (out / "history.json").write_text(json.dumps(rows, indent=2))
        (out / "summary.json").write_text(json.dumps(dict(rows[-1], completed=epoch + 1 == cfg.epochs, config=vars(cfg)), indent=2, default=str))
        print(json.dumps({k: rows[-1][k] for k in ("epoch", "left", "right", "avg", "gap")}), flush=True)
