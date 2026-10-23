import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import default_collate

from experiments.data import DenseDataset
from experiments.models import MTANSegNet
from experiments.utils import capture_rng_state, get_device, load_method_state, method_state, restore_rng_state, set_seed
from methods import METHODS
from methods.muon import muon_optimizer, previous_update_scores

N_TASKS = {"cityscapes": 2, "nyuv2": 3}
N_CLASSES = {"cityscapes": 7, "nyuv2": 13}
BATCH_SIZE = {"cityscapes": 8, "nyuv2": 2}
METRICS = ["Semantic Loss", "Mean IoU", "Pixel Accuracy", "Depth Loss", "Absolute Error", "Relative Error",
           "Normal Loss", "Normal Mean", "Normal Median", "Normal <11.25", "Normal <22.5", "Normal <30"]


class ConfMatrix:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, pred, target):
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=pred.device)
        with torch.no_grad():
            k = (target >= 0) & (target < n)
            self.mat += torch.bincount(n * target[k].to(torch.int64) + pred[k], minlength=n ** 2).reshape(n, n)

    def get_metrics(self):
        h = self.mat.float()
        acc = torch.diag(h).sum() / h.sum()
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return torch.mean(iu).cpu().numpy(), acc.cpu().numpy()


def depth_error(x_pred, x_output):
    mask = (torch.sum(x_output, dim=1) != 0).unsqueeze(1).to(x_pred.device)
    pred, target = x_pred.masked_select(mask), x_output.masked_select(mask)
    count = torch.nonzero(mask, as_tuple=False).size(0)
    abs_err = torch.abs(pred - target)
    return (torch.sum(abs_err) / count).item(), (torch.sum(abs_err / target) / count).item()


def normal_error(x_pred, x_output):
    mask = (torch.sum(x_output, dim=1) != 0)
    error = torch.acos(torch.clamp(torch.sum(x_pred * x_output, 1).masked_select(mask), -1, 1)).detach().cpu().numpy()
    error = np.degrees(error)
    return np.mean(error), np.median(error), np.mean(error < 11.25), np.mean(error < 22.5), np.mean(error < 30)


def task_losses(prediction, batch):
    mask = (torch.sum(batch[2], dim=1) != 0).float().unsqueeze(1).to(prediction[1].device)
    losses = [F.nll_loss(prediction[0], batch[1], ignore_index=-1),
              torch.sum(torch.abs(prediction[1] - batch[2]) * mask) / torch.nonzero(mask, as_tuple=False).size(0)]
    if len(prediction) == 3:
        mask = (torch.sum(batch[3], dim=1) != 0).float().unsqueeze(1).to(prediction[2].device)
        losses.append(1 - torch.sum((prediction[2] * batch[3]) * mask) / torch.nonzero(mask, as_tuple=False).size(0))
    return torch.stack(losses)


def record(cost, offset, losses, prediction, batch):
    cost[offset] = losses[0].item()
    cost[offset + 3] = losses[1].item()
    cost[offset + 4], cost[offset + 5] = depth_error(prediction[1], batch[2])
    if len(prediction) == 3:
        cost[offset + 6] = losses[2].item()
        cost[offset + 7 : offset + 12] = normal_error(prediction[2], batch[3])


def move(raw, device):
    return (raw[0].to(device), raw[1].long().to(device), *(x.to(device) for x in raw[2:]))


def batch_indices(length, batch_size, seed, epoch, stream):
    generator = torch.Generator()
    generator.manual_seed(seed + 1000003 * epoch + 7919 * stream)
    permutation = torch.randperm(length, generator=generator).tolist()
    return [permutation[i : i + batch_size] for i in range(0, length, batch_size)]


def load(dataset, indices):
    return default_collate([dataset[i] for i in indices])


def mgda_warm_gram(model, train_set, device, batch_size, n_tasks):
    shared = list(model.shared_parameters())
    sums = [[torch.zeros_like(p, device="cpu", dtype=torch.float32) for p in shared] for _ in range(n_tasks)]
    schedule = [list(range(i, min(i + batch_size, len(train_set)))) for i in range(0, len(train_set), batch_size)]
    rng_state = capture_rng_state()
    buffers = {name: b.detach().clone() for name, b in model.named_buffers()}
    model.train()
    for indices in schedule:
        model.zero_grad(set_to_none=True)
        batch = move(load(train_set, indices), device)
        losses = task_losses(model(batch[0]), batch)
        for t in range(n_tasks):
            grads = torch.autograd.grad(losses[t], shared, retain_graph=t < n_tasks - 1, allow_unused=True)
            for i, g in enumerate(grads):
                if g is not None:
                    sums[t][i].add_(g.detach().cpu(), alpha=1.0 / len(schedule))
    with torch.no_grad():
        current = dict(model.named_buffers())
        for name, saved in buffers.items():
            current[name].copy_(saved.to(current[name].device))
    model.zero_grad(set_to_none=True)
    restore_rng_state(rng_state)
    gram = torch.zeros((n_tasks, n_tasks), dtype=torch.float32)
    for i in range(n_tasks):
        for j in range(i, n_tasks):
            value = torch.zeros((), dtype=torch.float32)
            for gi, gj in zip(sums[i], sums[j]):
                value = value + torch.sum(gi * gj)
            gram[i, j] = gram[j, i] = value
    return gram


def train(cfg, result_path, checkpoint_path):
    dataset, name = cfg.dataset, cfg.method
    n_tasks = N_TASKS[dataset]
    half = 6 * (n_tasks - 1)
    normals = dataset == "nyuv2"
    batch_size = BATCH_SIZE[dataset]

    set_seed(cfg.seed)
    device = get_device(cfg.gpu)
    model = MTANSegNet(n_tasks, N_CLASSES[dataset]).to(device)
    train_set = DenseDataset(cfg.data_path, train=True, augmentation=True, normals=normals)
    test_set = DenseDataset(cfg.data_path, train=False, normals=normals)

    kwargs = {
        "stl": dict(main_task=cfg.main_task),
        "famo": dict(gamma=cfg.famo_gamma, w_lr=cfg.method_lr, max_norm=1.0),
        "entropic_lmo_mgda": dict(eta=cfg.method_lr, alpha=cfg.alpha, tau=cfg.tau, kappa=cfg.kappa),
        "moon": dict(w_lr=cfg.method_lr, gamma=cfg.logit_decay),
    }.get(name, {})
    method = METHODS[name](n_tasks=n_tasks, device=device, **kwargs)

    if name == "moon":
        optimizer = muon_optimizer(model, cfg.lr)
    elif name == "entropic_lmo_mgda":
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr)
    elif name == "mgda_warm":
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    else:
        optimizer = torch.optim.Adam([{"params": model.parameters(), "lr": cfg.lr}, {"params": method.parameters(), "lr": cfg.method_lr}])

    train_batches = math.ceil(len(train_set) / batch_size)
    test_batches = math.ceil(len(test_set) / batch_size)
    avg_cost = np.zeros((cfg.epochs, 2 * half), dtype=np.float32)
    start = 0
    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        load_method_state(method, state["method"])
        avg_cost[: state["epoch"]] = state["avg_cost"]
        start = state["epoch"]
        restore_rng_state(state["rng"])
        print(f"resuming from epoch {start}", flush=True)
    elif name == "mgda_warm":
        method.warm_start(mgda_warm_gram(model, train_set, device, batch_size, n_tasks))

    streams = 3 if name in {"modo", "mgda_warm"} else 1
    for epoch in range(start, cfg.epochs):
        begin = time.time()
        model.train()
        if cfg.schedule == "late150":
            factor = 1.0 if epoch < 100 else 0.5 if epoch < 150 else 0.25
        else:
            factor = 0.5 ** (epoch // 100)
        for i, group in enumerate(optimizer.param_groups):
            if name == "moon" or i == 0:
                group["lr"] = cfg.lr * factor
            else:
                group["lr"] = cfg.method_lr * factor

        confusion = ConfMatrix(N_CLASSES[dataset])
        schedules = [batch_indices(len(train_set), batch_size, cfg.seed, epoch, s) for s in range(streams)]
        for b in range(train_batches):
            optimizer.zero_grad()
            batches = [move(load(train_set, schedule[b]), device) for schedule in schedules]
            if streams == 3:
                outputs = [model(batch[0], return_representation=True) for batch in batches]
                loss_list = [task_losses(prediction, batch) for (prediction, _), batch in zip(outputs, batches)]
                method.backward(losses=loss_list, shared_parameters=list(model.shared_parameters()),
                                task_specific_parameters=list(model.task_specific_parameters()),
                                last_shared_parameters=list(model.last_shared_parameters()), representation=outputs[-1][1])
                k = -1 if name == "modo" else 0
                prediction, batch, losses = outputs[k][0], batches[k], loss_list[k]
            else:
                batch = batches[0]
                prediction, representation = model(batch[0], return_representation=True)
                losses = task_losses(prediction, batch)
                extra = {}
                if name == "moon":
                    method.update(previous_update_scores(losses, list(model.parameters()), optimizer))
                if name in {"entropic_lmo_mgda", "moon"}:
                    extra["model_parameters"] = list(model.parameters())
                method.backward(losses=losses, shared_parameters=list(model.shared_parameters()),
                                task_specific_parameters=list(model.task_specific_parameters()),
                                last_shared_parameters=list(model.last_shared_parameters()), representation=representation, **extra)
            optimizer.step()
            if name == "famo":
                with torch.no_grad():
                    method.update(task_losses(model(batch[0]), batch).detach())
            confusion.update(prediction[0].argmax(1).flatten(), batch[1].flatten())
            cost = np.zeros(2 * half, dtype=np.float32)
            record(cost, 0, losses, prediction, batch)
            avg_cost[epoch, :half] += cost[:half] / train_batches
        avg_cost[epoch, 1:3] = confusion.get_metrics()

        model.eval()
        confusion = ConfMatrix(N_CLASSES[dataset])
        with torch.no_grad():
            for b in range(test_batches):
                batch = move(load(test_set, range(b * batch_size, min((b + 1) * batch_size, len(test_set)))), device)
                prediction = model(batch[0])
                confusion.update(prediction[0].argmax(1).flatten(), batch[1].flatten())
                cost = np.zeros(2 * half, dtype=np.float32)
                record(cost, half, task_losses(prediction, batch), prediction, batch)
                avg_cost[epoch, half:] += cost[half:] / test_batches
        avg_cost[epoch, half + 1 : half + 3] = confusion.get_metrics()

        test = avg_cost[epoch, half:]
        print(f"epoch {epoch + 1:03d} mIoU={100 * test[1]:.2f} pix={100 * test[2]:.2f} abs={test[4]:.4f} rel={test[5]:.4f}"
              + (f" normal_mean={test[7]:.2f}" if normals else "") + f" time={time.time() - begin:.0f}s", flush=True)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"avg_cost": avg_cost[: epoch + 1].copy(), "keys": [f"{s} {m}" for s in ("Train", "Test") for m in METRICS[:half]],
                    "completed": epoch + 1 == cfg.epochs, "config": vars(cfg)}, result_path)
        torch.save({"epoch": epoch + 1, "avg_cost": avg_cost[: epoch + 1].copy(), "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(), "method": method_state(method), "rng": capture_rng_state()}, checkpoint_path)
