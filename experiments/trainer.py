import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import default_collate

from experiments.checkpointing import (
    atomic_torch_save,
    capture_rng_state,
    load_torch,
    restore_rng_state,
)
from experiments.utils import get_device, method_parameters, set_seed
from methods import WeightMethods


LMO_METHODS = {"entropic_lmo_mgda"}


def _semantic_loss(prediction, target):
    return F.nll_loss(prediction, target, ignore_index=-1)


def _masked_binary_mask(target):
    return (torch.sum(target, dim=1) != 0).float().unsqueeze(1)


def _depth_loss(prediction, target):
    mask = _masked_binary_mask(target).to(prediction.device)
    denominator = torch.nonzero(mask, as_tuple=False).size(0)
    return torch.sum(torch.abs(prediction - target) * mask) / denominator


def _normal_loss(prediction, target):
    mask = _masked_binary_mask(target).to(prediction.device)
    denominator = torch.nonzero(mask, as_tuple=False).size(0)
    return 1 - torch.sum((prediction * target) * mask) / denominator


def _dataset_components(dataset_name):
    if dataset_name == "cityscapes":
        from experiments.cityscapes.data import Cityscapes
        from experiments.cityscapes.models import SegNet, SegNetMtan
        from experiments.cityscapes.utils import ConfMatrix, delta_fn, depth_error

        return {
            "dataset": Cityscapes,
            "models": {"segnet": SegNet, "mtan": SegNetMtan},
            "confusion": ConfMatrix,
            "delta": delta_fn,
            "depth_error": depth_error,
            "normal_error": None,
            "tasks": ("semantic", "depth"),
            "metric_size": 12,
        }

    if dataset_name == "nyuv2":
        from experiments.nyuv2.data import NYUv2
        from experiments.nyuv2.models import SegNet, SegNetMtan
        from experiments.nyuv2.utils import (
            ConfMatrix,
            delta_fn,
            depth_error,
            normal_error,
        )

        return {
            "dataset": NYUv2,
            "models": {"segnet": SegNet, "mtan": SegNetMtan},
            "confusion": ConfMatrix,
            "delta": delta_fn,
            "depth_error": depth_error,
            "normal_error": normal_error,
            "tasks": ("semantic", "depth", "normal"),
            "metric_size": 24,
        }

    raise ValueError(f"unknown dataset {dataset_name}")


def _move_batch(batch, dataset_name, device):
    if dataset_name == "cityscapes":
        image, semantic, depth = batch
        return (
            image.to(device),
            semantic.long().to(device),
            depth.to(device),
        )

    image, semantic, depth, normal = batch
    return (
        image.to(device),
        semantic.long().to(device),
        depth.to(device),
        normal.to(device),
    )


def _forward_losses(model, batch, dataset_name):
    prediction, representation = model(batch[0], return_representation=True)

    if dataset_name == "cityscapes":
        losses = torch.stack(
            (
                _semantic_loss(prediction[0], batch[1]),
                _depth_loss(prediction[1], batch[2]),
            )
        )
    else:
        losses = torch.stack(
            (
                _semantic_loss(prediction[0], batch[1]),
                _depth_loss(prediction[1], batch[2]),
                _normal_loss(prediction[2], batch[3]),
            )
        )

    return prediction, representation, losses


def _updated_losses(model, batch, dataset_name):
    prediction = model(batch[0], return_representation=False)
    if dataset_name == "cityscapes":
        return torch.stack(
            (
                _semantic_loss(prediction[0], batch[1]),
                _depth_loss(prediction[1], batch[2]),
            )
        )

    return torch.stack(
        (
            _semantic_loss(prediction[0], batch[1]),
            _depth_loss(prediction[1], batch[2]),
            _normal_loss(prediction[2], batch[3]),
        )
    )


def _result_keys(dataset_name):
    if dataset_name == "cityscapes":
        return [
            "Train Semantic Loss",
            "Train Mean IoU",
            "Train Pixel Accuracy",
            "Train Depth Loss",
            "Train Absolute Error",
            "Train Relative Error",
            "Test Semantic Loss",
            "Test Mean IoU",
            "Test Pixel Accuracy",
            "Test Depth Loss",
            "Test Absolute Error",
            "Test Relative Error",
        ]

    return [
        "Train Semantic Loss",
        "Train Mean IoU",
        "Train Pixel Accuracy",
        "Train Depth Loss",
        "Train Absolute Error",
        "Train Relative Error",
        "Train Normal Loss",
        "Train Normal Mean",
        "Train Normal Median",
        "Train Normal <11.25",
        "Train Normal <22.5",
        "Train Normal <30",
        "Test Semantic Loss",
        "Test Mean IoU",
        "Test Pixel Accuracy",
        "Test Depth Loss",
        "Test Absolute Error",
        "Test Relative Error",
        "Test Normal Loss",
        "Test Normal Mean",
        "Test Normal Median",
        "Test Normal <11.25",
        "Test Normal <22.5",
        "Test Normal <30",
    ]


def _normalized_config(args):
    config = vars(args).copy()
    for key, value in list(config.items()):
        if isinstance(value, Path):
            config[key] = str(value)
    return config


def _resume_signature(args, dataset_name):
    config = _normalized_config(args)
    keys = [
        "n_epochs",
        "batch_size",
        "method",
        "lr",
        "method_params_lr",
        "seed",
        "main_task",
        "c",
        "gamma",
        "max_norm",
        "dwa_temp",
        "nashmtl_optim_niter",
        "update_weights_every",
        "moco_beta",
        "moco_beta_sigma",
        "moco_gamma",
        "moco_gamma_sigma",
        "moco_rho",
        "modo_gamma",
        "modo_rho",
        "mgda_warm_beta",
        "mgda_warm_rho",
        "mgda_warm_steps",
        "mgda_warm_beta0",
        "lmo_alpha",
        "muon_ns_steps",
        "model",
        "apply_augmentation",
        "data_path",
    ]

    signature = {
        "dataset": dataset_name,
        **{key: config[key] for key in keys},
    }

    if config["method"] in {"entropic_lmo_mgda", "moon"}:
        signature["lmo_lr_schedule"] = config["lmo_lr_schedule"]
        signature["muon_ns_steps"] = config["muon_ns_steps"]
        if config["method"] == "entropic_lmo_mgda":
            signature["entropy_tau"] = config["entropy_tau"]
            signature["weight_kappa"] = config["weight_kappa"]
        else:
            signature["moon_logit_decay"] = config["moon_logit_decay"]
            signature["moon_implementation"] = "upstream-37319d1-v1"

    return signature


def _save_result(
    path,
    summary_path,
    avg_cost,
    deltas,
    epoch_times,
    wall_time_seconds,
    epoch,
    args,
    dataset_name,
    completed,
    losses,
):
    common = {
        "avg_cost": avg_cost[: epoch + 1].copy(),
        "delta_m": deltas[: epoch + 1].copy(),
        "keys": _result_keys(dataset_name),
        "completed_epochs": epoch + 1,
        "n_epochs": args.n_epochs,
        "completed": bool(completed),
        "method": args.method,
        "dataset": dataset_name,
        "seed": args.seed,
        "config": _normalized_config(args),
        "epoch_times_seconds": epoch_times[: epoch + 1].copy(),
        "wall_time_seconds": float(wall_time_seconds),
    }
    payload = dict(common)
    payload["losses"] = losses
    atomic_torch_save(payload, path)
    atomic_torch_save(common, summary_path)


def _train_cost_city(cost, losses, prediction, batch, depth_error):
    cost[0] = losses[0].item()
    cost[3] = losses[1].item()
    cost[4], cost[5] = depth_error(prediction[1], batch[2])


def _train_cost_nyu(cost, losses, prediction, batch, depth_error, normal_error):
    cost[0] = losses[0].item()
    cost[3] = losses[1].item()
    cost[4], cost[5] = depth_error(prediction[1], batch[2])
    cost[6] = losses[2].item()
    cost[7], cost[8], cost[9], cost[10], cost[11] = normal_error(
        prediction[2], batch[3]
    )


def _evaluation_batch(model, raw_batch, dataset_name, device, components, avg_cost, epoch, confusion, test_batches):
    batch = _move_batch(raw_batch, dataset_name, device)
    prediction = model(batch[0])

    if dataset_name == "cityscapes":
        losses = torch.stack(
            (
                _semantic_loss(prediction[0], batch[1]),
                _depth_loss(prediction[1], batch[2]),
            )
        )
        confusion.update(
            prediction[0].argmax(1).flatten(),
            batch[1].flatten(),
        )
        cost = np.zeros(components["metric_size"], dtype=np.float32)
        cost[6] = losses[0].item()
        cost[9] = losses[1].item()
        cost[10], cost[11] = components["depth_error"](
            prediction[1],
            batch[2],
        )
        avg_cost[epoch, 6:] += cost[6:] / test_batches
        return

    losses = torch.stack(
        (
            _semantic_loss(prediction[0], batch[1]),
            _depth_loss(prediction[1], batch[2]),
            _normal_loss(prediction[2], batch[3]),
        )
    )
    confusion.update(
        prediction[0].argmax(1).flatten(),
        batch[1].flatten(),
    )
    cost = np.zeros(components["metric_size"], dtype=np.float32)
    cost[12] = losses[0].item()
    cost[15] = losses[1].item()
    cost[16], cost[17] = components["depth_error"](
        prediction[1],
        batch[2],
    )
    cost[18] = losses[2].item()
    (
        cost[19],
        cost[20],
        cost[21],
        cost[22],
        cost[23],
    ) = components["normal_error"](prediction[2], batch[3])
    avg_cost[epoch, 12:] += cost[12:] / test_batches


def _finalize_evaluation(dataset_name, components, avg_cost, epoch, confusion):
    if dataset_name == "cityscapes":
        avg_cost[epoch, 7:9] = confusion.get_metrics()
        metrics = avg_cost[epoch, [7, 8, 10, 11]]
    else:
        avg_cost[epoch, 13:15] = confusion.get_metrics()
        metrics = avg_cost[epoch, [13, 14, 16, 17, 19, 20, 21, 22, 23]]
    return components["delta"](metrics)


def _epoch_batch_indices(length, batch_size, seed, epoch, stream):
    generator = torch.Generator()
    generator.manual_seed(seed + 1000003 * epoch + 7919 * stream)
    permutation = torch.randperm(length, generator=generator).tolist()
    return [
        permutation[index : index + batch_size]
        for index in range(0, length, batch_size)
    ]


def _sequential_batch_indices(length, batch_size):
    return [
        list(range(index, min(index + batch_size, length)))
        for index in range(0, length, batch_size)
    ]


def _load_index_batch(dataset, indices):
    return default_collate([dataset[index] for index in indices])

def _mgda_warm_full_gram(
    model,
    train_set,
    dataset_name,
    device,
    batch_size,
):
    shared_parameters = list(model.shared_parameters())
    n_tasks = 2 if dataset_name == "cityscapes" else 3

    gradient_sums = [
        [
            torch.zeros_like(
                parameter,
                device="cpu",
                dtype=torch.float32,
            )
            for parameter in shared_parameters
        ]
        for _ in range(n_tasks)
    ]

    schedule = _sequential_batch_indices(
        len(train_set),
        batch_size,
    )

    rng_state = capture_rng_state()
    was_training = model.training

    buffer_state = {
        name: buffer.detach().clone()
        for name, buffer in model.named_buffers()
    }

    model.train()

    try:
        for batch_index, indices in enumerate(schedule):
            model.zero_grad(set_to_none=True)

            raw_batch = _load_index_batch(
                train_set,
                indices,
            )

            batch = _move_batch(
                raw_batch,
                dataset_name,
                device,
            )

            _, _, losses = _forward_losses(
                model,
                batch,
                dataset_name,
            )

            batch_weight = 1.0 / len(schedule)

            for task_index in range(n_tasks):
                gradients = torch.autograd.grad(
                    losses[task_index],
                    shared_parameters,
                    retain_graph=task_index < n_tasks - 1,
                    allow_unused=True,
                )

                for parameter_index, gradient in enumerate(gradients):
                    if gradient is None:
                        continue

                    gradient_sums[task_index][parameter_index].add_(
                        gradient.detach().cpu(),
                        alpha=batch_weight,
                    )

            if (
                batch_index == 0
                or (batch_index + 1) % 25 == 0
                or batch_index + 1 == len(schedule)
            ):
                print(
                    "MGDA-WARM FULL-GRAD "
                    f"{batch_index + 1}/{len(schedule)}",
                    flush=True,
                )

    finally:
        with torch.no_grad():
            current_buffers = dict(model.named_buffers())

            for name, saved_buffer in buffer_state.items():
                current_buffers[name].copy_(
                    saved_buffer.to(current_buffers[name].device)
                )

        model.train(was_training)
        model.zero_grad(set_to_none=True)
        restore_rng_state(rng_state)

    gram = torch.zeros(
        (n_tasks, n_tasks),
        dtype=torch.float32,
    )

    for task_i in range(n_tasks):
        for task_j in range(task_i, n_tasks):
            value = torch.zeros(
                (),
                dtype=torch.float32,
            )

            for gradient_i, gradient_j in zip(
                gradient_sums[task_i],
                gradient_sums[task_j],
            ):
                value = value + torch.sum(
                    gradient_i * gradient_j
                )

            gram[task_i, task_j] = value
            gram[task_j, task_i] = value

    print(
        "MGDA-WARM FULL-GRAM "
        f"{gram.tolist()}",
        flush=True,
    )

    return gram


def _checkpoint_payload(
    args,
    dataset_name,
    phase,
    epoch,
    next_batch_index,
    avg_cost,
    deltas,
    epoch_times,
    confusion,
    model,
    optimizer,
    weight_method,
    wall_time_seconds,
    epoch_time_seconds,
):
    return {
        "version": 3,
        "signature": _resume_signature(args, dataset_name),
        "phase": phase,
        "epoch": epoch,
        "next_batch_index": next_batch_index,
        "avg_cost": avg_cost.copy(),
        "delta_m": deltas.copy(),
        "epoch_times_seconds": epoch_times.copy(),
        "confusion": None
        if confusion is None or confusion.mat is None
        else confusion.mat.detach().cpu(),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "weight_method": weight_method.state_dict(),
        "rng_state": capture_rng_state(),
        "wall_time_seconds": float(wall_time_seconds),
        "epoch_time_seconds": float(epoch_time_seconds),
    }


def _restore_confusion(components, model, value, device):
    confusion = components["confusion"](model.segnet.class_nb)
    if value is not None:
        confusion.mat = value.to(device)
    return confusion


def _loss_memmap(path, rows, columns, reset, checkpoint_exists):
    path.parent.mkdir(parents=True, exist_ok=True)
    if reset and path.exists():
        path.unlink()
    if checkpoint_exists and not path.exists():
        raise RuntimeError(f"checkpoint loss cache is missing: {path}")
    expected_bytes = rows * columns * np.dtype(np.float32).itemsize
    if path.exists() and path.stat().st_size != expected_bytes:
        raise RuntimeError(f"checkpoint loss cache has an unexpected size: {path}")
    mode = "r+" if path.exists() else "w+"
    return np.memmap(
        path,
        dtype=np.float32,
        mode=mode,
        shape=(rows, columns),
    )


def _set_epoch_learning_rates(optimizer, args, epoch):
    if (
        args.method in LMO_METHODS
        and getattr(args, "lmo_lr_schedule", "legacy") == "late150"
    ):
        if epoch < 100:
            factor = 1.0
        elif epoch < 150:
            factor = 0.5
        else:
            factor = 0.25
    else:
        factor = 0.5 ** (epoch // 100)
    if args.method in {"moon"}:
        for group in optimizer.param_groups:
            group["lr"] = args.lr * factor
    else:
        optimizer.param_groups[0]["lr"] = args.lr * factor
        if len(optimizer.param_groups) > 1:
            optimizer.param_groups[1]["lr"] = args.method_params_lr * factor

def _print_loss_monitor(dataset_name, avg_cost, epoch):
    if dataset_name == "cityscapes":
        print(
            "LOSS-MONITOR "
            f"epoch={epoch + 1:03d} "
            f"train_semantic={avg_cost[epoch, 0]:.6f} "
            f"train_depth={avg_cost[epoch, 3]:.6f} "
            f"test_semantic={avg_cost[epoch, 6]:.6f} "
            f"test_depth={avg_cost[epoch, 9]:.6f}",
            flush=True,
        )
        return

    print(
        "LOSS-MONITOR "
        f"epoch={epoch + 1:03d} "
        f"train_semantic={avg_cost[epoch, 0]:.6f} "
        f"train_depth={avg_cost[epoch, 3]:.6f} "
        f"train_normal={avg_cost[epoch, 6]:.6f} "
        f"test_semantic={avg_cost[epoch, 12]:.6f} "
        f"test_depth={avg_cost[epoch, 15]:.6f} "
        f"test_normal={avg_cost[epoch, 18]:.6f}",
        flush=True,
    )


def _loss_monitor_rows(dataset_name, avg_cost, completed_epochs):
    rows = []
    if dataset_name == "cityscapes":
        for index in range(completed_epochs):
            rows.append(
                (
                    index + 1,
                    float(avg_cost[index, 0]),
                    float(avg_cost[index, 3]),
                    float(avg_cost[index, 6]),
                    float(avg_cost[index, 9]),
                )
            )
        return (
            ["epoch", "train_semantic", "train_depth", "test_semantic", "test_depth"],
            rows,
        )

    for index in range(completed_epochs):
        rows.append(
            (
                index + 1,
                float(avg_cost[index, 0]),
                float(avg_cost[index, 3]),
                float(avg_cost[index, 6]),
                float(avg_cost[index, 12]),
                float(avg_cost[index, 15]),
                float(avg_cost[index, 18]),
            )
        )
    return (
        [
            "epoch",
            "train_semantic",
            "train_depth",
            "train_normal",
            "test_semantic",
            "test_depth",
            "test_normal",
        ],
        rows,
    )


def _write_loss_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(",".join(header) + "\n")
        for row in rows:
            handle.write(",".join(str(value) for value in row) + "\n")
    tmp.replace(path)


def _svg_polyline(points, x0, y0, width, height, x_min, x_max, y_min, y_max):
    if not points:
        return ""
    x_span = max(float(x_max - x_min), 1.0)
    y_span = max(float(y_max - y_min), 1e-12)
    coords = []
    for x, y in points:
        px = x0 + (float(x) - x_min) / x_span * width
        py = y0 + height - (float(y) - y_min) / y_span * height
        coords.append(f"{px:.2f},{py:.2f}")
    return " ".join(coords)


def _write_loss_svg(path, rows, dataset_name):
    # Dependency-free SVG so loss monitoring does not depend on matplotlib being
    # installed in the training environment.
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    width = 1000
    height = 760
    left = 90
    right = 35
    top = 70
    bottom = 55
    gap = 75
    panel_h = (height - top - bottom - gap) / 2.0
    panel_w = width - left - right
    epochs = [row[0] for row in rows]
    x_min = min(epochs)
    x_max = max(epochs)

    if dataset_name == "cityscapes":
        panels = [
            ("Semantic loss", 1, 3),
            ("Depth loss", 2, 4),
        ]
    else:
        # Keep the automatic plot compact for NYUv2: semantic and depth here;
        # normal loss remains available in the CSV/logs.
        panels = [
            ("Semantic loss", 1, 4),
            ("Depth loss", 2, 5),
        ]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="500" y="34" text-anchor="middle" font-family="sans-serif" font-size="22" font-weight="bold">Training / test task losses</text>',
        f'<text x="500" y="55" text-anchor="middle" font-family="sans-serif" font-size="13">through epoch {x_max}</text>',
    ]

    colors = {"train": "#1f77b4", "test": "#d62728"}
    for panel_index, (title, train_col, test_col) in enumerate(panels):
        y0 = top + panel_index * (panel_h + gap)
        train_values = [float(row[train_col]) for row in rows]
        test_values = [float(row[test_col]) for row in rows]
        all_values = train_values + test_values
        y_min = min(all_values)
        y_max = max(all_values)
        if y_max <= y_min:
            y_max = y_min + 1e-6
        margin = 0.05 * (y_max - y_min)
        y_min = max(0.0, y_min - margin)
        y_max = y_max + margin

        parts.append(f'<text x="{left}" y="{y0 - 14:.1f}" font-family="sans-serif" font-size="17" font-weight="bold">{title}</text>')
        parts.append(f'<rect x="{left}" y="{y0:.1f}" width="{panel_w}" height="{panel_h:.1f}" fill="none" stroke="#444" stroke-width="1"/>')

        for tick in range(5):
            fraction = tick / 4.0
            yy = y0 + panel_h - fraction * panel_h
            value = y_min + fraction * (y_max - y_min)
            parts.append(f'<line x1="{left}" y1="{yy:.2f}" x2="{left + panel_w}" y2="{yy:.2f}" stroke="#dddddd" stroke-width="1"/>')
            parts.append(f'<text x="{left - 10}" y="{yy + 4:.2f}" text-anchor="end" font-family="monospace" font-size="11">{value:.4f}</text>')

        x_ticks = sorted(set([x_min, x_max] + [e for e in epochs if e % 10 == 0]))
        for value in x_ticks:
            xx = left if x_max == x_min else left + (value - x_min) / (x_max - x_min) * panel_w
            parts.append(f'<line x1="{xx:.2f}" y1="{y0}" x2="{xx:.2f}" y2="{y0 + panel_h:.2f}" stroke="#eeeeee" stroke-width="1"/>')
            parts.append(f'<text x="{xx:.2f}" y="{y0 + panel_h + 20:.2f}" text-anchor="middle" font-family="sans-serif" font-size="11">{value}</text>')

        train_points = list(zip(epochs, train_values))
        test_points = list(zip(epochs, test_values))
        train_poly = _svg_polyline(train_points, left, y0, panel_w, panel_h, x_min, x_max, y_min, y_max)
        test_poly = _svg_polyline(test_points, left, y0, panel_w, panel_h, x_min, x_max, y_min, y_max)
        parts.append(f'<polyline points="{train_poly}" fill="none" stroke="{colors["train"]}" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>')
        parts.append(f'<polyline points="{test_poly}" fill="none" stroke="{colors["test"]}" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>')

        parts.append(f'<line x1="{left + panel_w - 190}" y1="{y0 - 16:.1f}" x2="{left + panel_w - 165}" y2="{y0 - 16:.1f}" stroke="{colors["train"]}" stroke-width="3"/>')
        parts.append(f'<text x="{left + panel_w - 158}" y="{y0 - 11:.1f}" font-family="sans-serif" font-size="12">train</text>')
        parts.append(f'<line x1="{left + panel_w - 95}" y1="{y0 - 16:.1f}" x2="{left + panel_w - 70}" y2="{y0 - 16:.1f}" stroke="{colors["test"]}" stroke-width="3"/>')
        parts.append(f'<text x="{left + panel_w - 63}" y="{y0 - 11:.1f}" font-family="sans-serif" font-size="12">test</text>')

    parts.append(f'<text x="500" y="{height - 15}" text-anchor="middle" font-family="sans-serif" font-size="13">Epoch</text>')
    parts.append("</svg>")

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(parts), encoding="utf-8")
    tmp.replace(path)


def _update_loss_monitor(args, dataset_name, avg_cost, epoch, force_plot=False):
    completed_epochs = epoch + 1
    header, rows = _loss_monitor_rows(dataset_name, avg_cost, completed_epochs)
    monitor_dir = args.output_dir.parent / "loss_monitor" / dataset_name
    csv_path = monitor_dir / f"{args.result_name}.losses.csv"
    _write_loss_csv(csv_path, header, rows)

    if completed_epochs % 10 == 0 or force_plot:
        snapshot_path = monitor_dir / f"{args.result_name}.losses_epoch{completed_epochs:03d}.svg"
        latest_path = monitor_dir / f"{args.result_name}.losses_latest.svg"
        _write_loss_svg(snapshot_path, rows, dataset_name)
        _write_loss_svg(latest_path, rows, dataset_name)
        print(
            f"LOSS-PLOT epoch={completed_epochs:03d} path={snapshot_path} latest={latest_path}",
            flush=True,
        )


def _print_epoch(dataset_name, avg_cost, deltas, epoch, wall_seconds):
    if dataset_name == "cityscapes":
        print(
            f"Epoch {epoch + 1:03d}: "
            f"mIoU={100 * avg_cost[epoch, 7]:.3f} "
            f"pix={100 * avg_cost[epoch, 8]:.3f} "
            f"abs={avg_cost[epoch, 10]:.4f} "
            f"rel={avg_cost[epoch, 11]:.4f} "
            f"delta={deltas[epoch]:.3f} "
            f"wall_h={wall_seconds / 3600:.3f}",
            flush=True,
        )
        return
    print(
        f"Epoch {epoch + 1:03d}: "
        f"mIoU={100 * avg_cost[epoch, 13]:.3f} "
        f"pix={100 * avg_cost[epoch, 14]:.3f} "
        f"abs={avg_cost[epoch, 16]:.4f} "
        f"rel={avg_cost[epoch, 17]:.4f} "
        f"normal_mean={avg_cost[epoch, 19]:.3f} "
        f"delta={deltas[epoch]:.3f} "
        f"wall_h={wall_seconds / 3600:.3f}",
        flush=True,
    )


def train(args, dataset_name):
    components = _dataset_components(dataset_name)
    set_seed(args.seed)
    device = get_device(args.gpu)
    model = components["models"][args.model]().to(device)
    dataset_cls = components["dataset"]

    train_set = dataset_cls(
        root=args.data_path.as_posix(),
        train=True,
        augmentation=args.apply_augmentation,
    )
    test_set = dataset_cls(root=args.data_path.as_posix(), train=False)

    weight_method_kwargs = method_parameters(args)

    if args.method == "entropic_lmo_mgda":
        weight_method_kwargs = dict(eta=args.method_params_lr, alpha=args.lmo_alpha,
                                    ns_steps=args.muon_ns_steps, entropy_tau=args.entropy_tau,
                                    weight_kappa=args.weight_kappa)

    if args.method == "moon":
        weight_method_kwargs = dict(w_lr=args.method_params_lr, gamma=args.moon_logit_decay)

    weight_method = WeightMethods(
        args.method,
        n_tasks=len(components["tasks"]),
        device=device,
        **weight_method_kwargs,
    )

    if args.method == "moon":
        from methods.moon import make_reference_optimizer
        optimizer = make_reference_optimizer(model, dataset_name, args.lr)
        print("OPTIMIZER official MOON Muon+AdamW (previous-update scores)", flush=True)
    elif args.method == "entropic_lmo_mgda":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        print(f"OPTIMIZER Entropic LMO-MGDA lr={args.lr} eta={args.method_params_lr}", flush=True)
    elif args.method == "mgda_warm":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
        )
    else:
        optimizer = torch.optim.Adam(
            [
                {"params": model.parameters(), "lr": args.lr},
                {
                    "params": weight_method.parameters(),
                    "lr": args.method_params_lr,
                },
            ]
        )

    result_path = args.output_dir / f"{args.result_name}.stats"
    summary_path = args.summary_dir / f"{args.result_name}.summary"
    checkpoint_path = args.checkpoint_dir / f"{args.result_name}.pt"
    loss_path = args.checkpoint_dir / f"{args.result_name}.losses.dat"
    train_batches = math.ceil(len(train_set) / args.batch_size)
    test_batches = math.ceil(len(test_set) / args.batch_size)
    test_schedule = _sequential_batch_indices(len(test_set), args.batch_size)
    total_loss_rows = args.n_epochs * train_batches
    checkpoint_exists = checkpoint_path.exists() and not args.force
    loss_memmap = _loss_memmap(
        loss_path,
        total_loss_rows,
        len(components["tasks"]),
        reset=args.force or not checkpoint_exists,
        checkpoint_exists=checkpoint_exists,
    )

    avg_cost = np.zeros(
        (args.n_epochs, components["metric_size"]),
        dtype=np.float32,
    )
    deltas = np.zeros(args.n_epochs, dtype=np.float32)
    epoch_times = np.zeros(args.n_epochs, dtype=np.float64)
    phase = "train"
    epoch = 0
    next_batch_index = 0
    confusion = None
    elapsed_before = 0.0
    epoch_elapsed_before = 0.0

    if args.resume and checkpoint_exists:
        checkpoint = load_torch(checkpoint_path, map_location=device)
        if checkpoint.get("signature") != _resume_signature(args, dataset_name):
            raise RuntimeError(
                f"checkpoint configuration mismatch: {checkpoint_path}"
            )
        model.load_state_dict(checkpoint["model"])
        weight_method.load_state_dict(checkpoint.get("weight_method", {}))
        optimizer.load_state_dict(checkpoint["optimizer"])
        avg_cost = np.asarray(checkpoint["avg_cost"], dtype=np.float32)
        deltas = np.asarray(checkpoint["delta_m"], dtype=np.float32)
        epoch_times = np.asarray(
            checkpoint.get("epoch_times_seconds", epoch_times),
            dtype=np.float64,
        )
        phase = checkpoint["phase"]
        epoch = int(checkpoint["epoch"])
        next_batch_index = int(checkpoint["next_batch_index"])
        confusion = _restore_confusion(
            components,
            model,
            checkpoint.get("confusion"),
            device,
        )
        elapsed_before = float(checkpoint.get("wall_time_seconds", 0.0))
        epoch_elapsed_before = float(
            checkpoint.get("epoch_time_seconds", 0.0)
        )
        restore_rng_state(checkpoint.get("rng_state"))
        denominator = train_batches if phase == "train" else test_batches
        print(
            f"RESUME epoch={epoch + 1} phase={phase} "
            f"next_batch={next_batch_index}/{denominator}",
            flush=True,
        )

    if phase == "complete":
        return result_path

    segment_start = time.perf_counter()

    def total_elapsed():
        return elapsed_before + time.perf_counter() - segment_start

    def epoch_elapsed():
        return epoch_elapsed_before + time.perf_counter() - segment_start

    def save_checkpoint(current_phase, current_epoch, next_batch, current_confusion):
        payload = _checkpoint_payload(
            args,
            dataset_name,
            current_phase,
            current_epoch,
            next_batch,
            avg_cost,
            deltas,
            epoch_times,
            current_confusion,
            model,
            optimizer,
            weight_method,
            total_elapsed(),
            epoch_elapsed(),
        )
        atomic_torch_save(payload, checkpoint_path)
    
    if (
        args.method == "mgda_warm"
        and not weight_method.method.initialized
    ):
        print(
            "MGDA-WARM computing full-gradient warm start",
            flush=True,
        )

        full_gram = _mgda_warm_full_gram(
            model=model,
            train_set=train_set,
            dataset_name=dataset_name,
            device=device,
            batch_size=args.batch_size,
        )

        weight_method.method.warm_start(full_gram)

        save_checkpoint(
            "train",
            epoch,
            next_batch_index,
            confusion,
        )

        print(
            "MGDA-WARM warm start complete and checkpointed",
            flush=True,
        )

    while epoch < args.n_epochs:
        if phase == "train":
            model.train()
            _set_epoch_learning_rates(optimizer, args, epoch)
            if hasattr(weight_method.method, "set_outer_lr"):
                weight_method.method.set_outer_lr(
                    optimizer.param_groups[0]["lr"]
                )
            if confusion is None:
                confusion = components["confusion"](model.segnet.class_nb)

            streams = 3 if args.method in {"modo", "mgda_warm"} else 1
            batch_schedules = [
                _epoch_batch_indices(
                    len(train_set),
                    args.batch_size,
                    args.seed,
                    epoch,
                    stream,
                )
                for stream in range(streams)
            ]

            for batch_index in range(next_batch_index, train_batches):
                optimizer.zero_grad()
                raw_batches = [
                    _load_index_batch(train_set, schedule[batch_index])
                    for schedule in batch_schedules
                ]

                if args.method in {"modo", "mgda_warm"}:
                    batches = [
                        _move_batch(raw, dataset_name, device)
                        for raw in raw_batches
                    ]
                    predictions = []
                    representations = []
                    loss_vectors = []
                    for batch in batches:
                        prediction, representation, losses = _forward_losses(
                            model,
                            batch,
                            dataset_name,
                        )
                        predictions.append(prediction)
                        representations.append(representation)
                        loss_vectors.append(losses)

                    weight_method.backward(
                        losses=loss_vectors,
                        shared_parameters=list(model.shared_parameters()),
                        task_specific_parameters=list(
                            model.task_specific_parameters()
                        ),
                        last_shared_parameters=list(
                            model.last_shared_parameters()
                        ),
                        representation=representations[-1],
                    )

                    if args.method == "modo":
                        prediction = predictions[-1]
                        batch = batches[-1]
                        losses = loss_vectors[-1]
                    else:
                        prediction = predictions[0]
                        batch = batches[0]
                        losses = loss_vectors[0]
                else:
                    batch = _move_batch(raw_batches[0], dataset_name, device)
                    prediction, representation, losses = _forward_losses(
                        model,
                        batch,
                        dataset_name,
                    )
                    backward_kwargs = {}
                    if args.method == "moon":
                        from methods.moon import previous_update_scores
                        delta = previous_update_scores(losses, list(model.parameters()), optimizer)
                        weight_method.method.update(delta)
                    if args.method in {"entropic_lmo_mgda", "moon"}:
                        backward_kwargs["model_parameters"] = list(model.parameters())
                    weight_method.backward(
                        losses=losses,
                        shared_parameters=list(model.shared_parameters()),
                        task_specific_parameters=list(
                            model.task_specific_parameters()
                        ),
                        last_shared_parameters=list(
                            model.last_shared_parameters()
                        ),
                        representation=representation,
                        **backward_kwargs,
                    )

                optimizer.step()

                if args.method == "famo":
                    with torch.no_grad():
                        new_losses = _updated_losses(model, batch, dataset_name)
                        weight_method.method.update(new_losses.detach())

                confusion.update(
                    prediction[0].argmax(1).flatten(),
                    batch[1].flatten(),
                )
                cost = np.zeros(components["metric_size"], dtype=np.float32)

                if dataset_name == "cityscapes":
                    _train_cost_city(
                        cost,
                        losses,
                        prediction,
                        batch,
                        components["depth_error"],
                    )
                    avg_cost[epoch, :6] += cost[:6] / train_batches
                else:
                    _train_cost_nyu(
                        cost,
                        losses,
                        prediction,
                        batch,
                        components["depth_error"],
                        components["normal_error"],
                    )
                    avg_cost[epoch, :12] += cost[:12] / train_batches

                loss_row = epoch * train_batches + batch_index
                loss_memmap[loss_row] = losses.detach().cpu().numpy()
                loss_memmap.flush()

                next_batch_index = batch_index + 1
                if (
                    next_batch_index % args.cache_every_batches == 0
                    or next_batch_index == train_batches
                ):
                    save_checkpoint(
                        "train",
                        epoch,
                        next_batch_index,
                        confusion,
                    )

                if (
                    next_batch_index == 1
                    or next_batch_index % args.print_every_batches == 0
                    or next_batch_index == train_batches
                ):
                    values = " ".join(
                        f"{name}={losses[index].item():.3f}"
                        for index, name in enumerate(components["tasks"])
                    )
                    print(
                        f"[{epoch + 1} {next_batch_index}/{train_batches}] {values}",
                        flush=True,
                    )

            avg_cost[epoch, 1:3] = confusion.get_metrics()
            phase = "evaluate"
            next_batch_index = 0
            confusion = components["confusion"](model.segnet.class_nb)
            save_checkpoint("evaluate", epoch, 0, confusion)

        if phase == "evaluate":
            model.eval()
            if confusion is None:
                confusion = components["confusion"](model.segnet.class_nb)

            with torch.no_grad():
                for batch_index in range(next_batch_index, test_batches):
                    raw_batch = _load_index_batch(
                        test_set,
                        test_schedule[batch_index],
                    )
                    _evaluation_batch(
                        model,
                        raw_batch,
                        dataset_name,
                        device,
                        components,
                        avg_cost,
                        epoch,
                        confusion,
                        test_batches,
                    )
                    next_batch_index = batch_index + 1
                    if (
                        next_batch_index % args.cache_every_batches == 0
                        or next_batch_index == test_batches
                    ):
                        save_checkpoint(
                            "evaluate",
                            epoch,
                            next_batch_index,
                            confusion,
                        )

            deltas[epoch] = _finalize_evaluation(
                dataset_name,
                components,
                avg_cost,
                epoch,
                confusion,
            )
            epoch_times[epoch] = epoch_elapsed()
            phase = "commit"
            save_checkpoint("commit", epoch, test_batches, confusion)

        if phase == "commit":
            current_wall = total_elapsed()
            completed = epoch + 1 == args.n_epochs
            loss_count = (epoch + 1) * train_batches
            losses = np.asarray(loss_memmap[:loss_count]).tolist()
            _save_result(
                result_path,
                summary_path,
                avg_cost,
                deltas,
                epoch_times,
                current_wall,
                epoch,
                args,
                dataset_name,
                completed,
                losses,
            )
            _print_epoch(
                dataset_name,
                avg_cost,
                deltas,
                epoch,
                current_wall,
            )
            _print_loss_monitor(dataset_name, avg_cost, epoch)
            _update_loss_monitor(
                args,
                dataset_name,
                avg_cost,
                epoch,
                force_plot=completed,
            )

            if completed:
                phase = "complete"
                save_checkpoint("complete", epoch, test_batches, confusion)
                loss_memmap.flush()
                return result_path

            elapsed_before = current_wall
            epoch_elapsed_before = 0.0
            segment_start = time.perf_counter()
            epoch += 1
            next_batch_index = 0
            confusion = None
            phase = "train"
            save_checkpoint("train", epoch, 0, None)

    return result_path
