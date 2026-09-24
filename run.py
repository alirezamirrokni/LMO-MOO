import argparse
import os
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np

from experiments.checkpointing import load_torch
from experiments.trainer import train


METHODS = [
    "stl",
    "ls",
    "scaleinvls",
    "rlw",
    "dwa",
    "uw",
    "mgda",
    "pcgrad",
    "graddrop",
    "cagrad",
    "moco",
    "modo",
    "nashmtl",
    "famo",
    "mgda_warm",
    "entropic_lmo_mgda",
    "moon",
]

ALIASES = {
    "entropic-lmo-mgda": "entropic_lmo_mgda",
    "entropic lmo-mgda": "entropic_lmo_mgda",
    "ours": "entropic_lmo_mgda",
    "paper_lmo_moo": "entropic_lmo_mgda",
    "si": "scaleinvls",
    "nash-mtl": "nashmtl",
    "mgda-warm": "mgda_warm",
    "paper-lmo-moo": "entropic_lmo_mgda",
    "moon": "moon",
}

TASK_COUNTS = {"cityscapes": 2, "nyuv2": 3}


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
        return len(value)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return False


def default_data_root():
    environment = os.environ.get("LMO_MOO_DATA_ROOT")
    if environment:
        return Path(environment)
    scratch = Path("/mloscratch/homes/mirrokni/datasets")
    if scratch.exists():
        return scratch
    return Path(__file__).resolve().parent / "datasets"


def completed(path):
    summary_path = (
        path.parents[1]
        / "summaries"
        / path.parent.name
        / f"{path.stem}.summary"
    )
    source = summary_path if summary_path.exists() else path
    if not source.exists():
        return False
    try:
        data = load_torch(source, map_location="cpu")
    except Exception:
        return False
    if "completed" in data:
        return bool(data["completed"])
    values = np.asarray(data.get("avg_cost", []))
    return values.ndim == 2 and len(values) > 0 and not np.allclose(values[-1], 0)


def parser():
    p = argparse.ArgumentParser(
        description="Run the LMO-MOO multi-task baseline benchmark."
    )
    p.add_argument("--dataset", choices=["cityscapes", "nyuv2"], required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-root", type=Path, default=default_data_root())
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--model", choices=["segnet", "mtan"], default="mtan")
    p.add_argument("--no-augmentation", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--method-params-lr", type=float, default=None)
    p.add_argument("--max-norm", type=float, default=1.0)
    p.add_argument("--c", type=float, default=0.4)
    p.add_argument("--dwa-temp", type=float, default=2.0)
    p.add_argument("--nashmtl-optim-niter", type=int, default=20)
    p.add_argument("--update-weights-every", type=int, default=1)
    p.add_argument("--famo-gamma", type=float, default=None)
    p.add_argument("--moco-beta", type=float, default=0.5)
    p.add_argument("--moco-beta-sigma", type=float, default=0.5)
    p.add_argument("--moco-gamma", type=float, default=0.1)
    p.add_argument("--moco-gamma-sigma", type=float, default=0.5)
    p.add_argument("--moco-rho", type=float, default=0.0)
    p.add_argument("--modo-gamma", type=float, default=1e-3)
    p.add_argument("--modo-rho", type=float, default=0.1)
    p.add_argument("--mgda-warm-beta", type=float, default=0.5)
    p.add_argument("--mgda-warm-rho", type=float, default=0.5)
    p.add_argument("--mgda-warm-steps", type=int, default=40)
    p.add_argument("--mgda-warm-beta0", type=float, default=None)
    p.add_argument("--lmo-alpha", type=float, default=0.5)
    p.add_argument("--muon-ns-steps", type=int, default=5)
    p.add_argument("--entropy-tau", type=float, default=None)
    p.add_argument("--weight-kappa", type=float, default=None)
    p.add_argument(
        "--moon-logit-decay",
        type=float,
        default=None,
        help="MOON logit weight-decay coefficient (paper notation gamma).",
    )
    p.add_argument(
        "--lmo-lr-schedule",
        choices=["legacy", "late150"],
        default="legacy",
        help="Outer LMO parameter-step schedule; late150 halves again at epoch 150.",
    )
    p.add_argument("--cache-every-batches", type=int, default=1)
    p.add_argument("--print-every-batches", type=int, default=25)
    return p


def normalize_method(method):
    method = ALIASES.get(method.lower(), method.lower())
    if method not in METHODS:
        raise ValueError(
            f"unsupported method {method!r}; choose one of {', '.join(METHODS)}"
        )
    return method


def build_train_args(cli, method, main_task, result_name):
    if cli.lr is not None:
        learning_rate = cli.lr
    else:
        learning_rate = 1e-4

    famo_gamma = cli.famo_gamma
    if famo_gamma is None:
        famo_gamma = 0.01 if cli.dataset == "cityscapes" else 0.001

    batch_size = 8 if cli.dataset == "cityscapes" else 2

    return argparse.Namespace(
        data_path=cli.data_root / cli.dataset,
        output_dir=cli.output_root / cli.dataset,
        checkpoint_dir=cli.output_root / "checkpoints" / cli.dataset,
        summary_dir=cli.output_root / "summaries" / cli.dataset,
        result_name=result_name,
        n_epochs=cli.epochs,
        batch_size=batch_size,
        method=method,
        lr=learning_rate,
        method_params_lr=cli.method_params_lr,
        gpu=cli.gpu,
        seed=cli.seed,
        main_task=main_task,
        c=cli.c,
        gamma=famo_gamma,
        max_norm=cli.max_norm,
        dwa_temp=cli.dwa_temp,
        nashmtl_optim_niter=cli.nashmtl_optim_niter,
        update_weights_every=cli.update_weights_every,
        moco_beta=cli.moco_beta,
        moco_beta_sigma=cli.moco_beta_sigma,
        moco_gamma=cli.moco_gamma,
        moco_gamma_sigma=cli.moco_gamma_sigma,
        moco_rho=cli.moco_rho,
        modo_gamma=cli.modo_gamma,
        modo_rho=cli.modo_rho,
        mgda_warm_beta=cli.mgda_warm_beta,
        mgda_warm_rho=cli.mgda_warm_rho,
        mgda_warm_steps=cli.mgda_warm_steps,
        mgda_warm_beta0=cli.mgda_warm_beta0,
        model=cli.model,
        apply_augmentation=not cli.no_augmentation,
        cache_every_batches=cli.cache_every_batches,
        print_every_batches=cli.print_every_batches,
        resume=True,
        force=cli.force,
        lmo_alpha=cli.lmo_alpha,
        muon_ns_steps=cli.muon_ns_steps,
        entropy_tau=cli.entropy_tau,
        weight_kappa=cli.weight_kappa,
        lmo_lr_schedule=cli.lmo_lr_schedule,
        moon_logit_decay=cli.moon_logit_decay,
)


def _remove_if_exists(path):
    if path.exists():
        path.unlink()


def run_one(cli, method, main_task=0, result_name=None):
    if result_name is None:
        versioned = {"moon": "moon_reference", "entropic_lmo_mgda": "paper_lmo_moo"}.get(method, method)
        result_name = f"{versioned}_sd{cli.seed}"

    result_path = cli.output_root / cli.dataset / f"{result_name}.stats"
    checkpoint_path = (
        cli.output_root / "checkpoints" / cli.dataset / f"{result_name}.pt"
    )
    summary_path = (
        cli.output_root / "summaries" / cli.dataset / f"{result_name}.summary"
    )
    loss_path = (
        cli.output_root
        / "checkpoints"
        / cli.dataset
        / f"{result_name}.losses.dat"
    )
    log_path = cli.output_root / "logs" / cli.dataset / f"{result_name}.log"

    if completed(result_path) and not cli.force:
        print(f"SKIP completed result: {result_path}")
        return result_path

    if not (cli.data_root / cli.dataset).exists():
        raise FileNotFoundError(f"dataset not found: {cli.data_root / cli.dataset}")

    if cli.cache_every_batches < 1:
        raise ValueError("--cache-every-batches must be at least 1")
    if cli.method_params_lr is None:
        cli.method_params_lr = (1e-5 if cli.dataset == "nyuv2" else 1e-4) if method == "moon" else (1e-4 if method == "entropic_lmo_mgda" else 0.025)
    if cli.moon_logit_decay is None:
        cli.moon_logit_decay = 0.05 if cli.dataset == "cityscapes" else 1e-3
    if cli.entropy_tau is None:
        cli.entropy_tau = 0.0 if method == "entropic_lmo_mgda" else 0.015
    if cli.weight_kappa is None:
        cli.weight_kappa = 0.0 if method == "entropic_lmo_mgda" else 0.05
    if cli.lr is None and method == "moon":
        cli.lr = 1e-4 if cli.dataset == "cityscapes" else 1e-3
    if cli.print_every_batches < 1:
        raise ValueError("--print-every-batches must be at least 1")



    if method in {"entropic_lmo_mgda", "moon"}:
        if not np.isfinite(cli.method_params_lr) or cli.method_params_lr <= 0.0:
            raise ValueError("--method-params-lr must be finite and positive")
        if not np.isfinite(cli.lmo_alpha) or not (0.0 < cli.lmo_alpha <= 1.0):
            raise ValueError("--lmo-alpha must be finite and in (0, 1]")
        if cli.muon_ns_steps < 1:
            raise ValueError("--muon-ns-steps must be at least 1")
        if method == "entropic_lmo_mgda":
            if not np.isfinite(cli.entropy_tau) or cli.entropy_tau < 0.0:
                raise ValueError("--entropy-tau must be finite and non-negative")
            if not np.isfinite(cli.weight_kappa) or cli.weight_kappa < 0.0:
                raise ValueError("--weight-kappa must be finite and non-negative")
            if TASK_COUNTS[cli.dataset] * cli.weight_kappa >= 1.0:
                raise ValueError("--weight-kappa must satisfy K * kappa < 1")
        if method == "moon":
            if not np.isfinite(cli.moon_logit_decay) or cli.moon_logit_decay < 0.0:
                raise ValueError("--moon-logit-decay must be finite and non-negative")

    if cli.force:
        _remove_if_exists(result_path)
        _remove_if_exists(summary_path)
        _remove_if_exists(checkpoint_path)
        _remove_if_exists(loss_path)
        _remove_if_exists(log_path)

    args = build_train_args(cli, method, main_task, result_name)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if checkpoint_path.exists() else "w"

    with log_path.open(mode, encoding="utf-8", buffering=1) as log_file:
        stdout = Tee(sys.stdout, log_file)
        stderr = Tee(sys.stderr, log_file)
        with redirect_stdout(stdout), redirect_stderr(stderr):
            print(
                f"RUN dataset={cli.dataset} method={method} seed={cli.seed} "
                f"lr={args.lr} batch={args.batch_size} epochs={args.n_epochs} "
                f"model={args.model}",
                flush=True,
            )
            print(f"RESULT {result_path}", flush=True)
            print(f"CHECKPOINT {checkpoint_path}", flush=True)
            return train(args, cli.dataset)


def main():
    cli = parser().parse_args()
    method = normalize_method(cli.method)

    if method == "stl":
        for task in range(TASK_COUNTS[cli.dataset]):
            run_one(
                cli,
                method,
                main_task=task,
                result_name=f"stl_task{task}_sd{cli.seed}",
            )
    else:
        run_one(cli, method)


if __name__ == "__main__":
    main()
