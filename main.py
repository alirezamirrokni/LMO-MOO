import argparse
from pathlib import Path

import torch

from experiments import dense, multimnist
from methods import METHODS

MULTIMNIST_METHODS = ["entropic_lmo_mgda", "mgda", "mgda_muon", "famo", "famo_muon", "muon_ls", "moon"]
PAPER_SETTINGS = {
    ("cityscapes", "entropic_lmo_mgda"): dict(lr=5e-3, eta=1e-4, schedule="late150"),
    ("nyuv2", "entropic_lmo_mgda"): dict(lr=2.5e-3, eta=1e-4),
    ("cityscapes", "moon"): dict(lr=1e-4, eta=1e-4),
    ("nyuv2", "moon"): dict(lr=1e-3, eta=1e-5),
}
ABLATION_DEFAULTS = dict(oracle="spectral", weights="entropic", momentum="blended", eta=5e-5, kappa=0.0)


def run_dense(args):
    settings = dict(lr=1e-4, eta=0.025, schedule="step100") | PAPER_SETTINGS.get((args.dataset, args.method), {})
    for seed in args.seeds or [42]:
        for task in range(dense.N_TASKS[args.dataset]) if args.method == "stl" else [0]:
            cfg = argparse.Namespace(
                dataset=args.dataset, method=args.method, seed=seed, epochs=args.epochs or 200,
                lr=args.lr or settings["lr"], method_lr=args.eta or settings["eta"], schedule=settings["schedule"],
                alpha=args.alpha, tau=args.tau, kappa=args.kappa, main_task=task,
                famo_gamma=0.01 if args.dataset == "cityscapes" else 0.001,
                logit_decay=0.05 if args.dataset == "cityscapes" else 1e-3,
                data_path=str(args.data_root / args.dataset), gpu=args.gpu,
            )
            name = (f"stl_task{task}" if args.method == "stl" else args.method) + f"_sd{seed}"
            result = args.output_root / args.dataset / f"{name}.stats"
            checkpoint = args.output_root / "checkpoints" / args.dataset / f"{name}.pt"
            if args.force:
                result.unlink(missing_ok=True)
                checkpoint.unlink(missing_ok=True)
            if result.exists() and torch.load(result, weights_only=False)["completed"]:
                print(f"skipping completed run {result}")
                continue
            print(f"running {name} on {args.dataset}: lr={cfg.lr} eta={cfg.method_lr}", flush=True)
            dense.train(cfg, result, checkpoint)


def run_multimnist(args):
    args.eta = args.eta or ABLATION_DEFAULTS["eta"]
    changed = [f"{k}-{getattr(args, k):g}" if isinstance(v, float) else f"{k}-{getattr(args, k)}"
               for k, v in ABLATION_DEFAULTS.items() if getattr(args, k) != v]
    tag = ("validation-" if args.validation else "") + ("_".join(changed) or "main")
    for seed in args.seeds or [42, 43, 44]:
        cfg = argparse.Namespace(
            method=args.method, seed=seed, epochs=args.epochs or 100, batch_size=args.batch_size, lr=args.lr or 1e-3,
            alpha=args.alpha, eta=args.eta, kappa=args.kappa, oracle=args.oracle, weights=args.weights,
            momentum=args.momentum, validation=args.validation, data_path=args.data_root / "multimnist",
            device=f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu",
        )
        out = args.output_root / "multimnist" / tag / args.method / f"seed{seed}"
        print(f"running {args.method} seed={seed} -> {out}", flush=True)
        multimnist.train(cfg, out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=[*dense.N_TASKS, "multimnist"], required=True)
    p.add_argument("--method", choices=sorted(set(METHODS) | set(MULTIMNIST_METHODS)), required=True)
    p.add_argument("--seeds", type=int, nargs="+")
    p.add_argument("--epochs", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--eta", type=float)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--tau", type=float, default=0.0)
    p.add_argument("--kappa", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--oracle", choices=["spectral", "l2", "sign"], default="spectral")
    p.add_argument("--weights", choices=["entropic", "projected"], default="entropic")
    p.add_argument("--momentum", choices=["blended", "per-task", "none"], default="blended")
    p.add_argument("--validation", action="store_true")
    p.add_argument("--data-root", type=Path, default=Path("datasets"))
    p.add_argument("--output-root", type=Path, default=Path("results"))
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    if args.dataset == "multimnist":
        if args.method not in MULTIMNIST_METHODS:
            p.error(f"--method must be one of {MULTIMNIST_METHODS} for multimnist")
        run_multimnist(args)
    else:
        if args.method not in METHODS:
            p.error(f"--method {args.method} is not available for dense datasets")
        run_dense(args)


if __name__ == "__main__":
    main()
