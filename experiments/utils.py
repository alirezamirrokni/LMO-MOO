import argparse
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from methods import METHODS


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"yes", "true", "t", "y", "1"}:
        return True
    if value in {"no", "false", "f", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


common_parser = argparse.ArgumentParser(add_help=False)
common_parser.add_argument("--data-path", type=Path, required=True)
common_parser.add_argument("--output-dir", type=Path, required=True)
common_parser.add_argument("--result-name", type=str, required=True)
common_parser.add_argument("--n-epochs", type=int, default=200)
common_parser.add_argument("--batch-size", type=int, required=True)
common_parser.add_argument("--method", choices=sorted(METHODS), required=True)
common_parser.add_argument("--lr", type=float, required=True)
common_parser.add_argument("--method-params-lr", type=float, default=0.025)
common_parser.add_argument("--gpu", type=int, default=0)
common_parser.add_argument("--seed", type=int, default=42)
common_parser.add_argument("--main-task", type=int, default=0)
common_parser.add_argument("--c", type=float, default=0.4)
common_parser.add_argument("--gamma", type=float, default=0.01)
common_parser.add_argument("--max-norm", type=float, default=1.0)
common_parser.add_argument("--dwa-temp", type=float, default=2.0)
common_parser.add_argument("--nashmtl-optim-niter", type=int, default=20)
common_parser.add_argument("--update-weights-every", type=int, default=1)
common_parser.add_argument("--moco-beta", type=float, default=0.5)
common_parser.add_argument("--moco-beta-sigma", type=float, default=0.5)
common_parser.add_argument("--moco-gamma", type=float, default=0.1)
common_parser.add_argument("--moco-gamma-sigma", type=float, default=0.5)
common_parser.add_argument("--moco-rho", type=float, default=0.0)
common_parser.add_argument("--modo-gamma", type=float, default=1e-3)
common_parser.add_argument("--modo-rho", type=float, default=0.1)
common_parser.add_argument("--mgda-warm-beta", type=float, default=0.5)
common_parser.add_argument("--mgda-warm-rho", type=float, default=0.5)
common_parser.add_argument("--mgda-warm-steps", type=int, default=40)
common_parser.add_argument("--mgda-warm-beta0", type=float, default=None)
common_parser.add_argument("--lmo-alpha", type=float, default=0.5)
common_parser.add_argument("--muon-ns-steps", type=int, default=5)
common_parser.add_argument("--entropy-tau", type=float, default=0.015)
common_parser.add_argument("--weight-kappa", type=float, default=0.05)


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device(gpu=0):
    return torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")


def method_parameters(args):
    parameters = defaultdict(dict)
    parameters.update(
        {
            "nashmtl": {
                "update_weights_every": args.update_weights_every,
                "optim_niter": args.nashmtl_optim_niter,
                "max_norm": args.max_norm,
            },
            "stl": {"main_task": args.main_task},
            "dwa": {"temp": args.dwa_temp},
            "cagrad": {"c": args.c, "max_norm": args.max_norm},
            "famo": {
                "gamma": args.gamma,
                "w_lr": args.method_params_lr,
                "max_norm": args.max_norm,
            },
            "moco": {
                "beta": args.moco_beta,
                "beta_sigma": args.moco_beta_sigma,
                "gamma": args.moco_gamma,
                "gamma_sigma": args.moco_gamma_sigma,
                "rho": args.moco_rho,
            },
            "modo": {
                "gamma": args.modo_gamma,
                "rho": args.modo_rho,
            },
            "mgda_warm": {
                "beta": args.mgda_warm_beta,
                "rho": args.mgda_warm_rho,
                "warm_steps": args.mgda_warm_steps,
                "warm_beta": args.mgda_warm_beta0,
            },
        }
    )
    return parameters[args.method]
