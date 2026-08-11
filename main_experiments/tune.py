from __future__ import annotations
import argparse, copy, itertools, sys
from pathlib import Path
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cache import CacheManager
from src.config import load_config, resolve_method_config
from src.metrics import delta_m
from src.report import save_table
from src.reported import ZHANG_REPORTED
from src.runner import _run_cached


def parse_args():
    p = argparse.ArgumentParser(description="Grid-search Algorithm 1/4 hyperparameters with fingerprinted caches.")
    p.add_argument("--method", required=True, help="One proposed method, e.g. Alg1-Spectral or Alg4-Mixed")
    p.add_argument("--dataset", required=True, choices=["cityscapes", "nyuv2"])
    p.add_argument("--config", default=str(ROOT / "main_experiments" / "configs" / "main_comparison.yaml"))
    p.add_argument("--output", default=None, help="Default: D:/LMO-MOO-runs/tuning/<dataset>/<method>")
    p.add_argument("--seed", type=int, default=None, help="Default: first configured seed")
    p.add_argument("--epochs", type=int, default=None, help="Override epochs for each trial")
    p.add_argument("--max-trials", type=int, default=0, help="0 means the complete Cartesian grid")
    p.add_argument("--include-no-momentum", action="store_true", help="Also test momentum_alpha=null")
    p.add_argument("--reset-cache", action="store_true")
    return p.parse_args()


def grid_for(config, method):
    if method not in config.get("methods", {}):
        raise KeyError(f"Unknown method {method!r}")
    base = config["methods"][method]
    kind = str(base.get("kind", "")).lower()
    if kind not in {"alg1", "alg4"}:
        raise ValueError("tune.py intentionally tunes only the proposed Algorithm 1/4 variants")
    s = config.get("search_space", {})
    gammas = s.get("gamma", [base.get("gamma", 5e-4)])
    steps = s.get("weight_step", [base.get("eta", base.get("beta", 0.5))])
    radii = s.get("radius", [base.get("radius", 1.0)])
    momenta = s.get("momentum_alpha", [base.get("momentum_alpha")])
    if kind == "alg1":
        for gamma, eta, radius, mom in itertools.product(gammas, steps, radii, momenta):
            yield {"gamma": gamma, "eta": eta, "radius": radius, "momentum_alpha": mom}
    else:
        rhos = s.get("rho", [base.get("rho", 0.5)])
        for gamma, beta, rho, radius, mom in itertools.product(gammas, steps, rhos, radii, momenta):
            yield {"gamma": gamma, "beta": beta, "rho": rho, "radius": radius, "momentum_alpha": mom}


def main():
    args = parse_args()
    config = load_config(Path(args.config))
    if args.method not in config.get("methods", {}):
        raise SystemExit(f"No config entry for {args.method!r}")
    out = Path(args.output or f"D:/LMO-MOO-runs/tuning/{args.dataset}/{args.method}")
    out.mkdir(parents=True, exist_ok=True)
    seed = args.seed
    if seed is None:
        seeds = config.get("experiment", {}).get("seeds") or [config.get("experiment", {}).get("seed", 2025)]
        seed = int(seeds[0])

    trials = list(grid_for(config, args.method))
    if args.include_no_momentum:
        extra = []
        for trial in trials:
            t = dict(trial); t["momentum_alpha"] = None; extra.append(t)
        trials.extend(extra)
        # Stable deduplication.
        trials = list({tuple(sorted(x.items())): x for x in trials}.values())
    if args.max_trials > 0:
        trials = trials[:args.max_trials]

    rows = []
    stl = ZHANG_REPORTED[args.dataset]["STL"]
    for i, hp in enumerate(trials, start=1):
        trial_name = f"{args.method}__trial_{i:04d}"
        cfg = copy.deepcopy(config)
        cfg["methods"][trial_name] = resolve_method_config(cfg, args.method, args.dataset)
        cfg["methods"][trial_name].update(hp)
        cache = CacheManager(out, "tuning", cfg)
        if args.reset_cache and i == 1:
            cache.clear_all_methods()
        state = _run_cached(cache, cfg, args.dataset, trial_name, seed, max_epochs=args.epochs, progress=True)
        final = dict(state.get("final", {}))
        comparable = {k: final[k] for k in stl if k in final}
        score = delta_m(comparable, stl)
        row = {"trial": i, "method": args.method, "trial_name": trial_name, "dataset": args.dataset,
               "seed": seed, **hp, **final, "delta_m": score}
        rows.append(row)
        print(f"[trial {i}/{len(trials)}] Δm%={score:.6g}  {hp}")

    frame = pd.DataFrame(rows).sort_values("delta_m", ascending=True, na_position="last")
    save_table(frame, out / "tables" / "trials")
    if frame.empty:
        return
    best = frame.iloc[0].to_dict()
    best_cfg = resolve_method_config(config, args.method, args.dataset)
    for key in ("gamma", "eta", "beta", "rho", "radius", "momentum_alpha"):
        if key in best and pd.notna(best[key]):
            value = best[key]
            best_cfg[key] = value.item() if hasattr(value, "item") else value
    with (out / "best_method.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump({args.method: best_cfg}, f, sort_keys=False)
    print(f"Best trial: {int(best['trial'])}; Δm%={best['delta_m']:.6g}")
    print(f"Saved: {out / 'best_method.yaml'}")


if __name__ == "__main__":
    main()
