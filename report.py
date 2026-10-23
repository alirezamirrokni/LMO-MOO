import argparse
import json
from pathlib import Path

import numpy as np
import torch

DENSE_METHODS = [
    ("LS", "ls"), ("SI", "scaleinvls"), ("RLW", "rlw"), ("DWA", "dwa"), ("UW", "uw"), ("MGDA", "mgda"),
    ("PCGrad", "pcgrad"), ("GradDrop", "graddrop"), ("CAGrad", "cagrad"), ("MoCo", "moco"), ("MoDo", "modo"),
    ("Nash-MTL", "nashmtl"), ("FAMO", "famo"), ("MGDA (warm)", "mgda_warm"), ("MOON", "moon"),
    ("Entropic LMO-MGDA", "entropic_lmo_mgda"),
]
DENSE = {
    "cityscapes": dict(columns=["mIoU", "Pix Acc", "Abs Err", "Rel Err"], index=[7, 8, 10, 11], percent=[1, 1, 0, 0],
                       higher=[1, 1, 0, 0], stl_task=[0, 0, 1, 1]),
    "nyuv2": dict(columns=["mIoU", "Pix Acc", "Abs", "Rel", "Mean", "Med.", "<11.25", "<22.5", "<30"],
                  index=[13, 14, 16, 17, 19, 20, 21, 22, 23], percent=[1, 1, 0, 0, 0, 0, 1, 1, 1],
                  higher=[1, 1, 0, 0, 0, 0, 1, 1, 1], stl_task=[0, 0, 1, 1, 2, 2, 2, 2, 2]),
}
MULTIMNIST_METHODS = [("MGDA", "mgda"), ("MGDA + Muon", "mgda_muon"), ("FAMO", "famo"), ("FAMO + Muon", "famo_muon"),
                      ("Muon (LS)", "muon_ls"), ("MOON", "moon"), ("Entropic LMO-MGDA", "entropic_lmo_mgda")]


def print_table(title, header, rows):
    rows = [[r[0]] + ["--" if v is None else str(v) if isinstance(v, int) else f"{v:.4f}" for v in r[1:]] for r in rows]
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    print(f"\n{title}\n")
    for line in [header, ["-" * w for w in widths], *rows]:
        print("  ".join(str(x).ljust(w) for x, w in zip(line, widths)))


def final_metrics(path, spec):
    if not path.exists():
        return None
    result = torch.load(path, weights_only=False)
    if not result["completed"]:
        return None
    values = np.asarray(result["avg_cost"], dtype=float)[-10:].mean(axis=0)[spec["index"]]
    return values * np.where(spec["percent"], 100.0, 1.0)


def dense_table(root, dataset, seed):
    spec = DENSE[dataset]
    stl_tasks = [final_metrics(root / dataset / f"stl_task{t}_sd{seed}.stats", spec) for t in range(max(spec["stl_task"]) + 1)]
    stl = None if any(x is None for x in stl_tasks) else np.array([stl_tasks[t][i] for i, t in enumerate(spec["stl_task"])])
    rows = [["STL", *(stl if stl is not None else [None] * len(spec["index"])), 0.0 if stl is not None else None]]
    sign = np.where(spec["higher"], -1.0, 1.0)
    for label, name in DENSE_METHODS:
        metrics = final_metrics(root / dataset / f"{name}_sd{seed}.stats", spec)
        if metrics is None:
            rows.append([label, *[None] * (len(spec["index"]) + 1)])
            continue
        delta = None if stl is None else float(np.mean(sign * (metrics - stl) / stl) * 100)
        rows.append([label, *metrics, delta])
    print_table(dataset, ["Method", *spec["columns"], "Delta_m%"], rows)


def multimnist_runs(root, tag, method):
    runs = [json.loads(p.read_text()) for p in sorted((root / tag / method).glob("seed*/summary.json"))]
    return [r for r in runs if r["completed"]]


def multimnist_tables(root):
    rows = []
    for label, method in MULTIMNIST_METHODS:
        runs = multimnist_runs(root, "main", method)
        rows.append([label, *([float(np.mean([r[k] for r in runs])) for k in ("left", "right", "avg")] if runs else [None] * 3)])
    print_table("MultiMNIST", ["Method", "Left", "Right", "Avg."], rows)
    rows = []
    for folder in sorted(p for p in root.iterdir() if (p / "entropic_lmo_mgda").is_dir() and not p.name.startswith("validation")):
        runs = multimnist_runs(root, folder.name, "entropic_lmo_mgda")
        if runs:
            rows.append([folder.name, float(np.mean([r["avg"] for r in runs])), float(np.mean([r["gap"] for r in runs])), len(runs)])
    print_table("MultiMNIST ablations (Entropic LMO-MGDA)", ["Variant", "Avg.", "Gap", "Seeds"], rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    for dataset in DENSE:
        if (args.results / dataset).exists():
            dense_table(args.results, dataset, args.seed)
    if (args.results / "multimnist").exists():
        multimnist_tables(args.results / "multimnist")


if __name__ == "__main__":
    main()
