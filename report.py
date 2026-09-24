import argparse
from pathlib import Path

import importlib
import sys

import numpy as np
import torch

if int(np.__version__.split(".")[0]) < 2:
    for _suffix in (
        "",
        ".multiarray",
        ".numeric",
        "._multiarray_umath",
        ".umath",
        ".numerictypes",
        ".fromnumeric",
        ".shape_base",
        ".overrides",
        "._asarray",
        ".getlimits",
        "._dtype",
        "._internal",
    ):
        try:
            _module = importlib.import_module(f"numpy.core{_suffix}")
        except ImportError:
            continue
        sys.modules.setdefault(f"numpy._core{_suffix}", _module)


ORDER = [
    ("Entropic LMO-MGDA", "paper_lmo_moo"),
    ("MOON (official)", "moon_reference"),
    ("STL", "stl"),
    ("LS", "ls"),
    ("SI", "scaleinvls"),
    ("RLW", "rlw"),
    ("DWA", "dwa"),
    ("UW", "uw"),
    ("MGDA", "mgda"),
    ("PCGrad", "pcgrad"),
    ("GradDrop", "graddrop"),
    ("CAGrad", "cagrad"),
    ("MoCo", "moco"),
    ("MoDo", "modo"),
    ("Nash-MTL", "nashmtl"),
    ("FAMO", "famo"),
    ("MGDA-warm", "mgda_warm"),
]


def load_completed(path):
    summary_path = (
        path.parents[1]
        / "summaries"
        / path.parent.name
        / f"{path.stem}.summary"
    )
    source = summary_path if summary_path.exists() else path
    if not source.exists():
        return None

    try:
        try:
            data = torch.load(source, map_location="cpu", weights_only=False)
        except TypeError:
            data = torch.load(source, map_location="cpu")
    except Exception:
        return None

    values = np.asarray(data.get("avg_cost", []), dtype=float)
    if values.ndim != 2 or len(values) < 10:
        return None

    if "completed" in data:
        if not bool(data["completed"]):
            return None
    elif np.allclose(values[-1], 0):
        return None

    return data


def last_ten(data):
    return np.asarray(data["avg_cost"], dtype=float)[-10:].mean(axis=0)


def delta_last_ten(data):
    values = np.asarray(data.get("delta_m", []), dtype=float)
    if len(values) < 10 or not np.all(np.isfinite(values[-10:])):
        return None
    return float(values[-10:].mean())


def wall_hours(data):
    value = data.get("wall_time_seconds")
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value) or value < 0:
        return None
    return value / 3600.0


def city_row(data):
    values = last_ten(data)
    return [
        100 * values[7],
        100 * values[8],
        values[10],
        values[11],
        delta_last_ten(data),
        wall_hours(data),
    ]


def nyu_row(data):
    values = last_ten(data)
    return [
        100 * values[13],
        100 * values[14],
        values[16],
        values[17],
        values[19],
        values[20],
        100 * values[21],
        100 * values[22],
        100 * values[23],
        delta_last_ten(data),
        wall_hours(data),
    ]


def combined_wall_hours(data):
    values = [wall_hours(item) for item in data]
    if any(value is None for value in values):
        return None
    return float(sum(values))


def city_stl(result_dir, seed):
    paths = [
        result_dir / f"stl_task0_sd{seed}.stats",
        result_dir / f"stl_task1_sd{seed}.stats",
    ]
    data = [load_completed(path) for path in paths]
    if any(item is None for item in data):
        return None

    segmentation = last_ten(data[0])
    depth = last_ten(data[1])

    from experiments.cityscapes.utils import delta_fn

    metrics = np.array(
        [segmentation[7], segmentation[8], depth[10], depth[11]]
    )
    return [
        100 * segmentation[7],
        100 * segmentation[8],
        depth[10],
        depth[11],
        float(delta_fn(metrics)),
        combined_wall_hours(data),
    ]


def nyu_stl(result_dir, seed):
    paths = [
        result_dir / f"stl_task0_sd{seed}.stats",
        result_dir / f"stl_task1_sd{seed}.stats",
        result_dir / f"stl_task2_sd{seed}.stats",
    ]
    data = [load_completed(path) for path in paths]
    if any(item is None for item in data):
        return None

    segmentation = last_ten(data[0])
    depth = last_ten(data[1])
    normal = last_ten(data[2])

    from experiments.nyuv2.utils import delta_fn

    metrics = np.array(
        [
            segmentation[13],
            segmentation[14],
            depth[16],
            depth[17],
            normal[19],
            normal[20],
            normal[21],
            normal[22],
            normal[23],
        ]
    )
    return [
        100 * segmentation[13],
        100 * segmentation[14],
        depth[16],
        depth[17],
        normal[19],
        normal[20],
        100 * normal[21],
        100 * normal[22],
        100 * normal[23],
        float(delta_fn(metrics)),
        combined_wall_hours(data),
    ]


def format_value(value):
    if value is None:
        return "NA"
    if isinstance(value, str):
        return value
    if not np.isfinite(value):
        return "NA"
    return f"{value:.4f}"


def print_table(headers, rows):
    text_rows = [[format_value(value) for value in row] for row in rows]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in text_rows))
        for index in range(len(headers))
    ]

    print(
        "  ".join(
            headers[index].ljust(widths[index])
            for index in range(len(headers))
        )
    )
    print("  ".join("-" * width for width in widths))

    for row in text_rows:
        print(
            "  ".join(
                row[index].ljust(widths[index])
                for index in range(len(headers))
            )
        )


def collect(root, dataset, seed):
    result_dir = root / dataset
    width = 6 if dataset == "cityscapes" else 11
    rows = []

    for display, code in ORDER:
        if code == "stl":
            metrics = (
                city_stl(result_dir, seed)
                if dataset == "cityscapes"
                else nyu_stl(result_dir, seed)
            )
        else:
            path = result_dir / f"{code}_sd{seed}.stats"
            data = load_completed(path)
            if data is None:
                metrics = None
            else:
                metrics = city_row(data) if dataset == "cityscapes" else nyu_row(data)

        if metrics is None:
            metrics = [None] * width

        rows.append([display] + metrics)

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Print run-derived results. Missing or incomplete runs are NA."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    args = parser.parse_args()

    print("\n====================== CITYSCAPES ======================")
    print("Mean over the final 10 epochs of completed runs only\n")
    print_table(
        [
            "Method",
            "mIoU (%)",
            "Pix Acc (%)",
            "Abs Err",
            "Rel Err",
            "Delta_m (%)",
            "Wall h",
        ],
        collect(args.results, "cityscapes", args.seed),
    )

    print("\n======================== NYUv2 =========================")
    print("Mean over the final 10 epochs of completed runs only\n")
    print_table(
        [
            "Method",
            "mIoU (%)",
            "Pix Acc (%)",
            "Abs Err",
            "Rel Err",
            "Normal Mean",
            "Normal Median",
            "<11.25 (%)",
            "<22.5 (%)",
            "<30 (%)",
            "Delta_m (%)",
            "Wall h",
        ],
        collect(args.results, "nyuv2", args.seed),
    )

    print("\nNA = no completed run-derived result is available.\n")


if __name__ == "__main__":
    main()
