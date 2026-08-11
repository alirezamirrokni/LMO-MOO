# LMO-MOO

PyTorch codebase for the experiments in **Norm-Constrained LMO Methods for Multi-Objective Optimization**.

The repository intentionally follows the run/config/cache style of the supplied bilevel-optimization codebase: YAML-first experiments, `--problem` dispatch, per-method/per-dataset/per-seed fingerprinted caches, resumable checkpoints, `--plot-only`, selective method runs, and Windows-friendly output paths.

## What is implemented

The two proposed algorithms requested for this project are implemented exactly as separate controllers:

- **Algorithm 1 — Entropic LMO-MGDA** (`src/methods/alg1.py`)
  - one LMO call per optimization step;
  - blended gradient `G = sum_i w_i grad f_i`;
  - score `g_i = <D, grad f_i>`;
  - **positive-sign** entropic mirror step `w_i <- w_i exp(eta g_i)` followed by normalization;
  - parameter step `x <- x + gamma D`.
- **Algorithm 4 — LMO-MGDA with quadratic regularization / PGD** (`src/methods/alg4.py`)
  - one LMO call per step;
  - dual-norm value recovered from the LMO support;
  - centered regularizer `rho (w_i - 1/K)` as written in the project draft;
  - Euclidean projection onto the simplex;
  - parameter step `x <- x + gamma D`.

Each is available with the three requested geometries:

- `L2`: exact global Euclidean LMO.
- `Spectral`: matrix-/convolution-shaped parameters use a spectral LMO with Muon-style Newton–Schulz zeropower; vectors use the sign fallback. `spectral_backend: svd` is also available for exact/debug comparisons.
- `Mixed`: hidden matrix/convolution parameters use the spectral LMO; output layers and vectors/biases use sign. The `scion` width scaling option follows the normalized spectral scaling used by Scion, including the convolution kernel-area factor.

The six proposed method names are therefore:

`Alg1-L2`, `Alg1-Spectral`, `Alg1-Mixed`, `Alg4-L2`, `Alg4-Spectral`, `Alg4-Mixed`.

### Runnable baselines

The reproduction config also contains runnable implementations of:

`LS`, `SI`, `RLW`, `DWA`, `UW`, `MGDA`, `PCGrad`, `GradDrop`, `CAGrad`, `MoCo`, `MoDo`, `Nash-MTL`, `FAMO`, and `MGDA-warm`.

`SI` means the scale-invariant linear-scalarization baseline used by the public FAMO/Nash-MTL dense-prediction benchmark (`sum_i log(loss_i)`). `STL` is included as a **reported reference** from Zhang et al.; the main config can merge other reported Zhang numbers as well. See `THIRD_PARTY_REFERENCES.md` for implementation provenance and the places where an equivalent clean-room implementation is used instead of vendoring third-party source code.

The runnable baselines are intentionally **clean-room controllers inside this project's common training pipeline**. For experimental consistency, that pipeline follows the draft/Zhang protocol (SGD for model parameters); therefore it is not intended to be byte-for-byte identical to every baseline author's standalone trainer (for example, the public FAMO benchmark trainer uses its own optimizer setup). The default main comparison avoids overstating reproducibility by training the six proposed methods and merging Zhang et al.'s published baseline rows; use `main_comparison_reproduce.yaml` when you explicitly want to retrain the runnable baselines under this common pipeline.

## Repository layout

```text
LMO-MOO/
├── main_experiments/
│   ├── run.py
│   ├── tune.py
│   └── configs/
│       ├── main_comparison.yaml
│       ├── main_comparison_reproduce.yaml
│       └── wall_clock.yaml
├── ablations/
│   ├── run.py
│   └── configs/
│       ├── effect_k.yaml
│       ├── lmo_geometry.yaml
│       ├── local_smoothness.yaml
│       └── hyperparameter_transfer.yaml
├── src/
│   ├── cache.py
│   ├── config.py
│   ├── runner.py
│   ├── report.py
│   ├── reported.py
│   ├── data/
│   ├── models/
│   ├── methods/
│   └── experiments/
├── tests/
├── clear_method_caches.ps1
├── clear_method_caches.sh
└── requirements.txt
```

## Experiment mapping

| Draft experiment | Folder / `--problem` | What the code runs |
|---|---|---|
| 1. Main comparison | `main_experiments`, `main_comparison` | CityScapes + NYU-v2 table; proposed methods; optionally runnable and/or reported baselines |
| 2. Effect of K | `ablations`, `effect_k` | CityScapes `K=2` vs NYU-v2 `K=3` |
| 3. LMO geometry | `ablations`, `lmo_geometry` | L2 vs Spectral vs Mixed with algorithm held fixed |
| 4. Local smoothness vs gradient norm | `ablations`, `local_smoothness` | per-task single-task SGD scatter experiment |
| 5. Wall-clock | `main_experiments`, `wall_clock` | time/epoch plus method timing components |
| 6. Hyperparameter transfer | `ablations`, `hyperparameter_transfer` | choose gamma on width `0.5`, transfer to widths `1` and `2` |

The optional high-`K` CIFAR-100 extension mentioned in the draft is deliberately **not fabricated** here because the draft marks it optional and does not fully specify the task construction/model protocol.

## Environment

Recommended: Python 3.11+ and a CUDA-enabled PyTorch build for the full experiments.

```powershell
cd LMO-MOO
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` does not pin a specific CUDA wheel. Install the PyTorch build appropriate for the machine if the default PyPI resolution is not the one you want.

## Dataset format

The code expects the prepared `.npy` dense-prediction format used by the public MTAN/FAMO-style benchmark.

### CityScapes

Default config root: `D:/datasets/cityscapes`

```text
D:/datasets/cityscapes/
├── train/
│   ├── image/*.npy
│   ├── label_7/*.npy
│   └── depth/*.npy
└── val/                 # `test/` is also accepted if `val/` is absent
    ├── image/*.npy
    ├── label_7/*.npy
    └── depth/*.npy
```

A `label/` directory is accepted as a compatibility fallback, but `label_7/` is the public CityScapes benchmark convention.

### NYU-v2

Default config root: `D:/datasets/nyuv2`

```text
D:/datasets/nyuv2/
├── train/
│   ├── image/*.npy
│   ├── label/*.npy
│   ├── depth/*.npy
│   └── normal/*.npy
└── val/                 # `test/` is also accepted if `val/` is absent
    ├── image/*.npy
    ├── label/*.npy
    ├── depth/*.npy
    └── normal/*.npy
```

NYU-v2 training enables the public benchmark's random scale + horizontal-flip augmentation by default. CityScapes augmentation is disabled in the supplied configs.

## Running the main experiments

All supplied configs save beneath the requested `D:/LMO-MOO-runs` root.

### Experiment 1 — main comparison

The default is efficient: train the **six proposed methods** and merge the Zhang et al. reported baselines into the generated table.

```powershell
python main_experiments/run.py --problem main_comparison
```

To actually retrain all runnable baselines as well:

```powershell
python main_experiments/run.py --problem main_comparison `
  --config main_experiments/configs/main_comparison_reproduce.yaml
```

Run one dataset or a subset of methods:

```powershell
python main_experiments/run.py --problem main_comparison --dataset cityscapes
python main_experiments/run.py --problem main_comparison --methods Alg1-Spectral,Alg4-Spectral
```

### Experiment 5 — wall-clock

```powershell
python main_experiments/run.py --problem wall_clock
```

The table contains average minutes/epoch and, where available, gradient, LMO, weight-update, and parameter-update timing components. A warm-up epoch is excluded by default.

## Running the ablations

```powershell
python ablations/run.py --problem effect_k
python ablations/run.py --problem lmo_geometry
python ablations/run.py --problem local_smoothness
python ablations/run.py --problem hyperparameter_transfer
```

The same common overrides are available:

```text
--config PATH
--output PATH
--methods A,B,C
--dataset cityscapes|nyuv2
--reset-cache
--plot-only
```

For hyperparameter transfer, the first selected method is the method that is tuned/transferred. The supplied default is `Alg1-Spectral`, matching the draft's expected practical winner.

## Hyperparameter grid search

The draft's proposed-method search grids are stored under `search_space` in the YAML files:

- `gamma`: `1e-4, 2e-4, 5e-4, 1e-3`
- weight step `eta`/`beta`: `0.1, 0.5, 1.0`
- Algorithm 4 `rho`: `0.01, 0.05, 0.1, 0.5, 1.0`
- LMO radius `R`: `1, 5, 10`
- optional stochastic momentum `alpha`: `0.1, 0.5, 0.9`

Use `tune.py` for a fingerprint-cached Cartesian sweep:

```powershell
python main_experiments/tune.py --method Alg1-Spectral --dataset cityscapes
python main_experiments/tune.py --method Alg4-Mixed --dataset nyuv2
```

Useful development options:

```powershell
python main_experiments/tune.py --method Alg1-L2 --dataset cityscapes --max-trials 4 --epochs 2
python main_experiments/tune.py --method Alg1-Spectral --dataset nyuv2 --include-no-momentum
```

`tune.py` writes `tables/trials.csv`, `tables/trials.md`, and `best_method.yaml` under `D:/LMO-MOO-runs/tuning/<dataset>/<method>/` unless `--output` is supplied.

## Training protocol encoded in the configs

The default full-study protocol is:

- SegNet hard-sharing dense-prediction network.
- CityScapes: semantic segmentation (7 classes) + depth, batch size 8.
- NYU-v2: semantic segmentation (13 classes) + depth + surface normals, batch size 2.
- 200 epochs.
- three seeds: `2025, 2026, 2027`.
- final metric = mean over the last 10 epochs; tables also report across-seed standard deviations.
- model and proposed LMO updates follow the explicit algorithmic step sizes rather than silently introducing Adam.

The Zhang warm-start baseline defaults are also encoded: `beta=0.5`, `rho=0.5`, warm start `N=40`, with the paper's dataset-specific model step sizes (`5e-4` for CityScapes, `1e-4` for NYU-v2) applied through `by_dataset` overrides.

## Metrics

The implementation reports the draft/Zhang metrics:

- semantic segmentation: mIoU, pixel accuracy;
- depth: absolute error, relative error;
- NYU-v2 surface normals: mean/median angular error and percentages within `11.25`, `22.5`, and `30` degrees;
- `Delta m %` against the Zhang STL reference;
- method rank across available task metrics; when proposed methods are added, `mr` is recomputed over the expanded comparison set while Zhang's original paper rank is retained as `mr_reported`.

The `Delta m %` sign convention is unit-tested against Zhang's reported CityScapes MGDA row.

## Cache and resume behavior

Cache semantics intentionally mirror the supplied previous project.

Each result is isolated by:

- experiment/problem;
- dataset;
- method;
- seed;
- normalized config;
- source fingerprint of the relevant training/method implementation.

Consequences:

1. Completed jobs are reused.
2. Partial epoch checkpoints resume after interruption.
3. Removing one method's cache does **not** invalidate unrelated methods.
4. Changing a method/config/source creates a new fingerprint rather than accidentally reusing stale state.
5. `--plot-only` reads completed cache/report material and never starts missing training runs.

Clear everything for the selected problem with `--reset-cache`, or remove selected method caches with:

```powershell
.\clear_method_caches.ps1 -Root "D:\LMO-MOO-runs\main_experiments\main_comparison" -Methods "Alg1-L2","Alg4-L2"
```

Linux/macOS equivalent:

```bash
./clear_method_caches.sh /path/to/run/root Alg1-L2 Alg4-L2
```

## Output structure

A normal run creates:

```text
D:/LMO-MOO-runs/<group>/<problem>/
├── cache/
├── configs/
├── figures/
├── logs/
└── tables/
```

Tables are saved as CSV and, when the optional pandas markdown backend is available, Markdown. Figures are regenerated from aggregate results and do not require retraining when the cache is intact.

## Tests and smoke checks

Run all tests from the repository root:

```powershell
python -m unittest discover -s tests -v
```

The test suite covers, among other things:

- Euclidean simplex projection;
- exact L2-LMO radius/support identity;
- sign LMO;
- Newton–Schulz finite/descent orientation;
- Scion convolutional spectral scaling;
- exact Algorithm 1 positive-sign entropic update and Algorithm 4 centered PGD formula, plus weight-simplex invariants;
- CityScapes `label_7` data layout;
- MGDA min-norm weights;
- Zhang `Delta m %` consistency;
- runnable baseline one-step smoke tests, including MoDo's three independent batches;
- full one-epoch synthetic CityScapes Algorithm 1 run;
- full one-epoch synthetic NYU-v2 Algorithm 4 spectral run.

The synthetic dataset path (`root: __synthetic__`) exists **only for testing and development** and is not used by the paper configs.

## Reproducibility / practical caveat

The repository has been syntax-tested, unit-tested, and exercised end-to-end with synthetic data. No local test can certify bit-for-bit reproduction on your unseen prepared datasets, Windows/CUDA stack, or long 200-epoch jobs. Before launching the entire grid, the safest sequence is:

1. install dependencies;
2. point one config at the real CityScapes data;
3. run `MGDA` or `Alg1-L2` for a short development run;
4. verify metrics/data ranges;
5. run the complete grid.

That check is especially important for depth preprocessing, because dense-prediction repositories sometimes distribute preprocessed targets with different physical/scaling conventions.

## References / provenance

See `THIRD_PARTY_REFERENCES.md`. No third-party repository is copied wholesale into this project; the codebase uses compact, clean-room implementations cross-checked against the cited public sources.
