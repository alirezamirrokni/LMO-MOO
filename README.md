# Geometry-Aware Multi-Objective Optimization

Code for [Geometry-Aware Multi-Objective Optimization: Norm-Constrained Oracles with Entropic Weight Updates](https://arxiv.org/pdf/2610.13773).

Entropic LMO-MGDA updates the network with a linear minimization oracle (spectral by default, via Newton-Schulz) and updates the task weights with an entropic (mirror-descent) step on the simplex, using momentum on the blended gradient.

## Setup

```bash
conda create -n lmo-moo python=3.10 -y
conda activate lmo-moo
pip install -r requirements.txt
```

## Data

Cityscapes and NYUv2 use the preprocessed `.npy` versions from [MTAN](https://github.com/lorenmt/mtan). Place them as:

```
datasets/cityscapes/{train,val}/{image,label_7,depth}/*.npy
datasets/nyuv2/{train,val}/{image,label,depth,normal}/*.npy
```

Generate MultiMNIST (downloads MNIST, seed 2026, 10000 train / 1000 test images):

```bash
python prepare_multimnist.py
```

## Cityscapes and NYUv2

Each run trains for 200 epochs with seed 42. The paper's learning rates and schedules are applied automatically for every method.

```bash
for dataset in cityscapes nyuv2; do
  for method in stl ls scaleinvls rlw dwa uw mgda pcgrad graddrop cagrad moco modo nashmtl famo mgda_warm moon entropic_lmo_mgda; do
    python main.py --dataset $dataset --method $method
  done
done
```

Results are written to `results/<dataset>/<method>_sd42.stats`. Runs resume from `results/checkpoints/`. Add `--force` to restart from scratch.

## MultiMNIST

Each run trains for 100 epochs with batch size 256 and seeds 42, 43 and 44. Entropic LMO-MGDA uses lr 1e-3, η = 5e-5, α = 0.1 and κ = 0.

```bash
for method in entropic_lmo_mgda mgda mgda_muon famo famo_muon muon_ls moon; do
  python main.py --dataset multimnist --method $method
done
```

Ablations (Entropic LMO-MGDA):

```bash
python main.py --dataset multimnist --method entropic_lmo_mgda --oracle l2
python main.py --dataset multimnist --method entropic_lmo_mgda --oracle sign
python main.py --dataset multimnist --method entropic_lmo_mgda --weights projected
python main.py --dataset multimnist --method entropic_lmo_mgda --momentum per-task
python main.py --dataset multimnist --method entropic_lmo_mgda --momentum none
for eta in 1e-5 1e-4 5e-4 1e-3; do python main.py --dataset multimnist --method entropic_lmo_mgda --eta $eta; done
for kappa in 0.48 0.49 0.495 0.4975; do python main.py --dataset multimnist --method entropic_lmo_mgda --kappa $kappa; done
```

Add `--validation` to evaluate on a held-out 10% of the training set instead of the test set (used for hyperparameter selection).

## Results

```bash
python report.py
```

This prints all tables below. For Cityscapes and NYUv2, metrics are averaged over the last 10 epochs, and Δm% is measured relative to STL.

### Cityscapes

| Method | mIoU ↑ | Pix Acc ↑ | Abs Err ↓ | Rel Err ↓ | Δm% ↓ |
|---|---|---|---|---|---|
| STL | 74.11 | 93.44 | 0.0137 | 38.22 | 0.00 |
| LS | 74.80 | 93.37 | 0.0188 | 42.41 | 11.83 |
| SI | 67.77 | 89.97 | 0.0151 | 34.65 | 3.29 |
| RLW | 74.24 | 93.30 | 0.0169 | 47.09 | 11.63 |
| DWA | 74.42 | 93.41 | 0.0159 | 41.94 | 6.35 |
| UW | 71.56 | 92.15 | 0.0191 | 38.75 | 11.41 |
| MGDA | 73.12 | 92.80 | 0.0157 | 36.03 | 2.72 |
| PCGrad | 75.89 | 93.72 | 0.0150 | 45.83 | 6.67 |
| GradDrop | 75.85 | 93.70 | 0.0151 | 43.60 | 5.42 |
| CAGrad | 75.83 | 93.50 | 0.0145 | 35.89 | -0.66 |
| MoCo | 74.37 | 93.17 | 0.0221 | 81.06 | 43.33 |
| MoDo | 74.71 | 93.51 | 0.0152 | 44.31 | 6.50 |
| Nash-MTL | 74.76 | 93.29 | 0.0144 | 33.47 | -2.01 |
| FAMO | 74.19 | 93.28 | 0.0154 | **30.62** | -1.85 |
| MGDA (warm) | 75.64 | 93.60 | 0.0147 | 47.00 | 7.01 |
| MOON | <u>78.53</u> | <u>94.15</u> | <u>0.0141</u> | <u>31.39</u> | **-5.42** |
| **Entropic LMO-MGDA** | **79.03** | **94.48** | **0.0136** | 36.29 | <u>-3.38</u> |

### NYUv2

| Method | mIoU ↑ | Pix Acc ↑ | Abs ↓ | Rel ↓ | Mean ↓ | Med. ↓ | <11.25 ↑ | <22.5 ↑ | <30 ↑ | Δm% ↓ |
|---|---|---|---|---|---|---|---|---|---|---|
| STL | 38.77 | 63.98 | 0.6350 | 0.2654 | 25.10 | 19.06 | 30.12 | 57.66 | 69.53 | 0.00 |
| LS | 38.13 | 64.83 | 0.5523 | 0.2334 | 27.79 | 23.56 | 23.03 | 48.23 | 61.64 | 6.76 |
| SI | 38.59 | 64.57 | 0.5439 | 0.2339 | 27.57 | 23.43 | 22.81 | 48.45 | 62.18 | 6.32 |
| RLW | 38.45 | 64.61 | 0.5814 | 0.2454 | 27.83 | 23.46 | 23.20 | 48.51 | 61.81 | 7.53 |
| DWA | 39.57 | 65.19 | 0.5691 | 0.2399 | 27.40 | 22.87 | 24.26 | 49.61 | 62.81 | 5.37 |
| UW | 38.48 | 63.88 | 0.5555 | 0.2294 | 27.47 | 23.38 | 23.18 | 48.60 | 62.29 | 6.23 |
| MGDA | 33.79 | 62.12 | 0.5906 | 0.2398 | <u>24.47</u> | **18.65** | **31.28** | **58.50** | **70.28** | -1.33 |
| PCGrad | 38.79 | 65.08 | 0.5501 | 0.2212 | 27.18 | 22.74 | 24.10 | 49.93 | 63.36 | 4.23 |
| GradDrop | 40.37 | 66.04 | 0.5536 | 0.2353 | 27.97 | 23.76 | 22.78 | 47.97 | 61.39 | 6.39 |
| CAGrad | <u>41.61</u> | 66.95 | 0.5444 | 0.2251 | 25.75 | 20.79 | 27.44 | 53.99 | 66.85 | -1.18 |
| MoCo | 35.47 | 61.76 | 0.5808 | 0.2328 | 31.21 | 27.97 | 15.31 | 39.38 | 54.01 | 18.38 |
| MoDo | 39.41 | 65.82 | 0.5353 | 0.2225 | 25.54 | 20.08 | 28.57 | 55.34 | 67.78 | -1.96 |
| Nash-MTL | 38.18 | 65.39 | 0.5575 | 0.2325 | 25.12 | 19.85 | 29.13 | 55.97 | 68.48 | -1.48 |
| FAMO | 32.57 | 61.67 | 0.5411 | 0.2220 | 25.16 | 19.25 | 30.18 | 57.16 | 69.10 | -1.00 |
| MGDA (warm) | 38.41 | 63.76 | 0.5813 | 0.2453 | 28.13 | 24.03 | 21.43 | 47.24 | 61.13 | 9.15 |
| MOON | 39.88 | <u>67.30</u> | **0.4951** | <u>0.2152</u> | 25.50 | 20.02 | 29.44 | 55.34 | 67.71 | <u>-3.72</u> |
| **Entropic LMO-MGDA** | **42.71** | **68.07** | <u>0.5075</u> | **0.2129** | **23.76** | <u>19.18</u> | <u>30.27</u> | <u>57.34</u> | <u>69.60</u> | **-6.80** |

### MultiMNIST

Test accuracy (%), averaged over three seeds.

| Method | Left ↑ | Right ↑ | Avg. ↑ |
|---|---|---|---|
| MGDA | 83.63 | 85.13 | 84.38 |
| MGDA + Muon | 84.48 | 86.43 | 85.45 |
| FAMO | 85.24 | 87.29 | 86.27 |
| FAMO + Muon | 87.28 | 88.02 | 87.65 |
| Muon (LS) | <u>87.45</u> | 88.04 | <u>87.75</u> |
| MOON | 86.87 | <u>88.06</u> | 87.47 |
| **Entropic LMO-MGDA** | **89.86** | **91.23** | **90.54** |

### MultiMNIST ablations

| Component | Variant | Avg. ↑ | Gap ↓ |
|---|---|---|---|
| Oracle | l2 | 49.59 | 0.55 |
| | sign | 89.01 | 6.38 |
| | spectral | 90.54 | 0.90 |
| Weights | entropic | 90.54 | 0.90 |
| | projected | 90.51 | 0.97 |
| Momentum | blended | 90.54 | 0.90 |
| | per-task | 90.16 | 1.41 |
| | none | 81.68 | 1.65 |

| η | 1e-5 | 5e-5 | 1e-4 | 5e-4 | 1e-3 |
|---|---|---|---|---|---|
| Avg. ↑ | 90.20 | **90.54** | 90.08 | 89.69 | 89.53 |
| Gap ↓ | 1.97 | 0.90 | 0.84 | 0.29 | 1.04 |

| κ | 0 | 0.48 | 0.49 | 0.495 | 0.4975 |
|---|---|---|---|---|---|
| Avg. ↑ | 90.54 | 90.54 | 90.73 | **91.11** | 90.96 |
| Gap ↓ | **0.90** | **0.90** | 1.17 | 1.22 | 1.65 |

## Citation

```bibtex
@article{chayti2026geometry,
  title={Geometry-Aware Multi-Objective Optimization: Norm-Constrained Oracles with Entropic Weight Updates},
  author={Chayti, El Mahdi and Mirrokni, Alireza and Jaggi, Martin},
  journal={arXiv preprint arXiv:2610.13773},
  year={2026}
}
```
