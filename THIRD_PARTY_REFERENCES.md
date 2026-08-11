# Third-party references and implementation provenance

This repository is a clean-room implementation for the supplied project draft. It does **not** vendor the FAMO, LibMTL, Scion, MTAN, Muon, or other baseline repositories.

## Primary project sources supplied by the user

1. **Norm-Constrained LMO Methods for Multi-Objective Optimization — Setting, Algorithms, and Experimental Protocol** (`lmo_mgda-1.pdf`).
   - Source of proposed Algorithm 1 and Algorithm 4, the six-experiment protocol, datasets, metrics, and proposed hyperparameter grids.
2. **Qi Zhang, Peiyao Xiao, Shaofeng Zou, Kaiyi Ji. “MGDA Converges Under Generalized Smoothness, Provably.” ICLR 2025.** (`2405.19440v5.pdf`).
   - Source of the reported CityScapes/NYU-v2 reference tables, MGDA-warm-start experiment details, local-smoothness experiment, and wall-clock comparison.

## Public codebases checked during implementation

### FAMO unified dense-prediction baselines

- Repository: https://github.com/Cranial-XIX/FAMO
- Key source: https://raw.githubusercontent.com/Cranial-XIX/FAMO/refs/heads/main/methods/weight_methods.py

Used to cross-check the public benchmark definitions/formulas for:

- FAMO;
- linear scalarization (LS);
- scale-invariant linear scalarization (`scaleinvls`, called `SI` in the Zhang table);
- uncertainty weighting (UW);
- MGDA;
- PCGrad;
- GradDrop;
- CAGrad;
- Nash-MTL benchmark conventions.

The local Nash implementation solves the same positive Nash first-order system `A alpha = 1/alpha` directly through the convex objective `0.5 alpha^T A alpha - sum(log alpha)` with SciPy. This avoids adding CVXPY/ECOS as runtime dependencies while preserving the Nash bargaining fixed point. The raw Nash weights are deliberately not normalized like MGDA weights.

### FAMO / MTAN dense-prediction model and data conventions

- FAMO repository: https://github.com/Cranial-XIX/FAMO
- MTAN repository: https://github.com/lorenmt/mtan

Used to cross-check:

- SegNet-style hard-sharing architecture and initialization;
- CityScapes 7-class semantic + depth setup;
- NYU-v2 13-class semantic + depth + surface-normal setup;
- prepared `.npy` data organization;
- the CityScapes semantic directory name `label_7`;
- NYU-v2 random-scale/horizontal-flip augmentation family;
- averaging evaluation over the final training epochs.

### MoDo

- Author repository: https://github.com/heshandevaka/Trade-Off-MOL
- Cross-checked against the author-sharing implementation exposed by LibMTL: https://github.com/median-research-group/LibMTL

Important implementation detail preserved here: MoDo uses **three independent stochastic gradient samples/batches** in one iteration. Two are used in the stochastic cross-Gram weight update; a third is used for the model direction.

### MoCo

- Paper: Heshan Fernando et al., “Mitigating Gradient Bias in Multi-Objective Learning: A Provably Convergent Approach.”
- Author-sharing implementation cross-checked through LibMTL: https://github.com/median-research-group/LibMTL

The implementation keeps the gradient-tracking variable and the polynomially decaying tracking/weight steps rather than replacing MoCo with a generic MGDA approximation.

### Scion / normalized spectral LMO

- Repository: https://github.com/LIONS-EPFL/scion

Used to cross-check normalized spectral scaling. For a convolution weight with shape `(out, in, kh, kw)`, the Scion-style normalized spectral radius includes both the `sqrt(out/in)` factor and the spatial kernel-area divisor. Matrix and 1-D-convolution cases use their corresponding normalized factors.

### Muon / Newton-Schulz zeropower

- Modded nanoGPT / Muon reference: https://github.com/KellerJordan/modded-nanogpt
- Related optimizer/norm reference cited by the draft: Bernstein & Newhouse, “Old Optimizer, New Norm: An Anthology.”

Used to cross-check the quintic Newton–Schulz zeropower iteration employed for the approximate matrix polar factor.

## Baseline papers

The implementation was also checked against the algorithmic definitions in the original works named by the project draft:

- Désidéri (2012), MGDA.
- Sener & Koltun (2018), multi-task learning as multi-objective optimization.
- Kendall et al. (2018), uncertainty weighting.
- Liu et al. (2019), DWA / MTAN.
- Yu et al. (2020), PCGrad.
- Chen et al. (2020), GradDrop.
- Liu et al. (2021), CAGrad.
- Lin et al. (2021), RLW.
- Fernando et al. (2022), MoCo.
- Navon et al. (2022), Nash-MTL.
- Chen et al. (2024), MoDo.
- Liu et al. (2024), FAMO.

## Reported versus rerun baselines

`main_experiments/configs/main_comparison.yaml` trains the six proposed LMO-MGDA variants and merges the published Zhang et al. baseline numbers. This matches the draft instruction that baselines may be “reproduce[d] **or report[ed]** from the paper.”

`main_experiments/configs/main_comparison_reproduce.yaml` additionally reruns all baseline controllers implemented in this repository. The reported Zhang rows remain available as a reference, while aggregate-table logic does not duplicate a `(dataset, method)` entry when that method was actually rerun.

`STL` remains a reported reference because a fair STL reproduction requires separate task-specific networks/runs and is primarily used here as the fixed denominator for `Delta m %`.
