import math
import torch
_EPS = 1e-30

def _muon_zeropower_newton_schulz5(matrix, steps=5):
    if matrix.ndim != 2:
        raise ValueError(
            f"Muon Newton-Schulz expects a matrix, got shape={tuple(matrix.shape)}"
        )

    if steps <= 0:
        raise ValueError("Muon Newton-Schulz steps must be positive")

    original_dtype = matrix.dtype
    x = matrix.float()

    if x.shape[0] > x.shape[1]:
        x = x.transpose(0, 1)

    norm = torch.linalg.vector_norm(x)

    if not torch.isfinite(norm):
        raise FloatingPointError("Muon Newton-Schulz input norm is not finite")

    if float(norm.item()) <= _EPS:
        return torch.zeros_like(matrix)

    x = x / (norm + 1.0e-7)
    a, b, c = 3.4445, -4.7750, 2.0315

    for _ in range(int(steps)):
        aa = x @ x.transpose(0, 1)
        bb = b * aa + c * (aa @ aa)
        x = a * x + bb @ x

    if matrix.shape[0] > matrix.shape[1]:
        x = x.transpose(0, 1)

    return x.to(dtype=original_dtype)

def _unique_parameters(parameters, excluded_ids=None):
    excluded = set() if excluded_ids is None else set(excluded_ids)
    output = []
    seen = set()

    for parameter in parameters:
        if parameter is None or not parameter.requires_grad:
            continue

        identifier = id(parameter)

        if identifier in excluded or identifier in seen:
            continue

        seen.add(identifier)
        output.append(parameter)

    return output

def _task_gradient_matrix(losses, parameters, keep_graph_after):
    rows = []
    n_tasks = int(losses.numel())

    for task_index in range(n_tasks):
        gradients = torch.autograd.grad(
            losses[task_index],
            parameters,
            retain_graph=(task_index < n_tasks - 1) or keep_graph_after,
            allow_unused=True,
        )

        pieces = []

        for parameter, gradient in zip(parameters, gradients):
            value = torch.zeros_like(parameter) if gradient is None else gradient.detach()
            pieces.append(value.reshape(-1))

        rows.append(torch.cat(pieces))

    result = torch.stack(rows, dim=0)

    if not torch.isfinite(result).all():
        raise FloatingPointError("task gradients are non-finite")

    return result

def _kl_project_lower_bounded_simplex(probabilities, lower_bound):
    probabilities = probabilities.to(dtype=torch.float64)

    if probabilities.ndim != 1 or probabilities.numel() == 0:
        raise ValueError("probabilities must be a non-empty vector")
    if not torch.isfinite(probabilities).all():
        raise FloatingPointError("probabilities contain non-finite values")
    if torch.any(probabilities < 0):
        raise ValueError("probabilities must be non-negative")

    n = int(probabilities.numel())
    lower_bound = float(lower_bound)
    if not math.isfinite(lower_bound) or lower_bound < 0.0:
        raise ValueError("lower_bound must be finite and non-negative")
    if lower_bound * n >= 1.0:
        raise ValueError("lower_bound must satisfy n_tasks * lower_bound < 1")

    total = probabilities.sum()
    if float(total.item()) <= 0.0:
        raise ValueError("probabilities must have positive sum")

    probabilities = probabilities / total
    if lower_bound == 0.0:
        return probabilities

    free = torch.ones(n, dtype=torch.bool, device=probabilities.device)
    projected = torch.empty_like(probabilities)

    while True:
        fixed_count = n - int(free.sum().item())
        remaining_mass = 1.0 - fixed_count * lower_bound
        free_mass = probabilities[free].sum()

        if float(free_mass.item()) <= 0.0:
            raise FloatingPointError("KL projection encountered zero free probability mass")

        scale = remaining_mass / free_mass
        candidate = scale * probabilities
        newly_fixed = free & (candidate < lower_bound)

        if not bool(newly_fixed.any().item()):
            projected[~free] = lower_bound
            projected[free] = candidate[free]
            break

        free = free & ~newly_fixed

    projected = projected / projected.sum()

    if not torch.isfinite(projected).all():
        raise FloatingPointError("KL projection produced non-finite weights")
    if torch.any(projected < lower_bound - 1.0e-12):
        raise FloatingPointError("KL projection violated the lower bound")

    return projected
