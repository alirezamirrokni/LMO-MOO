import math

import torch

from methods.lmo_utils import (
    _kl_project_lower_bounded_simplex,
    _muon_zeropower_newton_schulz5,
    _task_gradient_matrix,
    _unique_parameters,
)

_EPS = 1.0e-12


def _positive_block_oracle(gradient, ns_steps=5):
    """Return the positive duality-map / polar direction for one parameter block.

    Matrix and convolutional tensors follow the MOON/Muon convention: flatten all
    dimensions except the output dimension, compute the Newton--Schulz polar
    factor, then reshape back. No width/kernel scaling is applied: the outer
    learning rate controls the step magnitude.

    0D/1D tensors use the Euclidean unit-ball LMO for a well-defined product
    geometry on BatchNorm/bias parameters. This is deliberately much milder than
    the legacy BiasRMS block scaling.
    """
    if not torch.isfinite(gradient).all():
        raise FloatingPointError("orthogonal LMO received non-finite gradient")

    if gradient.ndim >= 2:
        matrix = gradient.reshape(gradient.shape[0], -1)
        return _muon_zeropower_newton_schulz5(
            matrix, steps=ns_steps
        ).reshape_as(gradient)

    flat = gradient.reshape(-1)
    norm = torch.linalg.vector_norm(flat.float())
    if not torch.isfinite(norm):
        raise FloatingPointError("vector LMO norm is non-finite")
    if float(norm.item()) <= _EPS:
        return torch.zeros_like(gradient)
    return gradient / norm.to(dtype=gradient.dtype)


def _flatten_block_directions(buffer, parameters, ns_steps):
    """Product-geometry direction with layerwise matrix polar blocks.

    All 0D/1D parameters are treated as *one* Euclidean block instead of one
    radius per bias/BatchNorm tensor. This keeps the tiny non-matrix parameter
    fraction from receiving a disproportionate update budget while preserving a
    genuine norm-constrained LMO for the full parameter vector.
    """
    positive_flat = torch.zeros_like(buffer)
    direction_sq = torch.zeros((), dtype=torch.float64, device=buffer.device)
    matrix_sq = torch.zeros_like(direction_sq)
    vector_sq = torch.zeros_like(direction_sq)

    vector_ranges = []
    offset = 0
    for parameter in parameters:
        count = int(parameter.numel())
        piece = buffer[offset : offset + count].view_as(parameter)
        if parameter.ndim >= 2:
            positive = _positive_block_oracle(piece, ns_steps=ns_steps)
            positive_flat[offset : offset + count].copy_(positive.reshape(-1))
            block_sq = torch.sum(positive.double().square())
            direction_sq = direction_sq + block_sq
            matrix_sq = matrix_sq + block_sq
        else:
            vector_ranges.append((offset, offset + count))
        offset += count

    if offset != int(buffer.numel()):
        raise RuntimeError("orthogonal MOO parameter layout mismatch")

    if vector_ranges:
        vector_gradient = torch.cat(
            [buffer[a:b] for a, b in vector_ranges], dim=0
        )
        norm = torch.linalg.vector_norm(vector_gradient.float())
        if not torch.isfinite(norm):
            raise FloatingPointError("global vector-block norm is non-finite")
        if float(norm.item()) > _EPS:
            vector_direction = vector_gradient / norm.to(vector_gradient.dtype)
            cursor = 0
            for a, b in vector_ranges:
                count = b - a
                positive_flat[a:b].copy_(vector_direction[cursor:cursor + count])
                cursor += count
            vector_sq = torch.sum(vector_direction.double().square())
            direction_sq = direction_sq + vector_sq

    return positive_flat, direction_sq, matrix_sq, vector_sq



class EntropicLMOMGDA:
    """Entropic LMO-MGDA with a single blended gradient buffer.

    Persistent state is one blended gradient buffer, not one buffer per task.
    The model update is written into ``parameter.grad`` as a *positive* polar
    direction because the trainer uses SGD: x <- x - lr * grad. Equivalently,
    with D = LMO(M) = -polar(M), this is x <- x + lr * D.
    """

    method_name = "entropic_lmo_mgda"
    log_prefix = "ENTROPIC-LMO-MGDA"

    def __init__(
        self,
        n_tasks,
        device,
        eta=1.0e-4,
        alpha=0.1,
        ns_steps=5,
        entropy_tau=0.0,
        weight_kappa=0.0,
        log_every=100,
    ):
        self.n_tasks = int(n_tasks)
        self.device = device
        self.eta = float(eta)
        self.alpha = float(alpha)
        self.ns_steps = int(ns_steps)
        self.entropy_tau = float(entropy_tau)
        self.weight_kappa = float(weight_kappa)
        self.log_every = int(log_every)

        if self.n_tasks < 2:
            raise ValueError("orthogonal MOO requires at least two tasks")
        if not math.isfinite(self.eta) or self.eta <= 0.0:
            raise ValueError("eta must be finite and positive")
        if not math.isfinite(self.alpha) or not (0.0 < self.alpha <= 1.0):
            raise ValueError("alpha/momentum injection must lie in (0, 1]")
        if self.ns_steps <= 0:
            raise ValueError("Newton-Schulz steps must be positive")
        if not math.isfinite(self.entropy_tau) or self.entropy_tau < 0.0:
            raise ValueError("entropy_tau must be finite and non-negative")
        if not math.isfinite(self.weight_kappa) or self.weight_kappa < 0.0:
            raise ValueError("weight_kappa must be finite and non-negative")
        if self.weight_kappa * self.n_tasks >= 1.0:
            raise ValueError("weight_kappa must satisfy K*kappa < 1")


        self.weights = torch.full(
            (self.n_tasks,),
            1.0 / self.n_tasks,
            dtype=torch.float64,
            device=device,
        )
        self.blended_momentum = None
        self.step_counter = 0
        self.parameter_numels = None

    def parameters(self):
        return []

    def _paper_weight_update(self, scores):
        score64 = scores.to(device=self.device, dtype=torch.float64)
        denominator = 1.0 + self.eta * self.entropy_tau
        log_proposal = (
            torch.log(self.weights) + self.eta * score64
        ) / denominator
        proposal = torch.softmax(log_proposal, dim=0)
        if self.weight_kappa > 0.0:
            proposal = _kl_project_lower_bounded_simplex(
                proposal, self.weight_kappa
            )
        tiny = torch.finfo(proposal.dtype).tiny
        proposal = torch.clamp(proposal, min=tiny)
        self.weights = (proposal / proposal.sum()).detach()

    def _validate_layout(self, parameters):
        numels = tuple(int(p.numel()) for p in parameters)
        if self.parameter_numels is None:
            self.parameter_numels = numels
        elif numels != self.parameter_numels:
            raise RuntimeError("model parameter layout changed during training")

    def _ensure_buffer(self, task_gradients):
        width = int(task_gradients.shape[1])
        if self.blended_momentum is None:
            self.blended_momentum = torch.zeros(
                width,
                device=task_gradients.device,
                dtype=task_gradients.dtype,
            )
        if self.blended_momentum.numel() != width:
            raise RuntimeError("blended momentum parameter layout mismatch")
        if self.blended_momentum.device != task_gradients.device:
            self.blended_momentum = self.blended_momentum.to(task_gradients.device)
        if self.blended_momentum.dtype != task_gradients.dtype:
            self.blended_momentum = self.blended_momentum.to(task_gradients.dtype)

    def _write_optimizer_direction(self, parameters, positive_flat):
        offset = 0
        for parameter in parameters:
            count = int(parameter.numel())
            parameter.grad = positive_flat[offset : offset + count].view_as(parameter).detach().clone()
            offset += count
        if offset != int(positive_flat.numel()):
            raise RuntimeError("optimizer direction layout mismatch")

    def backward(
        self,
        losses,
        shared_parameters=None,
        task_specific_parameters=None,
        **kwargs,
    ):
        del shared_parameters, task_specific_parameters
        parameters = kwargs.pop("model_parameters", None)
        del kwargs

        if parameters is None:
            raise RuntimeError("Entropic LMO-MGDA requires model_parameters")
        parameters = _unique_parameters(parameters)
        if not parameters:
            raise RuntimeError("Entropic LMO-MGDA received no trainable parameters")
        self._validate_layout(parameters)

        if int(losses.numel()) != self.n_tasks:
            raise ValueError(
                f"expected {self.n_tasks} losses, got {int(losses.numel())}"
            )

        task_gradients = _task_gradient_matrix(losses, parameters, False)
        self._ensure_buffer(task_gradients)
        raw_task_l2 = torch.linalg.vector_norm(task_gradients, dim=1)

        used_weights = self.weights.detach().clone()
        blend_weights = used_weights.to(
            device=task_gradients.device, dtype=task_gradients.dtype
        )
        aggregate = torch.sum(
            blend_weights[:, None] * task_gradients,
            dim=0,
        )

        # Draft Algorithm 2: D_t = lmo(M_{t-1}), then use current scores,
        # then M_t <- (1-alpha_t)M_{t-1}+alpha_t Ghat_t.
        direction_buffer = self.blended_momentum
        positive_flat, direction_sq, matrix_sq, vector_sq = _flatten_block_directions(
            direction_buffer, parameters, self.ns_steps
        )
        descent_flat = -positive_flat
        scores = task_gradients @ descent_flat

        self._paper_weight_update(scores)

        alpha_t = 1.0 if self.step_counter == 0 else self.alpha
        self.blended_momentum.mul_(1.0 - alpha_t).add_(
            aggregate, alpha=alpha_t
        )

        if not torch.isfinite(scores).all():
            raise FloatingPointError("MOO task scores are non-finite")
        if not torch.isfinite(self.blended_momentum).all():
            raise FloatingPointError("blended momentum is non-finite")

        self._write_optimizer_direction(parameters, positive_flat)
        self.step_counter += 1

        step = self.step_counter
        if step <= 10 or (self.log_every > 0 and step % self.log_every == 0):
            print(
                f"{self.log_prefix} step={step} "
                f"weights_used={used_weights.cpu().tolist()} "
                f"weights_next={self.weights.cpu().tolist()} "
                f"scores={scores.detach().cpu().tolist()} "
                f"task_l2={raw_task_l2.detach().cpu().tolist()} "
                f"Gbar_norm={float(torch.linalg.vector_norm(aggregate).item()):.8g} "
                f"M_norm={float(torch.linalg.vector_norm(self.blended_momentum).item()):.8g} "
                f"D_norm={float(torch.sqrt(direction_sq).item()):.8g} "
                f"D_matrix={float(torch.sqrt(matrix_sq).item()):.8g} "
                f"D_vector={float(torch.sqrt(vector_sq).item()):.8g} "
                f"alpha_t={alpha_t:.8g} alpha={self.alpha:.8g} "
                f"eta={self.eta:.8g} ns_steps={self.ns_steps}",
                flush=True,
            )

        return None, {
            "weights": self.weights.detach().clone(),
            "weights_used": used_weights,
            "scores": scores.detach().clone(),
            "task_l2_norms": raw_task_l2.detach().clone(),
            "aggregate_norm": torch.linalg.vector_norm(aggregate).detach().clone(),
            "momentum_norm": torch.linalg.vector_norm(self.blended_momentum).detach().clone(),
            "direction_norm": torch.sqrt(direction_sq).detach().clone(),
        }

    def state_dict(self):
        return {
            "ortho_moo_version": 1,
            "method_name": self.method_name,
            "eta": self.eta,
            "alpha": self.alpha,
            "ns_steps": self.ns_steps,
            "entropy_tau": self.entropy_tau,
            "weight_kappa": self.weight_kappa,
            "step_counter": int(self.step_counter),
            "weights": self.weights.detach().cpu(),
            "blended_momentum": (
                None
                if self.blended_momentum is None
                else self.blended_momentum.detach().cpu()
            ),
            "parameter_numels": self.parameter_numels,
        }

    def load_state_dict(self, state):
        if int(state.get("ortho_moo_version", -1)) != 1:
            raise RuntimeError("incompatible orthogonal-MOO checkpoint")
        if state.get("method_name") != self.method_name:
            raise RuntimeError(
                f"method mismatch: checkpoint={state.get('method_name')} requested={self.method_name}"
            )

        scalar_checks = {
            "eta": self.eta,
            "alpha": self.alpha,
            "entropy_tau": self.entropy_tau,
            "weight_kappa": self.weight_kappa,
        }
        for key, requested in scalar_checks.items():
            saved = float(state.get(key, requested))
            if abs(saved - requested) > 1.0e-15:
                raise RuntimeError(
                    f"{key} mismatch: checkpoint={saved}, requested={requested}"
                )
        if int(state.get("ns_steps", self.ns_steps)) != self.ns_steps:
            raise RuntimeError("ns_steps mismatch in orthogonal-MOO checkpoint")

        self.step_counter = int(state["step_counter"])
        self.weights = state["weights"].to(self.device, dtype=torch.float64)
        momentum = state.get("blended_momentum")
        self.blended_momentum = None if momentum is None else momentum.to(self.device)
        saved_numels = state.get("parameter_numels")
        self.parameter_numels = None if saved_numels is None else tuple(saved_numels)

