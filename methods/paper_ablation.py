"""Controlled ablations of the supplied EntropicLMOMGDA implementation."""
import torch
from methods.entropic_lmo_mgda import EntropicLMOMGDA, _flatten_block_directions
from methods.lmo_utils import _task_gradient_matrix, _unique_parameters


def simplex_projection(v):
    u, _ = torch.sort(v, descending=True)
    cssv = torch.cumsum(u, 0) - 1
    idx = torch.arange(1, len(v)+1, device=v.device, dtype=v.dtype)
    rho = torch.nonzero(u-cssv/idx > 0)[-1,0]
    return (v-cssv[rho]/idx[rho]).clamp_min(0)


def positive_oracle(buffer, parameters, geometry='spectral', ns_steps=5, exact=False):
    if geometry == 'l2':
        return buffer / buffer.norm().clamp_min(1e-12)
    if geometry == 'sign':
        return buffer.sign()
    if not exact:
        return _flatten_block_directions(buffer, parameters, ns_steps)[0]
    # Exact product-ball oracle for gap measurement, not the finite-NS surrogate.
    result = torch.zeros_like(buffer)
    vector_indices = []
    start = 0
    for p in parameters:
        end = start+p.numel()
        if p.ndim >= 2:
            g = buffer[start:end].reshape(p.shape[0], -1)
            if torch.count_nonzero(g):
                u, s, vh = torch.linalg.svd(g, full_matrices=False)
                # Zero singular subspaces may be assigned zero (valid LMO).
                keep = (s > torch.finfo(s.dtype).eps * max(g.shape) * s.max()).to(s.dtype)
                result[start:end] = ((u*keep) @ vh).reshape(-1)
        else:
            vector_indices.extend(range(start,end))
        start=end
    if vector_indices:
        g = buffer[vector_indices]
        result[vector_indices] = g/g.norm().clamp_min(1e-12)
    return result


class PaperAblation(EntropicLMOMGDA):
    def __init__(self, *args, oracle='spectral', weight_update='entropic', momentum='blended', clipping='task', clip_value=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        if oracle not in {'spectral','l2','sign'} or weight_update not in {'entropic','projected'} or momentum not in {'blended','per-task','none'} or clipping not in {'task','none'}:
            raise ValueError('Invalid paper ablation')
        if clip_value <= 0:
            raise ValueError('clip_value must be positive')
        self.oracle, self.weight_update, self.momentum = oracle, weight_update, momentum
        self.clipping, self.clip_value = clipping, float(clip_value)
        if weight_update == 'projected' and (self.entropy_tau or self.weight_kappa):
            raise ValueError('Projected ablation is defined with tau=kappa=0')
        self.task_momentum = None

    def backward(self, losses, shared_parameters=None, task_specific_parameters=None, **kwargs):
        params = _unique_parameters(kwargs['model_parameters'])
        self._validate_layout(params)
        gradients = _task_gradient_matrix(losses, params, False)
        self._ensure_buffer(gradients)
        used = self.weights.clone()
        blend = used.to(gradients)
        aggregate = (blend[:,None]*gradients).sum(0)
        if self.momentum == 'blended':
            buffer = self.blended_momentum
        elif self.momentum == 'per-task':
            if self.task_momentum is None:
                self.task_momentum = torch.zeros_like(gradients)
            buffer = (blend[:,None]*self.task_momentum).sum(0)
        else:
            buffer = aggregate
        direction = positive_oracle(buffer, params, self.oracle, self.ns_steps)
        # Direction clipping is the key stabilization mechanism that differentiates
        # our LMO update from the un-clipped MOON-style polar update. The ablation
        # disables this normalization to measure its contribution.
        if self.clipping == 'task':
            direction_norm = direction.norm()
            if float(direction_norm.item()) > self.clip_value:
                direction = direction * (self.clip_value / direction_norm.clamp_min(1e-12))
        scores = gradients @ (-direction)
        if self.weight_update == 'entropic':
            self._paper_weight_update(scores)
        else:
            self.weights = simplex_projection(self.weights + self.eta*scores.to(self.weights)).detach()
        alpha = 1.0 if self.step_counter == 0 else self.alpha
        self.blended_momentum.mul_(1-alpha).add_(aggregate, alpha=alpha)
        if self.momentum == 'per-task':
            self.task_momentum.mul_(1-alpha).add_(gradients, alpha=alpha)
        self._write_optimizer_direction(params, direction)
        self.step_counter += 1
        if not torch.isfinite(direction).all() or not torch.isfinite(self.weights).all():
            raise FloatingPointError('Nonfinite paper direction/weights')
        return None, dict(weights=self.weights.clone(), weights_used=used, scores=scores.detach())

    def state_dict(self):
        state = super().state_dict()
        state.update(ablation_version=1, oracle=self.oracle, weight_update=self.weight_update,
                     momentum=self.momentum, task_momentum=None if self.task_momentum is None else self.task_momentum.cpu())
        return state

    def load_state_dict(self,state):
        for k in ('oracle','weight_update','momentum'):
            if state.get(k) != getattr(self,k):
                raise ValueError('Ablation checkpoint mismatch: '+k)
        super().load_state_dict(state)
        self.task_momentum = None if state['task_momentum'] is None else state['task_momentum'].to(self.device)

    def inner_gap(self, losses, parameters):
        """Simplex FW gap of h(w)=||sum_i w_i grad f_i||_* on a fixed probe.

        gap=<G_w,P(G_w)>-min_i <g_i,P(G_w)>; exact polar P for spectral.
        This is a minibatch inner optimization diagnostic, not an accuracy gap
        and not a claim of full-dataset Pareto stationarity.
        """
        params = _unique_parameters(parameters)
        gradients = _task_gradient_matrix(losses, params, False).double()
        weights = self.weights.to(gradients)
        aggregate = (weights[:,None]*gradients).sum(0)
        direction = positive_oracle(aggregate, params, self.oracle, self.ns_steps, exact=True)
        scores = gradients @ direction
        gap = (weights @ scores) - scores.min()
        return float(gap.clamp_min(0)), float((aggregate*direction).sum())
