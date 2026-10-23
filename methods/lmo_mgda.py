import torch


def newton_schulz(matrix, steps=5):
    x = matrix.float()
    if x.shape[0] > x.shape[1]:
        x = x.transpose(0, 1)
    norm = torch.linalg.vector_norm(x)
    if float(norm.item()) <= 1e-30:
        return torch.zeros_like(matrix)
    x = x / (norm + 1.0e-7)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        aa = x @ x.transpose(0, 1)
        x = a * x + (b * aa + c * (aa @ aa)) @ x
    if matrix.shape[0] > matrix.shape[1]:
        x = x.transpose(0, 1)
    return x.to(dtype=matrix.dtype)


def spectral_oracle(buffer, parameters, ns_steps=5, exact=False):
    direction = torch.zeros_like(buffer)
    vector_ranges = []
    offset = 0
    for parameter in parameters:
        count = parameter.numel()
        if parameter.ndim >= 2:
            matrix = buffer[offset : offset + count].reshape(parameter.shape[0], -1)
            if not exact:
                direction[offset : offset + count] = newton_schulz(matrix, ns_steps).reshape(-1)
            elif torch.count_nonzero(matrix):
                u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
                keep = (s > torch.finfo(s.dtype).eps * max(matrix.shape) * s.max()).to(s.dtype)
                direction[offset : offset + count] = ((u * keep) @ vh).reshape(-1)
        else:
            vector_ranges.append((offset, offset + count))
        offset += count
    if vector_ranges:
        vector = torch.cat([buffer[a:b] for a, b in vector_ranges])
        if exact:
            vector = vector / vector.norm().clamp_min(1e-12)
        else:
            norm = torch.linalg.vector_norm(vector.float())
            vector = vector / norm.to(vector.dtype) if float(norm.item()) > 1e-12 else torch.zeros_like(vector)
        cursor = 0
        for a, b in vector_ranges:
            direction[a:b] = vector[cursor : cursor + b - a]
            cursor += b - a
    return direction


def oracle(buffer, parameters, geometry="spectral", ns_steps=5, exact=False):
    if geometry == "l2":
        return buffer / buffer.norm().clamp_min(1e-12)
    if geometry == "sign":
        return buffer.sign()
    return spectral_oracle(buffer, parameters, ns_steps, exact)


def kl_project_lower_bounded_simplex(probabilities, lower_bound):
    probabilities = probabilities / probabilities.sum()
    if lower_bound == 0.0:
        return probabilities
    n = probabilities.numel()
    free = torch.ones(n, dtype=torch.bool, device=probabilities.device)
    projected = torch.empty_like(probabilities)
    while True:
        candidate = (1.0 - (n - int(free.sum().item())) * lower_bound) / probabilities[free].sum() * probabilities
        newly_fixed = free & (candidate < lower_bound)
        if not bool(newly_fixed.any().item()):
            projected[~free] = lower_bound
            projected[free] = candidate[free]
            return projected / projected.sum()
        free = free & ~newly_fixed


def simplex_projection(v):
    u, _ = torch.sort(v, descending=True)
    cssv = torch.cumsum(u, 0) - 1
    idx = torch.arange(1, len(v) + 1, device=v.device, dtype=v.dtype)
    rho = torch.nonzero(u - cssv / idx > 0)[-1, 0]
    return (v - cssv[rho] / idx[rho]).clamp_min(0)


def task_gradients(losses, parameters):
    rows = []
    for i in range(len(losses)):
        grads = torch.autograd.grad(losses[i], parameters, retain_graph=i < len(losses) - 1, allow_unused=True)
        rows.append(torch.cat([(torch.zeros_like(p) if g is None else g.detach()).reshape(-1) for p, g in zip(parameters, grads)]))
    return torch.stack(rows)


def unique_parameters(parameters):
    seen, output = set(), []
    for p in parameters:
        if p.requires_grad and id(p) not in seen:
            seen.add(id(p))
            output.append(p)
    return output


class EntropicLMOMGDA:
    def __init__(self, n_tasks, device, eta=1e-4, alpha=0.1, ns_steps=5, tau=0.0, kappa=0.0,
                 oracle="spectral", weights="entropic", momentum="blended"):
        self.n_tasks = n_tasks
        self.device = device
        self.eta = eta
        self.alpha = alpha
        self.ns_steps = ns_steps
        self.tau = tau
        self.kappa = kappa
        self.oracle = oracle
        self.weight_update = weights
        self.momentum = momentum
        self.weights = torch.full((n_tasks,), 1.0 / n_tasks, dtype=torch.float64, device=device)
        self.blended_momentum = None
        self.task_momentum = None
        self.step_counter = 0

    def parameters(self):
        return []

    def _entropic_update(self, scores):
        log_proposal = (torch.log(self.weights) + self.eta * scores.to(device=self.device, dtype=torch.float64)) / (1.0 + self.eta * self.tau)
        proposal = torch.softmax(log_proposal, dim=0)
        if self.kappa > 0.0:
            proposal = kl_project_lower_bounded_simplex(proposal, self.kappa)
        proposal = torch.clamp(proposal, min=torch.finfo(proposal.dtype).tiny)
        self.weights = (proposal / proposal.sum()).detach()

    def backward(self, losses, model_parameters, **kwargs):
        parameters = unique_parameters(model_parameters)
        gradients = task_gradients(losses, parameters)
        if self.blended_momentum is None:
            self.blended_momentum = torch.zeros(gradients.shape[1], device=gradients.device, dtype=gradients.dtype)
        blend = self.weights.clone().to(gradients)
        aggregate = (blend[:, None] * gradients).sum(0)

        if self.momentum == "blended":
            buffer = self.blended_momentum
        elif self.momentum == "per-task":
            if self.task_momentum is None:
                self.task_momentum = torch.zeros_like(gradients)
            buffer = (blend[:, None] * self.task_momentum).sum(0)
        else:
            buffer = aggregate
        direction = oracle(buffer, parameters, self.oracle, self.ns_steps)
        scores = gradients @ (-direction)

        if self.weight_update == "entropic":
            self._entropic_update(scores)
        else:
            self.weights = simplex_projection(self.weights + self.eta * scores.to(self.weights)).detach()

        alpha = 1.0 if self.step_counter == 0 else self.alpha
        self.blended_momentum.mul_(1 - alpha).add_(aggregate, alpha=alpha)
        if self.momentum == "per-task":
            self.task_momentum.mul_(1 - alpha).add_(gradients, alpha=alpha)
        offset = 0
        for p in parameters:
            p.grad = direction[offset : offset + p.numel()].view_as(p).detach().clone()
            offset += p.numel()
        self.step_counter += 1
        return None, {}

    def inner_gap(self, losses, parameters):
        parameters = unique_parameters(parameters)
        gradients = task_gradients(losses, parameters).double()
        aggregate = (self.weights.to(gradients)[:, None] * gradients).sum(0)
        direction = oracle(aggregate, parameters, self.oracle, self.ns_steps, exact=True)
        scores = gradients @ direction
        gap = self.weights.to(gradients) @ scores - scores.min()
        return float(gap.clamp_min(0))
