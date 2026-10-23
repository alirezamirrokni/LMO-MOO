import torch
import torch.nn.functional as F


def project_simplex(vector):
    values, _ = torch.sort(vector, descending=True)
    cumulative = torch.cumsum(values, dim=0) - 1.0
    indices = torch.arange(1, vector.numel() + 1, device=vector.device, dtype=vector.dtype)
    rho = torch.nonzero(values - cumulative / indices > 0, as_tuple=False)[-1, 0]
    return torch.clamp(vector - cumulative[rho] / indices[rho], min=0.0)


def gradient_list(loss, parameters, retain_graph):
    gradients = torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=True)
    return [torch.zeros_like(p) if g is None else g for p, g in zip(parameters, gradients)]


def gradient_matrix(losses, parameters):
    return torch.stack([torch.cat([g.reshape(-1) for g in gradient_list(loss, parameters, True)]) for loss in losses])


def assign_flat_gradient(parameters, gradient):
    offset = 0
    for p in parameters:
        p.grad = gradient[offset : offset + p.numel()].view_as(p).clone()
        offset += p.numel()


def assign_gradient_list(parameters, gradients):
    for p, g in zip(parameters, gradients):
        p.grad = g.clone()


class MoCo:
    def __init__(self, n_tasks, device, beta=0.5, beta_sigma=0.5, gamma=0.1, gamma_sigma=0.5, rho=0.0):
        self.n_tasks = n_tasks
        self.device = device
        self.beta, self.beta_sigma, self.gamma, self.gamma_sigma, self.rho = beta, beta_sigma, gamma, gamma_sigma, rho
        self.step = 0
        self.y = None
        self.lambd = torch.full((n_tasks,), 1.0 / n_tasks, device=device, dtype=torch.float32)

    def parameters(self):
        return []

    def backward(self, losses, shared_parameters, task_specific_parameters, **kwargs):
        shared_parameters = list(shared_parameters)
        task_specific_parameters = list(task_specific_parameters)
        self.step += 1
        gradients = gradient_matrix(losses, shared_parameters)
        specific = gradient_list(losses.sum(), task_specific_parameters, retain_graph=False)
        with torch.no_grad():
            normalized = gradients / (gradients.norm(dim=1, keepdim=True) + 1e-8) * losses.detach().unsqueeze(1)
            if self.y is None:
                self.y = torch.zeros_like(normalized)
            self.y = self.y - self.beta / self.step ** self.beta_sigma * (self.y - normalized)
            gram = self.y @ self.y.t()
            if self.rho != 0:
                gram = gram + self.rho * torch.eye(self.n_tasks, device=self.device)
            self.lambd = F.softmax(self.lambd - self.gamma / self.step ** self.gamma_sigma * (gram @ self.lambd), dim=-1)
            shared_gradient = self.y.t() @ self.lambd
        assign_flat_gradient(shared_parameters, shared_gradient)
        assign_gradient_list(task_specific_parameters, specific)
        return None, {}


class MoDo:
    def __init__(self, n_tasks, device, gamma=1e-3, rho=0.1):
        self.n_tasks = n_tasks
        self.device = device
        self.gamma, self.rho = gamma, rho
        self.lambd = torch.full((n_tasks,), 1.0 / n_tasks, device=device, dtype=torch.float32)

    def parameters(self):
        return []

    def backward(self, losses, shared_parameters, task_specific_parameters, **kwargs):
        shared_parameters = list(shared_parameters)
        task_specific_parameters = list(task_specific_parameters)
        matrices = []
        specific = [torch.zeros_like(p) for p in task_specific_parameters]
        for loss_vector in losses:
            matrices.append(gradient_matrix(loss_vector, shared_parameters))
            for i, g in enumerate(gradient_list(loss_vector.sum(), task_specific_parameters, retain_graph=False)):
                specific[i] = specific[i] + g / 3.0
        g1, g2, g3 = matrices
        with torch.no_grad():
            self.lambd = project_simplex(self.lambd - self.gamma * (g1 @ (g2.t() @ self.lambd) + self.rho * self.lambd))
            shared_gradient = g3.t() @ self.lambd
        assign_flat_gradient(shared_parameters, shared_gradient)
        assign_gradient_list(task_specific_parameters, specific)
        return None, {}


class MGDAWarm:
    def __init__(self, n_tasks, device, beta=0.5, rho=0.5, warm_steps=40, warm_beta=None):
        self.n_tasks = n_tasks
        self.device = device
        self.beta, self.rho, self.warm_steps = beta, rho, warm_steps
        self.warm_beta = beta if warm_beta is None else warm_beta
        self.weights = torch.full((n_tasks,), 1.0 / n_tasks, device=device, dtype=torch.float32)
        self.initialized = False
        self.step = 0

    def parameters(self):
        return []

    def _step(self, weights, gram, step_size):
        return project_simplex(weights - step_size * (gram @ weights + self.rho * weights))

    def warm_start(self, gram):
        gram = gram.detach().to(device=self.device, dtype=self.weights.dtype)
        with torch.no_grad():
            for _ in range(self.warm_steps):
                self.weights = self._step(self.weights, gram, self.warm_beta)
        self.initialized = True

    def backward(self, losses, shared_parameters, **kwargs):
        shared_parameters = list(shared_parameters)
        g2 = gradient_matrix(losses[1], shared_parameters)
        g3 = gradient_matrix(losses[2], shared_parameters)
        current = self.weights.detach().clone()
        with torch.no_grad():
            next_weights = project_simplex(current - self.beta * (g2 @ (g3.t() @ current) + self.rho * current))
        (self.n_tasks * torch.dot(losses[0], current)).backward()
        self.weights = next_weights.detach()
        self.step += 1
        return None, {}
