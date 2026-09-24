from typing import Optional

import torch
import torch.nn.functional as F


def project_simplex(vector):
    if vector.ndim != 1:
        raise ValueError("simplex projection expects a one-dimensional tensor")
    values, _ = torch.sort(vector, descending=True)
    cumulative = torch.cumsum(values, dim=0) - 1.0
    indices = torch.arange(
        1,
        vector.numel() + 1,
        device=vector.device,
        dtype=vector.dtype,
    )
    condition = values - cumulative / indices > 0
    rho = torch.nonzero(condition, as_tuple=False)[-1, 0]
    theta = cumulative[rho] / indices[rho]
    return torch.clamp(vector - theta, min=0.0)


def _flatten_gradients(loss, parameters, retain_graph):
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    pieces = []
    for parameter, gradient in zip(parameters, gradients):
        if gradient is None:
            pieces.append(torch.zeros_like(parameter).reshape(-1))
        else:
            pieces.append(gradient.reshape(-1))
    return torch.cat(pieces)


def task_gradient_matrix(losses, parameters, retain_graph=True):
    return torch.stack(
        [
            _flatten_gradients(loss, parameters, retain_graph)
            for loss in losses
        ]
    )


def _gradient_list(loss, parameters, retain_graph):
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    return [
        torch.zeros_like(parameter) if gradient is None else gradient
        for parameter, gradient in zip(parameters, gradients)
    ]


def _assign_flat_gradient(parameters, gradient):
    offset = 0
    for parameter in parameters:
        count = parameter.numel()
        parameter.grad = gradient[offset : offset + count].view_as(parameter).clone()
        offset += count
    if offset != gradient.numel():
        raise RuntimeError("gradient vector has an unexpected size")


def _assign_gradient_list(parameters, gradients):
    for parameter, gradient in zip(parameters, gradients):
        parameter.grad = gradient.clone()


class MoCo:
    def __init__(
        self,
        n_tasks,
        device,
        beta=0.5,
        beta_sigma=0.5,
        gamma=0.1,
        gamma_sigma=0.5,
        rho=0.0,
    ):
        self.n_tasks = n_tasks
        self.device = device
        self.beta = beta
        self.beta_sigma = beta_sigma
        self.gamma = gamma
        self.gamma_sigma = gamma_sigma
        self.rho = rho
        self.step = 0
        self.y = None
        self.lambd = torch.full(
            (n_tasks,),
            1.0 / n_tasks,
            device=device,
            dtype=torch.float32,
        )

    def parameters(self):
        return []

    def state_dict(self):
        return {
            "step": self.step,
            "y": None if self.y is None else self.y.detach().cpu(),
            "lambd": self.lambd.detach().cpu(),
        }

    def load_state_dict(self, state):
        self.step = int(state["step"])
        self.y = None if state["y"] is None else state["y"].to(self.device)
        self.lambd = state["lambd"].to(self.device)

    def backward(
        self,
        losses,
        shared_parameters,
        task_specific_parameters,
        **kwargs,
    ):
        shared_parameters = list(shared_parameters)
        task_specific_parameters = list(task_specific_parameters)
        self.step += 1

        gradients = task_gradient_matrix(
            losses,
            shared_parameters,
            retain_graph=True,
        )

        specific_gradients = _gradient_list(
            losses.sum(),
            task_specific_parameters,
            retain_graph=False,
        )

        with torch.no_grad():
            normalized = gradients / (
                gradients.norm(dim=1, keepdim=True) + 1e-8
            )
            normalized = normalized * losses.detach().unsqueeze(1)

            if self.y is None:
                self.y = torch.zeros_like(normalized)

            beta_t = self.beta / (self.step ** self.beta_sigma)
            gamma_t = self.gamma / (self.step ** self.gamma_sigma)
            self.y = self.y - beta_t * (self.y - normalized)

            gram = self.y @ self.y.t()
            if self.rho != 0:
                gram = gram + self.rho * torch.eye(
                    self.n_tasks,
                    device=self.device,
                )

            self.lambd = F.softmax(
                self.lambd - gamma_t * (gram @ self.lambd),
                dim=-1,
            )
            shared_gradient = self.y.t() @ self.lambd

        _assign_flat_gradient(shared_parameters, shared_gradient)
        _assign_gradient_list(task_specific_parameters, specific_gradients)

        return None, {"weights": self.lambd.detach().clone()}


class MoDo:
    requires_three_batches = True

    def __init__(
        self,
        n_tasks,
        device,
        gamma=1e-3,
        rho=0.1,
    ):
        self.n_tasks = n_tasks
        self.device = device
        self.gamma = gamma
        self.rho = rho
        self.lambd = torch.full(
            (n_tasks,),
            1.0 / n_tasks,
            device=device,
            dtype=torch.float32,
        )

    def parameters(self):
        return []

    def state_dict(self):
        return {"lambd": self.lambd.detach().cpu()}

    def load_state_dict(self, state):
        self.lambd = state["lambd"].to(self.device)

    def backward(
        self,
        losses,
        shared_parameters,
        task_specific_parameters,
        **kwargs,
    ):
        if len(losses) != 3:
            raise ValueError("MoDo requires three independent mini-batches per update")

        shared_parameters = list(shared_parameters)
        task_specific_parameters = list(task_specific_parameters)
        gradient_matrices = []
        specific_sums = [
            torch.zeros_like(parameter)
            for parameter in task_specific_parameters
        ]

        for loss_vector in losses:
            gradient_matrices.append(
                task_gradient_matrix(
                    loss_vector,
                    shared_parameters,
                    retain_graph=True,
                )
            )
            specific = _gradient_list(
                loss_vector.sum(),
                task_specific_parameters,
                retain_graph=False,
            )
            for index, gradient in enumerate(specific):
                specific_sums[index] = specific_sums[index] + gradient / 3.0

        gradients_1, gradients_2, gradients_3 = gradient_matrices

        with torch.no_grad():
            weight_gradient = (
                gradients_1 @ (gradients_2.t() @ self.lambd)
                + self.rho * self.lambd
            )
            self.lambd = project_simplex(
                self.lambd - self.gamma * weight_gradient
            )
            shared_gradient = gradients_3.t() @ self.lambd

        _assign_flat_gradient(shared_parameters, shared_gradient)
        _assign_gradient_list(task_specific_parameters, specific_sums)

        return None, {"weights": self.lambd.detach().clone()}


class MGDAWarm:
    requires_three_batches = True

    def __init__(
        self,
        n_tasks,
        device,
        beta=0.5,
        rho=0.5,
        warm_steps=40,
        warm_beta: Optional[float] = None,
    ):
        self.n_tasks = n_tasks
        self.device = device
        self.beta = beta
        self.rho = rho
        self.warm_steps = warm_steps
        self.warm_beta = beta if warm_beta is None else warm_beta

        self.weights = torch.full(
            (n_tasks,),
            1.0 / n_tasks,
            device=device,
            dtype=torch.float32,
        )

        self.initialized = False
        self.step = 0

    def parameters(self):
        return []

    def state_dict(self):
        return {
            "weights": self.weights.detach().cpu(),
            "initialized": self.initialized,
            "step": self.step,
        }

    def load_state_dict(self, state):
        self.weights = state["weights"].to(
            device=self.device,
            dtype=torch.float32,
        )
        self.initialized = bool(state.get("initialized", False))
        self.step = int(state.get("step", 0))

    def _step(self, weights, gram, step_size):
        gradient = gram @ weights + self.rho * weights

        return project_simplex(
            weights - step_size * gradient
        )

    def warm_start(self, gram):
        if self.initialized:
            return

        if gram.ndim != 2:
            raise ValueError(
                "MGDA-warm Gram matrix must be two-dimensional"
            )

        if gram.shape != (self.n_tasks, self.n_tasks):
            raise ValueError(
                "MGDA-warm Gram matrix has an unexpected shape: "
                f"{tuple(gram.shape)}"
            )

        gram = gram.detach().to(
            device=self.device,
            dtype=self.weights.dtype,
        )

        with torch.no_grad():
            for _ in range(self.warm_steps):
                self.weights = self._step(
                    self.weights,
                    gram,
                    self.warm_beta,
                )

        self.initialized = True

        print(
            "MGDA-WARM INIT "
            f"weights={self.weights.detach().cpu().tolist()} "
            f"gram={gram.detach().cpu().tolist()}",
            flush=True,
        )

    def backward(
        self,
        losses,
        shared_parameters,
        **kwargs,
    ):
        if not self.initialized:
            raise RuntimeError(
                "MGDAWarm.warm_start() must be called before training"
            )

        if len(losses) != 3:
            raise ValueError(
                "MGDA-warm requires three independent mini-batches per update"
            )

        shared_parameters = list(shared_parameters)

        gradient_2 = task_gradient_matrix(
            losses[1],
            shared_parameters,
            retain_graph=True,
        )

        gradient_3 = task_gradient_matrix(
            losses[2],
            shared_parameters,
            retain_graph=True,
        )

        current_weights = self.weights.detach().clone()

        with torch.no_grad():
            weight_gradient = (
                gradient_2
                @ (gradient_3.t() @ current_weights)
                + self.rho * current_weights
            )

            next_weights = project_simplex(
                current_weights
                - self.beta * weight_gradient
            )

        weighted_loss = self.n_tasks * torch.dot(
            losses[0],
            current_weights,
        )

        weighted_loss.backward()

        self.weights = next_weights.detach()
        self.step += 1

        if self.step <= 10 or self.step % 100 == 0:
            print(
                "MGDA-WARM WEIGHTS "
                f"step={self.step} "
                f"weights={self.weights.detach().cpu().tolist()}",
                flush=True,
            )

        return weighted_loss, {
            "weights": current_weights.detach().clone()
        }