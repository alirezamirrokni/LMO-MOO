import torch


def zeropower_via_newtonschulz5(G, steps):
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True, matched_adam_rms=0.2):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4:
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= matched_adam_rms * max(update.size(-2), update.size(-1)) ** 0.5
    return update


def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    return (buf1 / (1 - betas[0] ** step)) / ((buf2 / (1 - betas[1] ** step)).sqrt() + eps)


class MuonWithAuxAdam(torch.optim.Optimizer):
    def __init__(self, param_groups):
        for group in param_groups:
            if group["use_muon"]:
                group.setdefault("momentum", 0.95)
            else:
                group.setdefault("betas", (0.9, 0.95))
                group.setdefault("eps", 1e-10)
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                state = self.state[p]
                if group["use_muon"]:
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                else:
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"], state["step"], group["betas"], group["eps"])
                state["last_update"] = update.detach().reshape_as(p)
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.reshape(p.shape), alpha=-group["lr"])


def muon_optimizer(model, lr, multimnist=False):
    muon, adam = [], []
    first_conv = None if multimnist else model.encoder_block[0][0].weight
    for name, p in model.named_parameters():
        if multimnist:
            use_muon = name.startswith("transformer_enc") and p.ndim >= 2
        else:
            use_muon = "pred" not in name and p.ndim >= 2 and p is not first_conv
        (muon if use_muon else adam).append(p)
    return MuonWithAuxAdam([
        dict(params=muon, use_muon=True, lr=lr, weight_decay=0.001),
        dict(params=adam, use_muon=False, lr=lr, betas=(0.9, 0.95), weight_decay=0.001),
    ])


def previous_update_scores(losses, parameters, optimizer):
    params = [p for p in parameters if p.requires_grad]
    gradients = [torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True) for loss in losses]
    delta = torch.zeros(len(losses), device=losses.device, dtype=losses.dtype)
    for j, p in enumerate(params):
        update = optimizer.state[p].get("last_update")
        if update is None:
            continue
        for i, grads in enumerate(gradients):
            if grads[j] is not None:
                delta[i] += (grads[j] * update).sum()
    return delta.detach()
