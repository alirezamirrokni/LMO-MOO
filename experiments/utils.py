import random

import numpy as np
import torch


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device(gpu=0):
    return torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")


def capture_rng_state():
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def method_state(method):
    state = {}
    for k, v in vars(method).items():
        if isinstance(v, torch.optim.Optimizer):
            state[k] = v.state_dict()
        elif isinstance(v, torch.Tensor):
            state[k] = v.detach()
        elif isinstance(v, (np.ndarray, int, float, bool)):
            state[k] = v
    return state


def load_method_state(method, state):
    for k, v in state.items():
        current = getattr(method, k)
        if isinstance(current, torch.optim.Optimizer):
            current.load_state_dict(v)
        elif isinstance(current, torch.Tensor) and current.requires_grad:
            with torch.no_grad():
                current.copy_(v.to(current.device))
        else:
            setattr(method, k, v.to(method.device) if isinstance(v, torch.Tensor) else v)
    if hasattr(method, "_init_optim_problem") and method.step > 0:
        method._init_optim_problem()
