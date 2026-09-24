import os
import random
from pathlib import Path

import numpy as np
import torch


def capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    if not state:
        return

    if state.get("python") is not None:
        random.setstate(state["python"])

    if state.get("numpy") is not None:
        np.random.set_state(state["numpy"])

    torch_state = state.get("torch")

    if torch_state is not None:
        if torch.is_tensor(torch_state):
            torch_state = (
                torch_state
                .detach()
                .to(device="cpu", dtype=torch.uint8)
                .contiguous()
            )
        else:
            torch_state = torch.as_tensor(
                torch_state,
                dtype=torch.uint8,
                device="cpu",
            ).contiguous()

        torch.set_rng_state(torch_state)

    cuda_states = state.get("cuda")

    if torch.cuda.is_available() and cuda_states is not None:
        if torch.is_tensor(cuda_states):
            cuda_states = [cuda_states]

        normalized_cuda_states = []

        for cuda_state in cuda_states:
            if torch.is_tensor(cuda_state):
                cuda_state = (
                    cuda_state
                    .detach()
                    .to(device="cpu", dtype=torch.uint8)
                    .contiguous()
                )
            else:
                cuda_state = torch.as_tensor(
                    cuda_state,
                    dtype=torch.uint8,
                    device="cpu",
                ).contiguous()

            normalized_cuda_states.append(cuda_state)

        if normalized_cuda_states:
            torch.cuda.set_rng_state_all(normalized_cuda_states)


def atomic_torch_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_torch(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)
