from __future__ import annotations
import hashlib, json, os, pickle, random, tempfile
from pathlib import Path
from typing import Any
import numpy as np
import torch


def stable_seed(*parts: object) -> int:
    payload = ":".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32)


def hash_mapping(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def atomic_pickle_dump(value: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=destination.name+".", suffix=".tmp", dir=destination.parent)
    tmp = Path(name)
    try:
        with os.fdopen(fd, "wb") as h:
            pickle.dump(value, h, protocol=pickle.HIGHEST_PROTOCOL); h.flush(); os.fsync(h.fileno())
        os.replace(tmp, destination)
    finally:
        tmp.unlink(missing_ok=True)


def pickle_load(path: Path) -> Any:
    with path.open("rb") as h: return pickle.load(h)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda": torch.cuda.synchronize(device)
