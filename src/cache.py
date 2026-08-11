from __future__ import annotations
import hashlib, inspect, shutil
from pathlib import Path
from typing import Any
from src.util import atomic_pickle_dump, hash_mapping, pickle_load
from src.config import resolve_method_config

CACHE_SCHEMA = 2

def _source_hash(*objects: object) -> str:
    digest=hashlib.sha256(); seen=set()
    for obj in objects:
        try: source_file=inspect.getsourcefile(obj)
        except TypeError: source_file=None
        if source_file:
            p=Path(source_file).resolve()
            if p.exists() and str(p) not in seen:
                digest.update(p.name.encode()); digest.update(p.read_bytes()); seen.add(str(p)); continue
        try: digest.update(inspect.getsource(obj).encode())
        except (OSError, TypeError): digest.update(repr(obj).encode())
    return digest.hexdigest()

class CacheManager:
    def __init__(self, output_dir: Path, problem: str, config: dict[str, Any]):
        self.output_dir=output_dir; self.problem=problem; self.config=config
        self.root=output_dir/"cache"; self.root.mkdir(parents=True, exist_ok=True)
    def clear_all_methods(self) -> None:
        if self.root.exists(): shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
    def path(self, method: str, dataset: str, seed: int) -> Path:
        safe=method.replace("/","_").replace(" ","_")
        return self.root/safe/f"{dataset}_seed{seed}.pkl"
    def fingerprint(self, method: str, dataset: str, seed: int, source_objects: tuple[object,...]=()) -> str:
        # Selection-only fields must not invalidate unrelated completed runs.  In
        # particular, adding/removing another method or another seed should leave
        # this (method,dataset,seed) cache reusable.  True training semantics such
        # as epochs, averaging window, determinism, and device remain fingerprinted.
        experiment=dict(self.config.get("experiment",{}))
        for key in ("methods","datasets","seeds","num_seeds","seed"):
            experiment.pop(key,None)
        data_cfg=self.config.get("data",{}).get(dataset,{})
        relevant={"schema":CACHE_SCHEMA,"problem":self.problem,"dataset":dataset,"seed":seed,
                  "method":method,"method_config":resolve_method_config(self.config,method,dataset),
                  "experiment":experiment,"data":data_cfg,
                  "model":self.config.get("model",{}),"source":_source_hash(*source_objects) if source_objects else ""}
        return hash_mapping(relevant)
    def load(self, method: str, dataset: str, seed: int, fingerprint: str) -> dict[str,Any]|None:
        p=self.path(method,dataset,seed)
        if not p.exists(): return None
        payload=pickle_load(p)
        if payload.get("schema")!=CACHE_SCHEMA or payload.get("fingerprint")!=fingerprint: return None
        return payload
    def save(self, method: str, dataset: str, seed: int, fingerprint: str, state: dict[str,Any], *, complete: bool) -> None:
        atomic_pickle_dump({"schema":CACHE_SCHEMA,"fingerprint":fingerprint,"complete":bool(complete),"state":state}, self.path(method,dataset,seed))
