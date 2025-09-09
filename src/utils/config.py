from __future__ import annotations
import os, yaml
from typing import Any, Dict

def _env_expand(v: Any) -> Any:
    if isinstance(v, str):
        return os.path.expandvars(v)
    if isinstance(v, dict):
        return {k: _env_expand(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_env_expand(x) for x in v]
    return v

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return _env_expand(data)
