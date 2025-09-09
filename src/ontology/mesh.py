from __future__ import annotations
from typing import List

def map_mesh(mesh_terms: List[str]) -> List[str]:
    # identity mapping for now (already MeSH terms)
    return list(dict.fromkeys(t.strip() for t in mesh_terms if t and isinstance(t, str)))
