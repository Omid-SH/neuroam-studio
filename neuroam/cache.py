"""Content-hash keyed result cache.

Every cached object is keyed by a SHA-256 over the exact inputs that affect
it (arrays hashed by bytes, parameters by canonical JSON).  Changing a
waveform therefore never invalidates a cached field solution; changing the
model or materials does.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np


def hash_inputs(*parts: Any) -> str:
    h = hashlib.sha256()
    for p in parts:
        if isinstance(p, np.ndarray):
            h.update(b"nd")
            h.update(str(p.dtype).encode())
            h.update(str(p.shape).encode())
            h.update(np.ascontiguousarray(p).tobytes())
        elif isinstance(p, (bytes, bytearray)):
            h.update(p)
        else:
            h.update(json.dumps(p, sort_keys=True, default=str).encode())
    return h.hexdigest()[:24]


class ResultCache:
    def __init__(self, root=".neuroam_cache", enabled: bool = True):
        self.root = Path(root)
        self.enabled = enabled
        if enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, kind: str, key: str) -> Path:
        return self.root / f"{kind}-{key}.npz"

    def get_array(self, kind: str, key: str) -> Optional[Dict[str, np.ndarray]]:
        if not self.enabled:
            return None
        p = self._path(kind, key)
        if p.exists():
            with np.load(p, allow_pickle=False) as z:
                return {k: z[k] for k in z.files}
        return None

    def put_array(self, kind: str, key: str, **arrays: np.ndarray) -> None:
        if not self.enabled:
            return
        np.savez_compressed(self._path(kind, key), **arrays)

    def get_or_compute(self, kind: str, key: str,
                       compute: Callable[[], Dict[str, np.ndarray]]
                       ) -> Dict[str, np.ndarray]:
        hit = self.get_array(kind, key)
        if hit is not None:
            return hit
        out = compute()
        self.put_array(kind, key, **out)
        return out
