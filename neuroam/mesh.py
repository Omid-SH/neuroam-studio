"""Multiresolution mesh (.mrm) utilities.

NeuroAM does not reimplement the clustering mesher; it consumes ``.mrm``
files produced by the lab's mesher (``mesher.exe`` / ``mesher_CARC`` or the
pure-Python ``Mesher/mesher.py`` from the unreleased repo).  This module
reads/writes the record format and can generate the trivial all-unit-voxel
mesh from a model (useful for equivalence tests and small models).

Record format (Cela reference, Appendix D)::

    <hex-ish id> x y z sx sy sz material
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .model import VoxelModel, node_name
from .assembly import read_mrm  # re-export for convenience  # noqa: F401


def uniform_records(model: VoxelModel,
                    skip_materials: Optional[list] = None) -> np.ndarray:
    """All-size-1 voxel records for a model — the uniform 'mesh'."""
    nx, ny, nz = model.world
    X, Y, Z = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz),
                          indexing="ij")
    mat = model.labels
    keep = np.ones(mat.shape, dtype=bool)
    if skip_materials:
        keep = ~np.isin(mat, np.asarray(skip_materials))
    rec = np.stack([X[keep], Y[keep], Z[keep],
                    np.ones(keep.sum(), dtype=np.int64),
                    np.ones(keep.sum(), dtype=np.int64),
                    np.ones(keep.sum(), dtype=np.int64),
                    mat[keep]], axis=1).astype(np.int64)
    return rec


def write_mrm(path, records: np.ndarray) -> Path:
    path = Path(path)
    with open(path, "w") as f:
        for x, y, z, sx, sy, sz, mat in records.tolist():
            f.write(f"{node_name(x, y, z)} {x} {y} {z} {sx} {sy} {sz} {mat}\n")
    return path


def mesh_stats(records: np.ndarray) -> dict:
    sizes = records[:, 3:6]
    vol = sizes.prod(axis=1)
    uniq, cnt = np.unique(sizes, axis=0, return_counts=True)
    return {
        "n_voxels": int(len(records)),
        "unit_voxels_covered": int(vol.sum()),
        "size_histogram": {f"{a}x{b}x{c}": int(n)
                           for (a, b, c), n in zip(uniq.tolist(), cnt.tolist())},
        "reduction_factor": float(vol.sum() / max(len(records), 1)),
    }
