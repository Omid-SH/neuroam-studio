"""Field-to-neuron coupling: sampling at compartment coordinates and legacy
coordinates / ``.v`` file formats.

Legacy conventions (from ``interp3``/``get_coords.hoc``/``Makewaveform2.py``):

- coordinates file: one ``x y z`` per line, in unit-voxel (node lattice)
  coordinates; sub-voxel positions are floats.
- unit-field file (interp3 output): a single line of space-separated values,
  one per compartment — the static field sampled at the compartments.
- ``.v`` waveform file (Makewaveform2 output): T rows x n_compartments columns,
  space separated; NEURON's Stim hoc plays column *i* into compartment *i*'s
  ``e_extracellular``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
from scipy.ndimage import map_coordinates

from .waveforms import Waveform


# ------------------------------------------------------------- coordinates IO
def read_coordinates(path) -> np.ndarray:
    """(n, 3) float array of compartment coordinates in unit-voxel units."""
    arr = np.loadtxt(path, ndmin=2)
    if arr.shape[1] < 3:
        raise ValueError(f"{path}: expected 3 columns")
    return arr[:, :3].astype(float)


def write_coordinates(path, coords: np.ndarray) -> Path:
    np.savetxt(path, np.asarray(coords), fmt="%.6g")
    return Path(path)


def transform_coordinates(coords: np.ndarray,
                          translate=(0.0, 0.0, 0.0),
                          rotate_deg: Optional[Sequence[float]] = None,
                          pivot: Optional[Sequence[float]] = None,
                          scale: float = 1.0) -> np.ndarray:
    """Rigid placement of a morphology in the field domain (voxel units).

    ``rotate_deg`` = (rx, ry, rz) intrinsic rotations applied in x, y, z
    order about ``pivot`` (default: centroid).
    """
    c = np.asarray(coords, dtype=float) * scale
    if rotate_deg is not None:
        rx, ry, rz = np.deg2rad(rotate_deg)
        p = np.asarray(pivot, dtype=float) if pivot is not None else c.mean(axis=0)
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(rx), -np.sin(rx)],
                       [0, np.sin(rx), np.cos(rx)]])
        Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                       [0, 1, 0],
                       [-np.sin(ry), 0, np.cos(ry)]])
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                       [np.sin(rz), np.cos(rz), 0],
                       [0, 0, 1]])
        c = (c - p) @ (Rz @ Ry @ Rx).T + p
    return c + np.asarray(translate, dtype=float)


# ----------------------------------------------------------------- sampling
def sample_node_grid(grid: np.ndarray, coords: np.ndarray,
                     outside: str = "clip") -> np.ndarray:
    """Trilinear interpolation of node voltages at coordinates.

    Equivalent to the legacy ``interp3`` (Interpolate1D chain) on a uniform
    node grid.  ``outside='clip'`` clamps to the domain; ``'nan'`` marks
    out-of-domain compartments (detect neurons leaving the field volume).
    """
    c = np.asarray(coords, dtype=float).T          # (3, n)
    out_of = ((c[0] < 0) | (c[0] > grid.shape[0] - 1)
              | (c[1] < 0) | (c[1] > grid.shape[1] - 1)
              | (c[2] < 0) | (c[2] > grid.shape[2] - 1))
    vals = map_coordinates(grid, c, order=1, mode="nearest")
    if outside == "nan":
        vals = np.where(out_of, np.nan, vals)
    return vals


def interpolation_report(grid_shape, coords: np.ndarray) -> Dict[str, int]:
    c = np.asarray(coords)
    inside = np.all((c >= 0) & (c <= np.asarray(grid_shape) - 1), axis=1)
    return {"n_compartments": int(len(c)),
            "n_outside_domain": int((~inside).sum())}


# ----------------------------------------------------------------- .v files
def write_unit_field(path, values: np.ndarray) -> Path:
    """interp3-style single-line unit field (one value per compartment)."""
    Path(path).write_text(" ".join(f"{v:.6g}" for v in np.asarray(values)) + "\n")
    return Path(path)


def read_unit_field(path) -> np.ndarray:
    return np.fromstring(Path(path).read_text(), sep=" ")


def build_v_matrix(unit_values: np.ndarray, waveform: Waveform,
                   unit_scale: float = 1.0) -> np.ndarray:
    """(T, n_compartments) extracellular potential matrix.

    ``unit_values`` is the field at compartments for a 1 A source;
    the waveform samples (A) scale it per time step.  ``unit_scale``
    converts units (e.g. 1e3 for mV if the field is in volts).
    """
    return np.outer(waveform.samples, np.asarray(unit_values) * unit_scale)


def superpose_v_matrix(unit_fields: Dict[str, np.ndarray],
                       waveforms: Dict[str, Waveform],
                       unit_scale: float = 1.0) -> np.ndarray:
    names = list(unit_fields)
    T = max(len(waveforms[n].samples) for n in names)
    n = len(next(iter(unit_fields.values())))
    out = np.zeros((T, n))
    for nm in names:
        s = waveforms[nm].samples
        out[: len(s)] += np.outer(s, unit_fields[nm] * unit_scale)
    return out


def write_v_file(path, v_matrix: np.ndarray) -> Path:
    """Makewaveform2-compatible .v file (rows = time, cols = compartments)."""
    np.savetxt(path, v_matrix, fmt="%.6g", delimiter=" ")
    return Path(path)


def read_v_file(path) -> np.ndarray:
    return np.loadtxt(path, ndmin=2)
