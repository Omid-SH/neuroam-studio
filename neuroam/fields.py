"""Field post-processing: node grids, E and J, exports, ROI dose metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .assembly import System
from .model import VoxelModel, node_name


# ------------------------------------------------------------------ grids
def node_grid(system: System, v_reduced: np.ndarray) -> np.ndarray:
    """(nx+1, ny+1, nz+1) node-voltage grid (ground = 0)."""
    return system.node_grid(v_reduced)


def fill_hanging_nodes(grid: np.ndarray, system: System) -> np.ndarray:
    """For multires systems, fill lattice nodes that are not mesh nodes.

    Missing values are filled by iterative local averaging of defined
    neighbors — adequate for visualization and voxel-average export.  Mesh
    nodes are never modified.
    """
    nx1, ny1, nz1 = grid.shape
    defined = np.zeros(grid.shape, dtype=bool)
    c = system.node_coords
    defined[c[:, 0], c[:, 1], c[:, 2]] = True
    if defined.all():
        return grid
    out = grid.copy()
    todo = ~defined
    for _ in range(max(grid.shape)):
        if not todo.any():
            break
        acc = np.zeros_like(out)
        cnt = np.zeros(out.shape)
        known = ~todo
        for axis in range(3):
            for shift in (1, -1):
                k = np.roll(known, shift, axis=axis)
                v = np.roll(out, shift, axis=axis)
                # roll wraps; mask the wrapped border
                sl = [slice(None)] * 3
                sl[axis] = 0 if shift == 1 else -1
                k = k.copy()
                k[tuple(sl)] = False
                acc += np.where(k, v, 0.0)
                cnt += k
        newly = todo & (cnt > 0)
        out[newly] = acc[newly] / cnt[newly]
        todo = todo & ~newly
    return out


def voxel_average(grid: np.ndarray) -> np.ndarray:
    """Voxel-center potential = mean of the 8 corner nodes (legacy 'vavg')."""
    g = grid
    return 0.125 * (g[:-1, :-1, :-1] + g[1:, :-1, :-1] + g[:-1, 1:, :-1]
                    + g[:-1, :-1, 1:] + g[1:, 1:, :-1] + g[1:, :-1, 1:]
                    + g[:-1, 1:, 1:] + g[1:, 1:, 1:])


def efield(grid: np.ndarray, dx: float) -> Tuple[np.ndarray, np.ndarray,
                                                 np.ndarray, np.ndarray]:
    """E at voxel centers from corner potentials: E = -grad V.

    Each component averages the 4 parallel edge differences of the voxel.
    Returns (Ex, Ey, Ez, |E|) with shape (nx, ny, nz), in V/m.
    """
    g = grid
    ex = (g[1:, :, :] - g[:-1, :, :]) / dx          # on x-edges
    Ex = -0.25 * (ex[:, :-1, :-1] + ex[:, 1:, :-1] + ex[:, :-1, 1:] + ex[:, 1:, 1:])
    ey = (g[:, 1:, :] - g[:, :-1, :]) / dx
    Ey = -0.25 * (ey[:-1, :, :-1] + ey[1:, :, :-1] + ey[:-1, :, 1:] + ey[1:, :, 1:])
    ez = (g[:, :, 1:] - g[:, :, :-1]) / dx
    Ez = -0.25 * (ez[:-1, :-1, :] + ez[1:, :-1, :] + ez[:-1, 1:, :] + ez[1:, 1:, :])
    return Ex, Ey, Ez, np.sqrt(Ex ** 2 + Ey ** 2 + Ez ** 2)


def current_density(model: VoxelModel, Ex, Ey, Ez):
    """J = sigma E per voxel (A/m^2); anisotropy honored per axis."""
    rx, ry, rz = model.materials.rho_arrays(model.labels)
    Jx, Jy, Jz = Ex / rx, Ey / ry, Ez / rz
    return Jx, Jy, Jz, np.sqrt(Jx ** 2 + Jy ** 2 + Jz ** 2)


# ------------------------------------------------------------------ exports
def write_vof(path, system: System, v_reduced: np.ndarray) -> Path:
    """Legacy .vof: one row per mesh node, 'name amplitude [phase]'."""
    path = Path(path)
    full = system.expand(v_reduced)
    c = system.node_coords
    gset = set(map(tuple, c[system.ground_all]))
    with open(path, "w") as f:
        for (x, y, z), val in zip(c.tolist(), full.tolist()):
            nm = "0" if (x, y, z) in gset else node_name(x, y, z)
            f.write(f"{nm} {val:.10g}\n")
    return path


def read_vof(path, world: Sequence[int]) -> np.ndarray:
    """Read a legacy .vof into a dense (nx+1, ny+1, nz+1) grid (NaN = absent)."""
    nx, ny, nz = world
    grid = np.full((nx + 1, ny + 1, nz + 1), np.nan)
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2 or parts[0] == "0":
                continue
            nm = parts[0]
            if len(nm) != 12 or not nm.isdigit():
                continue
            x, y, z = int(nm[0:4]), int(nm[4:8]), int(nm[8:12])
            grid[x, y, z] = float(parts[1])
    return grid


def write_vavg(path, vavg: np.ndarray) -> Path:
    """Voxel-average potentials, .model-compatible layout (nz blocks of ny x nx)."""
    nx, ny, nz = vavg.shape
    m2d = vavg.transpose(2, 1, 0).reshape(nz * ny, nx)
    np.savetxt(path, m2d, fmt="%.8g")
    return Path(path)


def write_raw(path, arr: np.ndarray) -> Path:
    """float32 raw volume (ParaView-importable), legacy '_V.raw' style."""
    np.ascontiguousarray(arr.transpose(2, 1, 0), dtype=np.float32).tofile(path)
    return Path(path)


# ------------------------------------------------------------------ ROI metrics
def roi_metrics(mask: np.ndarray, Emag: np.ndarray, Jmag: np.ndarray,
                name: str = "roi") -> Dict[str, float]:
    m = mask.astype(bool)
    if not m.any():
        return {"name": name, "voxels": 0}
    return {
        "name": name,
        "voxels": int(m.sum()),
        "E_mean_V_per_m": float(Emag[m].mean()),
        "E_median_V_per_m": float(np.median(Emag[m])),
        "E_p95_V_per_m": float(np.percentile(Emag[m], 95)),
        "E_max_V_per_m": float(Emag[m].max()),
        "J_mean_A_per_m2": float(Jmag[m].mean()),
        "J_p95_A_per_m2": float(np.percentile(Jmag[m], 95)),
        "J_max_A_per_m2": float(Jmag[m].max()),
    }


def mask_from_labels(model: VoxelModel, label_ids: Sequence[int]) -> np.ndarray:
    return np.isin(model.labels, np.asarray(label_ids))


def mask_sphere_shell(world: Sequence[int], center, r_in: float, r_out: float,
                      cone_axis: Optional[Sequence[float]] = None,
                      cone_half_angle_deg: Optional[float] = None) -> np.ndarray:
    """Spherical shell mask (optionally capped to a cone) — the retina-mask
    pattern used in the lab's ``calculate_retina_opticnerve_EF.m``."""
    nx, ny, nz = world
    X, Y, Z = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz),
                          indexing="ij")
    cx, cy, cz = center
    vx, vy, vz = X - cx, Y - cy, Z - cz
    d2 = vx ** 2 + vy ** 2 + vz ** 2
    mask = (d2 >= r_in ** 2) & (d2 <= r_out ** 2)
    if cone_axis is not None and cone_half_angle_deg is not None:
        u = np.asarray(cone_axis, dtype=float)
        u /= np.linalg.norm(u)
        cosang = (vx * u[0] + vy * u[1] + vz * u[2]) / (np.sqrt(d2) + 1e-12)
        mask &= cosang >= np.cos(np.deg2rad(cone_half_angle_deg))
    return mask
