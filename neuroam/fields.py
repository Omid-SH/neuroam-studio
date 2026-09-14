"""Field post-processing: node grids, E and J, exports, ROI dose metrics."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .assembly import System
from .model import VoxelModel, node_name


# ------------------------------------------------------------------ grids
def node_grid(system: System, v_reduced: np.ndarray) -> np.ndarray:
    """(nx+1, ny+1, nz+1) node-voltage grid (ground = 0)."""
    return system.node_grid(v_reduced)


def _trilinear_weights(sx: int, sy: int, sz: int) -> np.ndarray:
    """Local ``(sx+1, sy+1, sz+1, 8)`` trilinear shape-function weights for a
    block of unit-voxel size ``(sx, sy, sz)``.

    Corner order is ``(dx, dy, dz)`` in ``{0,1}^3``, dx-major — must match the
    corner-gather order in :func:`_fill_block_group`.
    """
    lx, ly, lz = np.arange(sx + 1) / sx, np.arange(sy + 1) / sy, np.arange(sz + 1) / sz
    X, Y, Z = np.meshgrid(lx, ly, lz, indexing="ij")
    w = []
    for dx in (0, 1):
        wx = X if dx else 1.0 - X
        for dy in (0, 1):
            wy = Y if dy else 1.0 - Y
            for dz in (0, 1):
                wz = Z if dz else 1.0 - Z
                w.append(wx * wy * wz)
    return np.stack(w, axis=-1)  # (sx+1, sy+1, sz+1, 8)


def _fill_block_group(records: np.ndarray, grid: np.ndarray,
                      ny1: int, nz1: int) -> Tuple[np.ndarray, np.ndarray]:
    """Flat lattice indices and trilinearly-interpolated values for every
    lattice point (corners included) inside every block of ``records`` —
    ``records`` must all share one ``(sx, sy, sz)``."""
    x, y, z = records[:, 0], records[:, 1], records[:, 2]
    sx, sy, sz = int(records[0, 3]), int(records[0, 4]), int(records[0, 5])
    n = len(records)

    corner_vals = np.empty((n, 8))
    ci = 0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                corner_vals[:, ci] = grid[x + dx * sx, y + dy * sy, z + dz * sz]
                ci += 1

    w = _trilinear_weights(sx, sy, sz).reshape(-1, 8)              # (npts, 8)
    ox, oy, oz = np.meshgrid(np.arange(sx + 1), np.arange(sy + 1),
                             np.arange(sz + 1), indexing="ij")
    ox, oy, oz = ox.ravel(), oy.ravel(), oz.ravel()                 # (npts,)

    vals = corner_vals @ w.T                                        # (n, npts)
    gx = (x[:, None] + ox[None, :]).ravel()
    gy = (y[:, None] + oy[None, :]).ravel()
    gz = (z[:, None] + oz[None, :]).ravel()
    idx = (gx * ny1 + gy) * nz1 + gz
    return idx, vals.ravel()


def fill_hanging_nodes(grid: np.ndarray, system: System) -> np.ndarray:
    """For multires systems, fill lattice nodes that are not mesh nodes.

    Every ``.mrm`` block is a trilinear, 8-corner resistor element (12 edge
    resistors between corners — see :mod:`neuroam.assembly`), so the
    potential inside a block is, by construction, the trilinear
    interpolation of its own 8 solved corner values. This is the same "final
    interpolation" the lab's legacy PAM toolchain uses to bring a
    multiresolution solve back onto the model's original full (unit-voxel)
    resolution — exact for this element, not an approximation (unlike the
    neighbor-averaging this replaced). Mesh nodes are never modified.

    Blocks are grouped by size — typically a handful of distinct sizes in a
    real mesh — and every group's corner-gather + trilinear matmul (the
    expensive part, and independent of every other block/group) runs
    concurrently in its own thread; numpy releases the GIL for these, so
    this scales with core count without copying the (often multi-GB) grid
    between processes. A real mesh node is never overwritten by another
    block's interpolated estimate: at a fine/coarse boundary a coarse
    block's face-interior point can numerically coincide with a node that a
    neighboring finer block already solved for exactly (a T-junction), so
    every group's contribution to an already-real node is dropped rather
    than raced. Where two blocks of matching size share a face of otherwise-
    hanging points, both derive the same value from the same shared corner
    nodes, so duplicate contributions agree and are simply averaged with
    themselves; the (rare, only possible with >1-level size jumps at one
    interface) case of two differently-sized blocks disagreeing about a
    still-hanging point is also resolved by averaging, not by whichever
    thread happens to finish last.
    """
    nx1, ny1, nz1 = grid.shape
    records = system.records
    if records is None or len(records) == 0:
        return grid

    defined = np.zeros(grid.shape, dtype=bool)
    c = system.node_coords
    defined[c[:, 0], c[:, 1], c[:, 2]] = True
    if defined.all():
        return grid
    defined_flat = defined.ravel()

    sizes = records[:, 3:6]
    uniq_sizes = np.unique(sizes, axis=0)

    def work(size_key: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mask = np.all(sizes == size_key, axis=1)
        idx, vals = _fill_block_group(records[mask], grid, ny1, nz1)
        keep = ~defined_flat[idx]
        return idx[keep], vals[keep]

    workers = min(len(uniq_sizes), os.cpu_count() or 1)
    if workers <= 1:
        results = [work(s) for s in uniq_sizes]
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(work, uniq_sizes))

    n = nx1 * ny1 * nz1
    idx_all = (np.concatenate([r[0] for r in results])
              if results else np.empty(0, dtype=np.int64))
    val_all = (np.concatenate([r[1] for r in results])
              if results else np.empty(0))
    acc = np.bincount(idx_all, weights=val_all, minlength=n)
    cnt = np.bincount(idx_all, minlength=n)

    out = grid.ravel().copy()
    hit = cnt > 0
    out[hit] = acc[hit] / cnt[hit]
    return out.reshape(grid.shape)


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
    Returns (Ex, Ey, Ez, ``|E|``) with shape (nx, ny, nz), in V/m.
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
    """Legacy .vof: one row per mesh node, 'name amplitude [phase]'.

    Vectorized (numpy string ops + one bulk write) rather than a per-row
    Python loop: at multires-mesh scale (tens of millions of nodes for a
    full-head model) the naive per-row ``f"{...}"`` + ``file.write()`` loop
    this replaced took long enough on its own to matter.
    """
    path = Path(path)
    full = system.expand(v_reduced)
    c = system.node_coords
    is_ground = np.zeros(len(c), dtype=bool)
    is_ground[system.ground_all] = True

    names = np.char.add(np.char.add(
        np.char.zfill(c[:, 0].astype(str), 4),
        np.char.zfill(c[:, 1].astype(str), 4)),
        np.char.zfill(c[:, 2].astype(str), 4))
    names = np.where(is_ground, "0", names)
    vals = np.char.mod("%.10g", full)
    lines = np.char.add(np.char.add(names, " "), vals)
    path.write_text("\n".join(lines.tolist()) + "\n")
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
