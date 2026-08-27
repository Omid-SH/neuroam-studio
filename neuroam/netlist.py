"""Legacy ``.net`` (SPICE-subset) reading and netgen-equivalent writing.

Used for cross-validation against the historical toolchain and to keep old
meshes runnable.  The reader replicates the parsing semantics of
``AM_v10.2_res_multi.py`` (Res lines summed as parallel conductances,
ground node named ``0`` eliminated, ``Cap`` entries ignored in the resistive
solver) but is vectorized and returns the same :class:`~neuroam.assembly.System`
structure as direct assembly.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse

from .assembly import System, _laplacian_from_edges, _pack
from .model import VoxelModel, parse_node_name, node_name


def write_net(model: VoxelModel, records: np.ndarray, path,
              cur_names: Optional[List[str]] = None) -> Path:
    """Write a netgen-equivalent ``.net`` for the given multires records.

    ``records`` is the (n, 7) int array ``x y z sx sy sz material`` (an
    all-ones-size record set reproduces the uniform mesh).  Resistor values
    follow  R_a = 4 rho_a s_a / (s_b s_c dx).
    """
    path = Path(path)
    dx = model.dx
    grounds = model.ground_nodes or []
    lines: List[str] = ["* NeuroAM netgen-equivalent file",
                        f"* param infile {model.name}.in"]
    for gn in grounds:
        lines.append(f"* param nodename {node_name(*gn)} 0")
    lines.append("")
    lines.append("* Start of network")
    for i, s in enumerate(model.sources):
        cur = cur_names[i] if cur_names else s.name
        lines.append(f" I {cur} {node_name(*s.node)} 0")
    lines.append("")

    gset = set(grounds)

    def nm(x, y, z):
        return "0" if (x, y, z) in gset else node_name(x, y, z)

    out = [ "\n".join(lines) ]
    chunk: List[str] = []
    for rec in records:
        x, y, z, sx, sy, sz, mat = (int(v) for v in rec)
        rho = model.materials[mat].rho
        Rx = 4.0 * rho[0] * sx / (sy * sz * dx)
        Ry = 4.0 * rho[1] * sy / (sx * sz * dx)
        Rz = 4.0 * rho[2] * sz / (sx * sy * dx)
        chunk.append(f"* voxel {x} {y} {z} {sx} {sy} {sz} {mat}")
        for dy in (0, 1):
            for dz in (0, 1):
                chunk.append(f"Res {nm(x, y + dy * sy, z + dz * sz)} "
                             f"{nm(x + sx, y + dy * sy, z + dz * sz)} {Rx:g}")
        for dxx in (0, 1):
            for dz in (0, 1):
                chunk.append(f"Res {nm(x + dxx * sx, y, z + dz * sz)} "
                             f"{nm(x + dxx * sx, y + sy, z + dz * sz)} {Ry:g}")
        for dxx in (0, 1):
            for dy in (0, 1):
                chunk.append(f"Res {nm(x + dxx * sx, y + dy * sy, z)} "
                             f"{nm(x + dxx * sx, y + dy * sy, z + sz)} {Rz:g}")
        chunk.append("")
    out.append("\n".join(chunk))
    path.write_text("\n".join(out))
    return path


def read_net(path, model: VoxelModel) -> System:
    """Parse a legacy ``.net`` into a reduced admittance System."""
    node_a: List[str] = []
    node_b: List[str] = []
    values: List[float] = []
    grounds: List[Tuple[int, int, int]] = []
    sources: List[Tuple[str, str]] = []          # (cur name, node name)
    n_cap = 0

    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "Res":
                node_a.append(parts[1]); node_b.append(parts[2])
                values.append(float(parts[3]))
            elif parts[0] == "Cap":
                n_cap += 1
            elif parts[0] == "*" and len(parts) >= 5 and parts[1] == "param" \
                    and parts[2] == "nodename" and parts[-1] == "0":
                grounds.append(parse_node_name(parts[3]))
            elif parts[0] == "I" and len(parts) >= 4:
                sources.append((parts[1], parts[2]))

    if n_cap:
        warnings.warn(f"{n_cap} Cap entries ignored (resistive solve)")

    # ground node in Res lines appears literally as "0"
    g_txt = node_name(*grounds[0]) if grounds else None

    def to_coords(names: List[str]) -> np.ndarray:
        arr = np.empty((len(names), 3), dtype=np.int64)
        for i, nm in enumerate(names):
            if nm == "0":
                if g_txt is None:
                    raise ValueError(".net references ground '0' but no "
                                     "'* param nodename ... 0' header found")
                arr[i] = grounds[0]
            else:
                arr[i] = parse_node_name(nm)
        return arr

    A = to_coords(node_a)
    B = to_coords(node_b)
    g = 1.0 / np.asarray(values)

    world = model.world
    pa = _pack(A, world)
    pb = _pack(B, world)
    packed, inverse = np.unique(np.concatenate([pa, pb]), return_inverse=True)
    rows = inverse[: len(pa)]
    cols = inverse[len(pa):]

    nzp = world[2] + 1
    nyp = world[1] + 1
    zc = packed % nzp
    yc = (packed // nzp) % nyp
    xc = packed // (nzp * nyp)
    node_coords = np.stack([xc, yc, zc], axis=1).astype(np.int64)

    G_all = _laplacian_from_edges(rows, cols, g, len(packed))

    # ground / source bookkeeping mirrors assembly._reduce_and_index but the
    # net file's own declarations take precedence over the model's.
    tmp = VoxelModel(labels=model.labels, dx=model.dx, materials=model.materials,
                     sources=list(model.sources), ground_nodes=grounds or
                     list(model.ground_nodes), name=model.name)
    if sources:
        from .model import Source
        tmp.sources = [Source(name=c, node=parse_node_name(n)) for c, n in sources]
    from .assembly import _reduce_and_index
    return _reduce_and_index(G_all, node_coords, packed, tmp, world, model.dx)
