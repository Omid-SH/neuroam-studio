"""Direct sparse assembly of the admittance system — no text netlist.

Physics (verified against netgen output, see docs/PLAN.md):

Each voxel of unit-size (sx, sy, sz) and per-axis resistivity rho contributes
12 edge resistors between its 8 corner nodes; along axis ``a``::

    R_a = 4 * rho_a * s_a / (s_b * s_c * dx)      [ohm]
    g_a = 1 / R_a = s_b * s_c * dx / (4 * rho_a * s_a)   [S]

Edges shared between voxels accumulate conductance in parallel.  The nodal
system is  G V = I  with ground nodes eliminated (V=0 reference).

Two assembly paths produce identical systems for uniform meshes:

- :func:`assemble_uniform` — vectorized over the full voxel grid; nodes are
  the (nx+1)(ny+1)(nz+1) unit-lattice corners.
- :func:`assemble_mrm` — vectorized over multiresolution ``.mrm`` records;
  only corner nodes referenced by some voxel exist (matches netgen).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse

from .model import VoxelModel


@dataclass
class System:
    """Assembled admittance system with node bookkeeping."""

    G: sparse.csr_matrix                 # reduced system (ground eliminated)
    node_coords: np.ndarray              # (n_all, 3) int lattice coords of ALL nodes
    keep: np.ndarray                     # indices (into all nodes) kept in G
    index_of: np.ndarray                 # all-node -> reduced index (-1 for ground)
    ground_all: np.ndarray               # all-node indices that are grounded
    source_rows: Dict[str, int]          # source name -> reduced row index
    world: Tuple[int, int, int]
    dx: float

    @property
    def n(self) -> int:
        return self.G.shape[0]

    def expand(self, v_reduced: np.ndarray) -> np.ndarray:
        """Reduced solution -> full node vector (ground nodes = 0)."""
        full = np.zeros(len(self.node_coords), dtype=v_reduced.dtype)
        full[self.keep] = v_reduced
        return full

    def node_grid(self, v_reduced: np.ndarray) -> np.ndarray:
        """Reduced solution -> dense (nx+1, ny+1, nz+1) node-voltage grid.

        For multiresolution systems, only existing nodes are filled; the rest
        are interpolation targets handled in :mod:`neuroam.fields`.
        """
        nx, ny, nz = self.world
        grid = np.zeros((nx + 1, ny + 1, nz + 1))
        full = self.expand(v_reduced)
        c = self.node_coords
        grid[c[:, 0], c[:, 1], c[:, 2]] = full
        return grid


# --------------------------------------------------------------------------- helpers
def _pack(coords: np.ndarray, world: Sequence[int]) -> np.ndarray:
    nx, ny, nz = world
    return (coords[..., 0].astype(np.int64) * (ny + 1)
            + coords[..., 1]) * (nz + 1) + coords[..., 2]


def _laplacian_from_edges(rows: np.ndarray, cols: np.ndarray, g: np.ndarray,
                          n: int) -> sparse.csr_matrix:
    """Symmetric graph Laplacian from edge lists (single direction given)."""
    data = np.concatenate([-g, -g])
    r = np.concatenate([rows, cols])
    c = np.concatenate([cols, rows])
    off = sparse.coo_matrix((data, (r, c)), shape=(n, n)).tocsr()
    diag = -np.asarray(off.sum(axis=1)).ravel()
    return (off + sparse.diags(diag)).tocsr()


def _reduce_and_index(G_all: sparse.csr_matrix, node_coords: np.ndarray,
                      packed: np.ndarray, model: VoxelModel,
                      world, dx, drop_isolated: bool = False) -> System:
    ground_packed = _pack(np.array(model.ground_nodes, dtype=np.int64).reshape(-1, 3),
                          world) if model.ground_nodes else np.array([], dtype=np.int64)
    ground_all = np.searchsorted(packed, ground_packed)
    valid = (ground_all < len(packed))
    ground_all = ground_all[valid]
    ground_all = ground_all[packed[ground_all] == ground_packed[valid]]
    if model.ground_nodes and len(ground_all) == 0:
        raise ValueError("no ground node coincides with an existing mesh node")

    n_all = len(node_coords)
    mask = np.ones(n_all, dtype=bool)
    mask[ground_all] = False
    if drop_isolated:
        # nodes with no incident conductance (e.g. lattice nodes inside void
        # regions) would make the system singular; the legacy mesher simply
        # never creates them.
        mask &= np.asarray(G_all.diagonal() != 0.0)
    keep = np.nonzero(mask)[0]
    index_of = np.full(n_all, -1, dtype=np.int64)
    index_of[keep] = np.arange(len(keep))

    G = G_all[keep][:, keep].tocsr()

    source_rows: Dict[str, int] = {}
    for s in model.sources:
        p = _pack(np.array(s.node, dtype=np.int64).reshape(1, 3), world)[0]
        j = np.searchsorted(packed, p)
        if j >= len(packed) or packed[j] != p:
            raise ValueError(f"source node {s.node} is not a mesh node")
        r = index_of[j]
        if r < 0:
            raise ValueError(f"source node {s.node} is grounded")
        source_rows[s.name] = int(r)

    return System(G=G, node_coords=node_coords, keep=keep, index_of=index_of,
                  ground_all=ground_all, source_rows=source_rows,
                  world=tuple(world), dx=dx)


# --------------------------------------------------------------------- uniform path
def assemble_uniform(model: VoxelModel,
                     void_materials: Optional[Sequence[int]] = None,
                     auto_void_undefined: bool = True) -> System:
    """Assemble G directly from the uniform voxel grid.

    ``void_materials`` (plus, by default, any label missing from the material
    library) are treated as unmeshed "outside world" — they contribute no
    conductance and their otherwise-isolated nodes are dropped, matching the
    legacy mesher's behavior of not meshing material-0 space.
    """
    nx, ny, nz = model.world
    dx = model.dx

    labels = model.labels
    void = np.zeros(labels.shape, dtype=bool)
    if void_materials is not None:
        void |= np.isin(labels, np.asarray(list(void_materials)))
    if auto_void_undefined:
        for u in np.unique(labels):
            if int(u) not in model.materials and \
                    model.materials.default_material is None:
                void |= labels == u
    any_void = bool(void.any())

    safe_labels = labels.copy()
    if any_void:
        defined = [m.id for m in model.materials]
        safe_labels[void] = defined[0]
    rx, ry, rz = model.materials.rho_arrays(safe_labels)
    if any_void:
        rx = np.where(void, np.inf, rx)
        ry = np.where(void, np.inf, ry)
        rz = np.where(void, np.inf, rz)

    # per-voxel per-axis edge conductance (each of the voxel's 4 edges on that axis)
    gx = dx / (4.0 * rx)
    gy = dx / (4.0 * ry)
    gz = dx / (4.0 * rz)

    # accumulate onto edge grids
    Gx = np.zeros((nx, ny + 1, nz + 1))
    Gy = np.zeros((nx + 1, ny, nz + 1))
    Gz = np.zeros((nx + 1, ny + 1, nz))
    for dj in (0, 1):
        for dk in (0, 1):
            Gx[:, dj:ny + dj, dk:nz + dk] += gx
            Gy[dj:nx + dj, :, dk:nz + dk] += gy
            Gz[dj:nx + dj, dk:ny + dk, :] += gz

    def node_id(x, y, z):
        return (x * (ny + 1) + y) * (nz + 1) + z

    # x-edges
    ex_i, ex_j, ex_k = np.meshgrid(np.arange(nx), np.arange(ny + 1),
                                   np.arange(nz + 1), indexing="ij")
    rows_x = node_id(ex_i, ex_j, ex_k).ravel()
    cols_x = node_id(ex_i + 1, ex_j, ex_k).ravel()
    # y-edges
    ey_i, ey_j, ey_k = np.meshgrid(np.arange(nx + 1), np.arange(ny),
                                   np.arange(nz + 1), indexing="ij")
    rows_y = node_id(ey_i, ey_j, ey_k).ravel()
    cols_y = node_id(ey_i, ey_j + 1, ey_k).ravel()
    # z-edges
    ez_i, ez_j, ez_k = np.meshgrid(np.arange(nx + 1), np.arange(ny + 1),
                                   np.arange(nz), indexing="ij")
    rows_z = node_id(ez_i, ez_j, ez_k).ravel()
    cols_z = node_id(ez_i, ez_j, ez_k + 1).ravel()

    rows = np.concatenate([rows_x, rows_y, rows_z])
    cols = np.concatenate([cols_x, cols_y, cols_z])
    g = np.concatenate([Gx.ravel(), Gy.ravel(), Gz.ravel()])

    n_all = (nx + 1) * (ny + 1) * (nz + 1)
    G_all = _laplacian_from_edges(rows, cols, g, n_all)

    xs, ys, zs = np.meshgrid(np.arange(nx + 1), np.arange(ny + 1),
                             np.arange(nz + 1), indexing="ij")
    node_coords = np.stack([xs.ravel(), ys.ravel(), zs.ravel()], axis=1)
    packed = _pack(node_coords, model.world)   # already sorted by construction

    return _reduce_and_index(G_all, node_coords, packed, model, model.world, dx,
                             drop_isolated=any_void)


# ---------------------------------------------------------------------- .mrm path
def read_mrm(path) -> np.ndarray:
    """Read an ``.mrm`` file -> int array (n, 7): x y z sx sy sz material."""
    rec = np.loadtxt(path, usecols=(1, 2, 3, 4, 5, 6, 7), dtype=np.int64, ndmin=2)
    return rec


def assemble_mrm(model: VoxelModel, records: np.ndarray) -> System:
    """Assemble G from multiresolution voxel records (netgen-equivalent)."""
    dx = model.dx
    world = model.world
    x, y, z = records[:, 0], records[:, 1], records[:, 2]
    sx, sy, sz = (records[:, 3].astype(float), records[:, 4].astype(float),
                  records[:, 5].astype(float))
    mat = records[:, 6]
    rx, ry, rz = model.materials.rho_arrays(mat)

    gx = (sy * sz * dx) / (4.0 * rx * sx)
    gy = (sx * sz * dx) / (4.0 * ry * sy)
    gz = (sx * sy * dx) / (4.0 * rz * sz)

    isx, isy, isz = records[:, 3], records[:, 4], records[:, 5]

    edges_r: List[np.ndarray] = []
    edges_c: List[np.ndarray] = []
    edges_g: List[np.ndarray] = []

    def corner(ddx, ddy, ddz):
        return np.stack([x + ddx * isx, y + ddy * isy, z + ddz * isz], axis=1)

    # 4 x-edges: from (0,dy,dz) to (1,dy,dz)
    for dy in (0, 1):
        for dz in (0, 1):
            edges_r.append(corner(0, dy, dz))
            edges_c.append(corner(1, dy, dz))
            edges_g.append(gx)
    # 4 y-edges
    for dxx in (0, 1):
        for dz in (0, 1):
            edges_r.append(corner(dxx, 0, dz))
            edges_c.append(corner(dxx, 1, dz))
            edges_g.append(gy)
    # 4 z-edges
    for dxx in (0, 1):
        for dy in (0, 1):
            edges_r.append(corner(dxx, dy, 0))
            edges_c.append(corner(dxx, dy, 1))
            edges_g.append(gz)

    A = np.concatenate(edges_r, axis=0)
    B = np.concatenate(edges_c, axis=0)
    g = np.concatenate(edges_g, axis=0)

    pa = _pack(A, world)
    pb = _pack(B, world)
    packed, inverse = np.unique(np.concatenate([pa, pb]), return_inverse=True)
    rows = inverse[: len(pa)]
    cols = inverse[len(pa):]

    nzp = (nzp1 := world[2] + 1)
    nyp = world[1] + 1
    zc = packed % nzp
    yc = (packed // nzp) % nyp
    xc = packed // (nzp * nyp)
    node_coords = np.stack([xc, yc, zc], axis=1).astype(np.int64)

    G_all = _laplacian_from_edges(rows, cols, g, len(packed))
    return _reduce_and_index(G_all, node_coords, packed, model, world, dx)


def assemble(model: VoxelModel, mrm_path=None) -> System:
    """Dispatch: uniform direct assembly, or .mrm-based multiresolution."""
    if mrm_path is None:
        return assemble_uniform(model)
    return assemble_mrm(model, read_mrm(mrm_path))
