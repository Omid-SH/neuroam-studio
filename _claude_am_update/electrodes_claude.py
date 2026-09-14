"""Parametric electrode geometry: build, place, fit, and compare.

Two jobs:

1. **Build** electrodes from parameters so a montage can be re-placed on any
   model (another species, another resolution) rather than hand-painted once:
   spherical bands (contact-lens ring / corneal cap), swept tubes (needle,
   hooked "J" lead, wire), discs, rings, boxes, cylinders, and conformal
   surface patches.
2. **Fit** those parameters back out of an existing labelled model, so an
   electrode that already exists as voxels becomes a reusable spec — and so a
   newly generated electrode can be checked against the original
   voxel-for-voxel (:func:`dice`).

All coordinates are in **voxel units**; voxel *centres* sit at index + 0.5.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ==========================================================================
# shape rasterizers  (all return a boolean mask of the model's shape)
# ==========================================================================
def _centres(shape, sl=None):
    """Voxel-centre coordinate grids for a volume (or a sub-box)."""
    if sl is None:
        rngs = [np.arange(s) + 0.5 for s in shape]
    else:
        rngs = [np.arange(s.start, s.stop) + 0.5 for s in sl]
    return np.meshgrid(*rngs, indexing="ij")


def _bbox_slices(lo, hi, shape) -> Tuple[slice, slice, slice]:
    lo = np.floor(np.asarray(lo)).astype(int)
    hi = np.ceil(np.asarray(hi)).astype(int) + 1
    lo = np.clip(lo, 0, np.asarray(shape))
    hi = np.clip(hi, 0, np.asarray(shape))
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def spherical_band(shape, center, r_in: float, r_out: float,
                   axis: Optional[Sequence[float]] = None,
                   theta_min_deg: float = 0.0,
                   theta_max_deg: float = 180.0) -> np.ndarray:
    """Spherical shell restricted to an angular band about ``axis``.

    ``theta`` is measured from ``axis``.  ``theta_min=0`` gives a **cap**
    (disc-like contact lens); a narrow band around 90° gives a **ring**
    (limbal / annular contact-lens electrode).
    """
    center = np.asarray(center, dtype=float)
    sl = _bbox_slices(center - r_out - 1, center + r_out + 1, shape)
    X, Y, Z = _centres(shape, sl)
    d = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2 + (Z - center[2]) ** 2)
    m = (d >= r_in) & (d <= r_out)
    if axis is not None and (theta_min_deg > 0 or theta_max_deg < 180):
        u = np.asarray(axis, dtype=float)
        u = u / np.linalg.norm(u)
        cos = ((X - center[0]) * u[0] + (Y - center[1]) * u[1]
               + (Z - center[2]) * u[2]) / np.maximum(d, 1e-12)
        cos = np.clip(cos, -1.0, 1.0)
        ang = np.degrees(np.arccos(cos))
        m &= (ang >= theta_min_deg) & (ang <= theta_max_deg)
    out = np.zeros(shape, dtype=bool)
    out[sl] = m
    return out


def tube(shape, path: Sequence[Sequence[float]], radius: float,
         capped: bool = True, section: str = "round") -> np.ndarray:
    """Solid tube swept along a polyline — needle, wire, hooked "J" lead.

    ``section='round'`` gives a circular cross-section (radius = half the
    diameter); ``'square'`` gives a square one of half-width ``radius``, which
    is how hand-drawn voxel leads usually look.
    """
    P = np.asarray(path, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3 or len(P) < 2:
        raise ValueError("path must be (n>=2, 3)")
    out = np.zeros(shape, dtype=bool)
    for a, b in zip(P[:-1], P[1:]):
        lo = np.minimum(a, b) - radius - 1
        hi = np.maximum(a, b) + radius + 1
        sl = _bbox_slices(lo, hi, shape)
        if any(s.stop <= s.start for s in sl):
            continue
        X, Y, Z = _centres(shape, sl)
        ab = b - a
        L2 = float(ab @ ab)
        if L2 == 0:
            t = np.zeros(X.shape)
        else:
            t = ((X - a[0]) * ab[0] + (Y - a[1]) * ab[1] + (Z - a[2]) * ab[2]) / L2
            t = np.clip(t, 0.0, 1.0) if capped else t
        dx = X - (a[0] + t * ab[0])
        dy = Y - (a[1] + t * ab[1])
        dz = Z - (a[2] + t * ab[2])
        if section == "square":
            out[sl] |= (np.maximum(np.maximum(np.abs(dx), np.abs(dy)),
                                   np.abs(dz)) <= radius)
        else:
            out[sl] |= (dx * dx + dy * dy + dz * dz) <= radius * radius
    return out


def tube_shell(shape, path, r_in: float, r_out: float,
               section: str = "round") -> np.ndarray:
    """Annular sheath around a tube — insulation over a lead."""
    return (tube(shape, path, r_out, section=section)
            & ~tube(shape, path, r_in, section=section))


def cylinder(shape, center, radius: float, height: float,
             axis: str = "z") -> np.ndarray:
    c = np.asarray(center, dtype=float)
    u = np.zeros(3); u["xyz".index(axis)] = 1.0
    return tube(shape, [c - u * height / 2, c + u * height / 2], radius)


def sphere(shape, center, radius: float) -> np.ndarray:
    return spherical_band(shape, center, 0.0, radius)


def box(shape, corner, size) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    c = np.asarray(corner, int); s = np.asarray(size, int)
    out[c[0]:c[0] + s[0], c[1]:c[1] + s[1], c[2]:c[2] + s[2]] = True
    return out


def disc(shape, center, radius: float, thickness: float,
         normal=(0, 0, 1)) -> np.ndarray:
    """Flat disc/plate electrode of given normal."""
    n = np.asarray(normal, float); n /= np.linalg.norm(n)
    c = np.asarray(center, float)
    return tube(shape, [c - n * thickness / 2, c + n * thickness / 2], radius)


def ring(shape, center, r_in: float, r_out: float, thickness: float,
         normal=(0, 0, 1)) -> np.ndarray:
    """Flat annulus (ring electrode)."""
    return (disc(shape, center, r_out, thickness, normal)
            & ~disc(shape, center, r_in, thickness, normal))


def surface_patch(labels: np.ndarray, tissue_ids: Sequence[int],
                  center, radius: float, depth: float = 2.0,
                  outside_ids: Optional[Sequence[int]] = None) -> np.ndarray:
    """A patch conforming to a tissue surface — cuff/scalp/plate electrodes.

    Selects voxels within ``radius`` of ``center`` that lie within ``depth``
    of the boundary of ``tissue_ids`` (optionally on the ``outside_ids`` side).
    """
    from scipy import ndimage
    tis = np.isin(labels, np.asarray(list(tissue_ids)))
    if outside_ids is not None:
        out = np.isin(labels, np.asarray(list(outside_ids)))
    else:
        out = ~tis
    dist = ndimage.distance_transform_edt(out)
    near = out & (dist <= depth) & ndimage.binary_dilation(tis, iterations=int(np.ceil(depth)))
    ball = sphere(labels.shape, center, radius)
    return near & ball


# ==========================================================================
# specs & placement
# ==========================================================================
@dataclass
class ElectrodeSpec:
    """A placeable electrode: geometry + electrical role + optional sheath."""

    name: str
    role: str                                   # 'source' | 'ground'
    material: int                               # metal label id
    shape: str                                  # rasterizer name
    params: Dict                                # rasterizer kwargs
    insulation: Optional[Dict] = None           # {'material': 110, 'shape':..., 'params':...}
    waveform: Optional[str] = None
    node: Optional[Tuple[int, int, int]] = None  # injection node (else auto)
    grid: int = 1                                # authoring lattice (see rasterize)

    def rasterize(self, shape) -> np.ndarray:
        return rasterize(self.shape, shape, grid=self.grid, **self.params)

    def rasterize_insulation(self, shape) -> Optional[np.ndarray]:
        if not self.insulation:
            return None
        ins = dict(self.insulation)
        ins.pop("material", None)
        name = ins.pop("shape")
        g = int(ins.pop("grid", self.grid))
        return rasterize(name, shape, grid=g, **ins.get("params", ins))

    def to_dict(self) -> Dict:
        d = {"name": self.name, "role": self.role, "material": self.material,
             "shape": self.shape, "params": _jsonable(self.params),
             "grid": self.grid}
        if self.insulation:
            d["insulation"] = _jsonable(self.insulation)
        if self.waveform:
            d["waveform"] = self.waveform
        if self.node:
            d["node"] = list(self.node)
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "ElectrodeSpec":
        d = dict(d)
        node = d.pop("node", None)
        return cls(node=tuple(node) if node else None, **d)


_RASTER = {
    "spherical_band": spherical_band, "tube": tube, "tube_shell": tube_shell,
    "cylinder": cylinder, "sphere": sphere, "box": box, "disc": disc,
    "ring": ring,
}

#: parameters that carry a length and therefore rescale with the raster grid
_SCALE_KEYS = {
    "spherical_band": ("center", "r_in", "r_out"),
    "tube": ("path", "radius"),
    "tube_shell": ("path", "r_in", "r_out"),
    "cylinder": ("center", "radius", "height"),
    "sphere": ("center", "radius"),
    "box": ("corner", "size"),
    "disc": ("center", "radius", "thickness"),
    "ring": ("center", "r_in", "r_out", "thickness"),
}


def rasterize(shape_name: str, volume_shape, grid: int = 1, **params
              ) -> np.ndarray:
    """Rasterize a named shape, optionally on a coarser lattice.

    ``grid=n`` builds the shape on an n-times coarser voxel lattice and
    expands it back, so every ``n×n×n`` block is uniformly filled.  Lab models
    are frequently authored this way (e.g. electrodes drawn at 166 µm inside an
    83 µm model); matching the authoring lattice is what makes a generated
    electrode reproduce a hand-built one voxel-for-voxel.
    """
    fn = _RASTER[shape_name]
    if grid is None or grid <= 1:
        return fn(volume_shape, **params)
    g = int(grid)
    cshape = tuple(int(np.ceil(s / g)) for s in volume_shape)
    p = dict(params)
    for k in _SCALE_KEYS.get(shape_name, ()):
        if k in p and p[k] is not None:
            v = np.asarray(p[k], dtype=float) / g
            p[k] = v.tolist() if v.ndim else float(v)
    if shape_name == "box":
        p["corner"] = np.floor(np.asarray(p["corner"], float)).astype(int)
        p["size"] = np.maximum(1, np.round(np.asarray(p["size"], float)).astype(int))
    m = fn(cshape, **p)
    out = np.repeat(np.repeat(np.repeat(m, g, 0), g, 1), g, 2)
    return out[:volume_shape[0], :volume_shape[1], :volume_shape[2]]


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def place(model, specs: Sequence[ElectrodeSpec], protect: Sequence[int] = (),
          register: bool = True) -> Dict[str, np.ndarray]:
    """Paint electrodes (and sheaths) into ``model.labels`` and register nodes.

    Metal is painted first and never overwritten by another electrode's
    insulation.  ``protect`` lists material ids that must not be overpainted.
    Returns the boolean mask actually painted per electrode name.
    """
    from .materials import Material, METAL_RHO, INSULATOR_RHO
    from .model import Source

    labels = model.labels
    shape = labels.shape
    protected = np.isin(labels, np.asarray(list(protect))) if protect else None
    masks: Dict[str, np.ndarray] = {}

    metal_all = np.zeros(shape, dtype=bool)
    for sp in specs:
        m = sp.rasterize(shape)
        if protected is not None:
            m &= ~protected
        masks[sp.name] = m
        metal_all |= m

    for sp in specs:                                  # metal
        m = masks[sp.name]
        if sp.material not in model.materials:
            model.materials.add(Material.isotropic_rho(
                sp.material, METAL_RHO, name=f"{sp.name} ({sp.role})"))
        labels[m] = sp.material

    for sp in specs:                                  # insulation
        ins = sp.rasterize_insulation(shape)
        if ins is None:
            continue
        ins &= ~metal_all
        if protected is not None:
            ins &= ~protected
        mid = int(sp.insulation.get("material", 110))
        if mid not in model.materials:
            model.materials.add(Material.isotropic_rho(
                mid, INSULATOR_RHO, name=f"{sp.name} insulation"))
        labels[ins] = mid
        masks[sp.name + ":insulation"] = ins

    if register:
        for sp in specs:
            node = sp.node
            if node is None:
                idx = np.argwhere(masks[sp.name])
                if len(idx) == 0:
                    raise ValueError(f"electrode {sp.name!r} painted no voxels")
                node = tuple(int(v) for v in idx[len(idx) // 2])
            node = tuple(int(v) for v in node)
            if sp.role == "source":
                model.sources.append(Source(name=sp.waveform or sp.name,
                                            node=node))
            else:
                model.ground_nodes.append(node)
    return masks


# ==========================================================================
# fitting from an existing labelled model
# ==========================================================================
def voxels_of(labels: np.ndarray, mat: int) -> np.ndarray:
    """Indices of a material's voxels, (n, 3) int."""
    return np.argwhere(labels == mat)


def dice(a: np.ndarray, b: np.ndarray) -> float:
    """Sørensen–Dice overlap of two boolean masks (1.0 = identical)."""
    a = a.astype(bool); b = b.astype(bool)
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else float(2.0 * (a & b).sum() / s)


def fit_sphere(points_center: np.ndarray) -> Tuple[np.ndarray, float]:
    """Algebraic sphere fit to voxel-centre coordinates."""
    A = np.c_[2 * points_center, np.ones(len(points_center))]
    b = (points_center ** 2).sum(1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = sol[:3]
    return c, float(np.sqrt(sol[3] + (c ** 2).sum()))


def fit_spherical_band(voxels: np.ndarray,
                       axis_hint: Optional[Sequence[float]] = None
                       ) -> Dict:
    """Recover (center, r_in, r_out, axis, theta_min, theta_max) from voxels.

    ``axis_hint`` fixes the band axis (e.g. globe-centre → cornea-centre);
    without it the axis is taken from the band centroid, which is only
    meaningful for a cap, not a ring.
    """
    P = voxels + 0.5
    c, r = fit_sphere(P)
    rad = np.linalg.norm(P - c, axis=1)
    if axis_hint is not None:
        u = np.asarray(axis_hint, float)
    else:
        u = P.mean(0) - c
    u = u / np.linalg.norm(u)
    cos = np.clip(((P - c) @ u) / np.maximum(rad, 1e-12), -1, 1)
    ang = np.degrees(np.arccos(cos))
    return {"center": c.tolist(), "r_in": float(rad.min()),
            "r_out": float(rad.max()), "axis": u.tolist(),
            "theta_min_deg": float(ang.min()), "theta_max_deg": float(ang.max()),
            "_r_fit": r, "_n": int(len(voxels))}


def fit_tube_path(voxels: np.ndarray, simplify_tol: float = 0.8,
                  max_nodes: int = 40, bin_width: float = 2.0) -> Dict:
    """Recover a centre-line path and radius from a tube-like voxel set.

    Builds a 26-connectivity graph over the voxels (bridging any disconnected
    pieces), finds the two geodesic extremes, then takes the centroid of each
    geodesic-distance shell as a centre-line node.  This tracks curved leads —
    the hooked "J" — that no single cylinder or straight-line fit describes.
    """
    from scipy import sparse
    from scipy.sparse import csgraph
    from scipy.spatial import cKDTree

    P = voxels.astype(float) + 0.5          # voxel centres, model coordinates

    # 26-connectivity graph, weighted by euclidean distance
    tree = cKDTree(P)
    pairs = np.array(sorted(tree.query_pairs(r=np.sqrt(3) + 1e-9)))
    if len(pairs) == 0:
        raise ValueError("cannot build a path: skeleton has no adjacent points")
    w = np.linalg.norm(P[pairs[:, 0]] - P[pairs[:, 1]], axis=1)
    n = len(P)
    A = sparse.coo_matrix((np.r_[w, w], (np.r_[pairs[:, 0], pairs[:, 1]],
                                         np.r_[pairs[:, 1], pairs[:, 0]])),
                          shape=(n, n)).tocsr()

    # skeletons of curved tubes often fragment; bridge components along their
    # closest points so the walk can traverse the whole lead
    ncomp, lab = csgraph.connected_components(A, directed=False)
    if ncomp > 1:
        extra_r, extra_c, extra_w = [], [], []
        groups = [np.nonzero(lab == k)[0] for k in range(ncomp)]
        trees = [cKDTree(P[g]) for g in groups]
        for i in range(ncomp):
            for j in range(i + 1, ncomp):
                d, idx = trees[i].query(P[groups[j]])
                k = int(np.argmin(d))
                a_, b_ = int(groups[i][idx[k]]), int(groups[j][k])
                extra_r += [a_, b_]; extra_c += [b_, a_]
                extra_w += [float(d[k])] * 2
        A = (A + sparse.coo_matrix((extra_w, (extra_r, extra_c)),
                                   shape=(n, n)).tocsr())

    def farthest(src: int):
        d = csgraph.dijkstra(A, directed=False, indices=src)
        finite = np.isfinite(d)
        return int(np.argmax(np.where(finite, d, -1))), d

    # the two geodesic extremes of the lead
    a, _ = farthest(0)
    b, d_from_a = farthest(a)

    # Centre-line = centroid of each geodesic-distance shell.  This tracks a
    # curved lead correctly (a straight-line fit or per-slice centroid would
    # not) and needs no skeletonization.
    d = d_from_a[np.isfinite(d_from_a)]
    dv = np.where(np.isfinite(d_from_a), d_from_a, np.nan)
    span = float(np.nanmax(dv))
    nbins = max(2, int(np.ceil(span / max(bin_width, 1e-6))) + 1)
    edges = np.linspace(0, span + 1e-9, nbins + 1)
    which = np.digitize(dv, edges) - 1
    nodes = []
    for k in range(nbins):
        sel = which == k
        if sel.any():
            nodes.append(P[sel].mean(0))
    path = np.asarray(nodes)
    if len(path) < 2:
        path = np.vstack([P[a], P[b]])
    # bin centroids stop about half a bin short of each tip; anchor the ends
    # on the true geodesic extremes so the lead keeps its full length
    if np.linalg.norm(path[0] - P[a]) > 1e-9:
        path = np.vstack([P[a], path])
    if np.linalg.norm(path[-1] - P[b]) > 1e-9:
        path = np.vstack([path, P[b]])

    path = _rdp(path, simplify_tol)
    if len(path) > max_nodes:
        keep = np.linspace(0, len(path) - 1, max_nodes).astype(int)
        path = path[keep]

    # radius from cross-sectional area: n_vox ≈ π r² · length
    length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
    radius = float(np.sqrt(max(len(voxels), 1) / (np.pi * max(length, 1e-9))))
    return {"path": path.tolist(), "radius": radius, "_length": length,
            "_n": int(len(voxels))}


def _rdp(points: np.ndarray, tol: float) -> np.ndarray:
    """Ramer–Douglas–Peucker polyline simplification."""
    if len(points) < 3:
        return points
    a, b = points[0], points[-1]
    ab = b - a
    L = np.linalg.norm(ab)
    if L < 1e-9:
        d = np.linalg.norm(points - a, axis=1)
    else:
        t = np.clip(((points - a) @ ab) / (L * L), 0, 1)
        proj = a + t[:, None] * ab
        d = np.linalg.norm(points - proj, axis=1)
    i = int(np.argmax(d))
    if d[i] <= tol:
        return np.vstack([a, b])
    return np.vstack([_rdp(points[:i + 1], tol)[:-1], _rdp(points[i:], tol)])


def refine_by_dice(reference: np.ndarray, shape_name: str, params: Dict,
                   sweep: Dict[str, Sequence[float]],
                   volume_shape, grid: int = 1, passes: int = 2
                   ) -> Tuple[Dict, float]:
    """Coordinate-descent refinement of parameters to maximise Dice.

    ``sweep`` maps a parameter name to candidate offsets added to the current
    value.  Returns the best parameters and the Dice achieved.
    """
    best = dict(params)
    best_d = dice(reference, rasterize(shape_name, volume_shape, grid=grid, **best))
    for _ in range(max(1, passes)):
        improved = False
        for key, deltas in sweep.items():
            if key not in best:
                continue
            base = best[key]
            for d in deltas:
                trial = dict(best)
                trial[key] = base + d
                try:
                    sc = dice(reference, rasterize(shape_name, volume_shape,
                                                   grid=grid, **trial))
                except Exception:
                    continue
                if sc > best_d + 1e-12:
                    best_d, best, improved = sc, trial, True
        if not improved:
            break
    return best, best_d


def compare_masks(a: np.ndarray, b: np.ndarray, name: str = "") -> Dict:
    """Geometric agreement report between a generated and a reference mask."""
    a = a.astype(bool); b = b.astype(bool)
    inter = int((a & b).sum())
    return {
        "name": name,
        "generated_voxels": int(a.sum()),
        "reference_voxels": int(b.sum()),
        "intersection": inter,
        "dice": dice(a, b),
        "jaccard": float(inter / max((a | b).sum(), 1)),
        "missed_reference": int((b & ~a).sum()),
        "extra_generated": int((a & ~b).sum()),
    }
