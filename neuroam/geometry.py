"""Physical-units geometry for electrode design.

Electrodes are described as *regions* in millimetres, independent of the voxel
grid they will be rasterized onto, so one spec is reusable across the 166 /
83 / 41.5 um models and across species.

Two kinds of region compose freely with ``|`` (union), ``&`` (intersection)
and ``-`` (difference):

- :class:`SDF` -- an analytic signed-distance function (sphere, torus,
  spherical-shell band, needle, cuff, swept polyline, ...).  Cheap, exact,
  rotatable, and resolution-independent.
- :class:`MaskRegion` -- a boolean voxel mask, used to bring anatomy into the
  expression (e.g. "the shell just outside the sclera"), see
  :mod:`neuroam.frames`.

Rasterization is always restricted to the region's bounding box and
supersampled, so painting a 150 um ring into a 912x900x504 model touches only
a few thousand voxels' worth of arithmetic.

Conventions
-----------
Voxel ``(i, j, k)`` spans lattice nodes ``i..i+1`` etc., so its **centre** is
at ``(i + 0.5) * dx``.  :class:`Grid` implements that mapping; all region
maths is in millimetres.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "Grid", "Region", "SDF", "MaskRegion", "rasterize",
    "Sphere", "Box", "Cylinder", "Capsule", "Cone", "Torus", "Disc",
    "Annulus", "SphericalShell", "SphericalCap", "SphericalBand",
    "Needle", "Cuff", "Polyline", "Everything", "Nothing",
    "rotation_to", "rotation_axis_angle",
]

_EPS = 1e-12


# --------------------------------------------------------------------- grid
@dataclass(frozen=True)
class Grid:
    """Voxel lattice: shape, isotropic spacing, and world origin (all mm)."""

    shape: Tuple[int, int, int]
    dx_mm: float
    origin_mm: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    @classmethod
    def from_model(cls, model) -> "Grid":
        return cls(tuple(int(s) for s in model.labels.shape), model.dx * 1e3)

    def centers_mm(self, idx: np.ndarray) -> np.ndarray:
        """Voxel-centre coordinates (mm) for integer indices ``(..., 3)``."""
        return (np.asarray(idx, dtype=float) + 0.5) * self.dx_mm + np.asarray(self.origin_mm)

    def nodes_mm(self, idx: np.ndarray) -> np.ndarray:
        """Lattice-node coordinates (mm) for integer node indices."""
        return np.asarray(idx, dtype=float) * self.dx_mm + np.asarray(self.origin_mm)

    def index_of_mm(self, p_mm) -> np.ndarray:
        """Nearest voxel index containing the point."""
        p = (np.asarray(p_mm, dtype=float) - np.asarray(self.origin_mm)) / self.dx_mm
        return np.floor(p).astype(np.int64)

    def clip_box(self, lo_mm, hi_mm, pad: int = 1):
        """Integer index box (inclusive lo, exclusive hi) covering an AABB."""
        lo = self.index_of_mm(lo_mm) - pad
        hi = self.index_of_mm(hi_mm) + 1 + pad
        lo = np.maximum(lo, 0)
        hi = np.minimum(hi, np.asarray(self.shape))
        return lo, hi


# ------------------------------------------------------------------ regions
class Region:
    """Base class: something that can say which points are inside it."""

    # -- interface --------------------------------------------------------
    def contains(self, P_mm: np.ndarray) -> np.ndarray:      # pragma: no cover
        raise NotImplementedError

    def bounds_mm(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """World AABB, or ``None`` if unbounded."""
        return None

    def describe(self) -> dict:
        return {"type": type(self).__name__}

    # -- boolean algebra --------------------------------------------------
    def __or__(self, other: "Region") -> "Region":
        return Union(self, other)

    def __and__(self, other: "Region") -> "Region":
        return Intersect(self, other)

    def __sub__(self, other: "Region") -> "Region":
        return Difference(self, other)

    def __invert__(self) -> "Region":
        return Complement(self)


def _aabb_union(a, b):
    if a is None or b is None:
        return None
    return np.minimum(a[0], b[0]), np.maximum(a[1], b[1])


def _aabb_intersect(a, b):
    if a is None:
        return b
    if b is None:
        return a
    lo = np.maximum(a[0], b[0])
    hi = np.minimum(a[1], b[1])
    return lo, hi


class Union(Region):
    def __init__(self, a: Region, b: Region):
        self.a, self.b = a, b

    def contains(self, P):
        return self.a.contains(P) | self.b.contains(P)

    def bounds_mm(self):
        return _aabb_union(self.a.bounds_mm(), self.b.bounds_mm())

    def describe(self):
        return {"type": "union", "a": self.a.describe(), "b": self.b.describe()}


class Intersect(Region):
    def __init__(self, a: Region, b: Region):
        self.a, self.b = a, b

    def contains(self, P):
        return self.a.contains(P) & self.b.contains(P)

    def bounds_mm(self):
        return _aabb_intersect(self.a.bounds_mm(), self.b.bounds_mm())

    def describe(self):
        return {"type": "intersect", "a": self.a.describe(), "b": self.b.describe()}


class Difference(Region):
    def __init__(self, a: Region, b: Region):
        self.a, self.b = a, b

    def contains(self, P):
        return self.a.contains(P) & ~self.b.contains(P)

    def bounds_mm(self):
        return self.a.bounds_mm()

    def describe(self):
        return {"type": "difference", "a": self.a.describe(), "b": self.b.describe()}


class Complement(Region):
    def __init__(self, a: Region):
        self.a = a

    def contains(self, P):
        return ~self.a.contains(P)

    def bounds_mm(self):
        return None

    def describe(self):
        return {"type": "complement", "a": self.a.describe()}


class Everything(Region):
    def contains(self, P):
        return np.ones(P.shape[:-1], dtype=bool)


class Nothing(Region):
    def contains(self, P):
        return np.zeros(P.shape[:-1], dtype=bool)

    def bounds_mm(self):
        return np.zeros(3), np.zeros(3)


# ---------------------------------------------------------------- mask region
class MaskRegion(Region):
    """A boolean voxel mask lifted into world space (anatomy in an expression)."""

    def __init__(self, mask: np.ndarray, grid: Grid, name: str = "mask"):
        self.mask = np.ascontiguousarray(mask.astype(bool))
        self.grid = grid
        self.name = name

    def contains(self, P):
        idx = self.grid.index_of_mm(P)
        shape = np.asarray(self.grid.shape)
        ok = np.all((idx >= 0) & (idx < shape), axis=-1)
        out = np.zeros(P.shape[:-1], dtype=bool)
        i = idx[ok]
        out[ok] = self.mask[i[..., 0], i[..., 1], i[..., 2]]
        return out

    def bounds_mm(self):
        nz = np.argwhere(self.mask)
        if nz.size == 0:
            return np.zeros(3), np.zeros(3)
        return (self.grid.centers_mm(nz.min(0)) - 0.5 * self.grid.dx_mm,
                self.grid.centers_mm(nz.max(0)) + 0.5 * self.grid.dx_mm)

    def describe(self):
        return {"type": "mask", "name": self.name, "voxels": int(self.mask.sum())}


# ----------------------------------------------------------------- rotations
def rotation_axis_angle(axis, angle_deg: float) -> np.ndarray:
    """Rodrigues rotation matrix."""
    a = np.asarray(axis, dtype=float)
    n = np.linalg.norm(a)
    if n < _EPS:
        return np.eye(3)
    a = a / n
    t = math.radians(angle_deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(t) * K + (1 - math.cos(t)) * (K @ K)


def rotation_to(direction, from_axis=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Shortest rotation taking ``from_axis`` onto ``direction``."""
    u = np.asarray(from_axis, float); u = u / max(np.linalg.norm(u), _EPS)
    v = np.asarray(direction, float); v = v / max(np.linalg.norm(v), _EPS)
    c = float(np.dot(u, v))
    if c > 1 - 1e-12:
        return np.eye(3)
    if c < -1 + 1e-12:
        # 180 deg: any perpendicular axis
        perp = np.array([1.0, 0.0, 0.0])
        if abs(u[0]) > 0.9:
            perp = np.array([0.0, 1.0, 0.0])
        axis = np.cross(u, perp)
        return rotation_axis_angle(axis, 180.0)
    axis = np.cross(u, v)
    s = np.linalg.norm(axis)
    return rotation_axis_angle(axis, math.degrees(math.atan2(s, c)))


# --------------------------------------------------------------------- SDFs
class SDF(Region):
    """Signed-distance region in a local frame, with a rigid placement.

    Subclasses implement :meth:`_sdf_local` (points in the local frame) and
    :meth:`_bounds_local` (local AABB).  Placement is ``world = R @ local + t``.
    """

    def __init__(self):
        self._R = np.eye(3)
        self._t = np.zeros(3)

    # -- placement --------------------------------------------------------
    def _clone(self) -> "SDF":
        import copy
        return copy.copy(self)

    def transform(self, R=None, t=None) -> "SDF":
        """Apply ``x -> R x + t`` *after* the current placement."""
        o = self._clone()
        R = np.eye(3) if R is None else np.asarray(R, float)
        t = np.zeros(3) if t is None else np.asarray(t, float)
        o._R = R @ self._R
        o._t = R @ self._t + t
        return o

    def translate(self, v) -> "SDF":
        return self.transform(t=np.asarray(v, float))

    def rotate(self, axis, angle_deg: float, about=None) -> "SDF":
        R = rotation_axis_angle(axis, angle_deg)
        if about is None:
            return self.transform(R=R)
        c = np.asarray(about, float)
        return self.transform(R=R, t=c - R @ c)

    def align_z_to(self, direction, roll_deg: float = 0.0) -> "SDF":
        """Rotate so the local +z axis points along ``direction``."""
        R = rotation_to(direction)
        if roll_deg:
            R = R @ rotation_axis_angle((0, 0, 1), roll_deg)
        return self.transform(R=R)

    def place(self, origin, axis=(0, 0, 1), roll_deg: float = 0.0) -> "SDF":
        """Align local +z to ``axis`` then move the local origin to ``origin``."""
        return self.align_z_to(axis, roll_deg).translate(origin)

    # -- evaluation -------------------------------------------------------
    def _to_local(self, P):
        return (np.asarray(P, float) - self._t) @ self._R      # R^T (P - t)

    def sdf(self, P_mm: np.ndarray) -> np.ndarray:
        """Signed distance in mm (negative inside).  Rigid motions preserve it."""
        return self._sdf_local(self._to_local(P_mm))

    def contains(self, P_mm: np.ndarray) -> np.ndarray:
        return self.sdf(P_mm) <= 0.0

    def offset(self, d_mm: float) -> "SDF":
        """Dilate (``d>0``) or erode (``d<0``) by a uniform distance."""
        return _Offset(self, d_mm)

    def shell(self, thickness_mm: float) -> "SDF":
        """Hollow: keep only points within ``thickness/2`` of the surface."""
        return _Shell(self, thickness_mm)

    def bounds_mm(self):
        b = self._bounds_local()
        if b is None:
            return None
        lo, hi = np.asarray(b[0], float), np.asarray(b[1], float)
        corners = np.array([[x, y, z] for x in (lo[0], hi[0])
                            for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        w = corners @ self._R.T + self._t
        return w.min(0), w.max(0)

    # -- to implement -----------------------------------------------------
    def _sdf_local(self, P):                                  # pragma: no cover
        raise NotImplementedError

    def _bounds_local(self):                                  # pragma: no cover
        raise NotImplementedError

    def describe(self):
        d = {"type": type(self).__name__}
        d.update({k: (v.tolist() if isinstance(v, np.ndarray) else v)
                  for k, v in vars(self).items() if not k.startswith("_")})
        d["placement"] = {"R": self._R.tolist(), "t": self._t.tolist()}
        return d


class _Offset(SDF):
    def __init__(self, base: SDF, d: float):
        super().__init__()
        self.base, self.d = base, float(d)

    def _sdf_local(self, P):
        return self.base.sdf(P) - self.d

    def _bounds_local(self):
        b = self.base.bounds_mm()
        return None if b is None else (b[0] - self.d, b[1] + self.d)

    def describe(self):
        return {"type": "offset", "d_mm": self.d, "base": self.base.describe()}


class _Shell(SDF):
    def __init__(self, base: SDF, t: float):
        super().__init__()
        self.base, self.t = base, float(t)

    def _sdf_local(self, P):
        return np.abs(self.base.sdf(P)) - 0.5 * self.t

    def _bounds_local(self):
        b = self.base.bounds_mm()
        return None if b is None else (b[0] - 0.5 * self.t, b[1] + 0.5 * self.t)

    def describe(self):
        return {"type": "shell", "thickness_mm": self.t, "base": self.base.describe()}


# ------------------------------------------------------------- primitives
class Sphere(SDF):
    def __init__(self, radius_mm: float):
        super().__init__(); self.radius_mm = float(radius_mm)

    def _sdf_local(self, P):
        return np.linalg.norm(P, axis=-1) - self.radius_mm

    def _bounds_local(self):
        r = self.radius_mm
        return np.full(3, -r), np.full(3, r)


class Box(SDF):
    """Axis-aligned box of full size ``size_mm`` centred on the local origin."""

    def __init__(self, size_mm):
        super().__init__(); self.size_mm = np.asarray(size_mm, float)

    def _sdf_local(self, P):
        q = np.abs(P) - 0.5 * self.size_mm
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
        inside = np.minimum(np.max(q, axis=-1), 0.0)
        return outside + inside

    def _bounds_local(self):
        return -0.5 * self.size_mm, 0.5 * self.size_mm


class Cylinder(SDF):
    """Capped cylinder along local +z, centred, height ``height_mm``."""

    def __init__(self, radius_mm: float, height_mm: float):
        super().__init__()
        self.radius_mm = float(radius_mm); self.height_mm = float(height_mm)

    def _sdf_local(self, P):
        d_r = np.linalg.norm(P[..., :2], axis=-1) - self.radius_mm
        d_z = np.abs(P[..., 2]) - 0.5 * self.height_mm
        d = np.stack([d_r, d_z], axis=-1)
        return (np.minimum(np.max(d, axis=-1), 0.0)
                + np.linalg.norm(np.maximum(d, 0.0), axis=-1))

    def _bounds_local(self):
        r, h = self.radius_mm, 0.5 * self.height_mm
        return np.array([-r, -r, -h]), np.array([r, r, h])


Disc = Cylinder


class Capsule(SDF):
    """Cylinder with hemispherical caps; ``length_mm`` is the axial segment."""

    def __init__(self, radius_mm: float, length_mm: float):
        super().__init__()
        self.radius_mm = float(radius_mm); self.length_mm = float(length_mm)

    def _sdf_local(self, P):
        h = 0.5 * self.length_mm
        q = P.copy()
        q[..., 2] -= np.clip(q[..., 2], -h, h)
        return np.linalg.norm(q, axis=-1) - self.radius_mm

    def _bounds_local(self):
        r, h = self.radius_mm, 0.5 * self.length_mm + self.radius_mm
        return np.array([-r, -r, -h]), np.array([r, r, h])


class Cone(SDF):
    """Cone: base radius at ``z = -height/2``, apex at ``z = +height/2``."""

    def __init__(self, radius_mm: float, height_mm: float):
        super().__init__()
        self.radius_mm = float(radius_mm); self.height_mm = float(height_mm)

    def _sdf_local(self, P):
        h = self.height_mm
        r = np.linalg.norm(P[..., :2], axis=-1)
        z = P[..., 2] + 0.5 * h                      # 0 at base, h at apex
        # distance to the lateral surface (2-D in (r, z)), then cap at base
        a = np.stack([r, z], -1)
        tipdir = np.array([-self.radius_mm, h])
        tipdir = tipdir / np.linalg.norm(tipdir)
        v = a - np.array([self.radius_mm, 0.0])
        proj = np.clip((v * tipdir).sum(-1), 0.0, np.linalg.norm([self.radius_mm, h]))
        lateral = np.linalg.norm(v - proj[..., None] * tipdir, axis=-1)
        inside = (z >= 0) & (z <= h) & (r <= self.radius_mm * (1 - z / h))
        d_base = -z
        d = np.maximum(lateral, d_base) if False else np.where(
            z < 0, np.hypot(np.maximum(r - self.radius_mm, 0.0), -z), lateral)
        return np.where(inside, -np.minimum(lateral, np.abs(z)), d)

    def _bounds_local(self):
        r, h = self.radius_mm, 0.5 * self.height_mm
        return np.array([-r, -r, -h]), np.array([r, r, h])


class Torus(SDF):
    """Ring of radius ``ring_radius_mm`` in the local xy-plane, wire radius ``tube``."""

    def __init__(self, ring_radius_mm: float, tube_radius_mm: float):
        super().__init__()
        self.ring_radius_mm = float(ring_radius_mm)
        self.tube_radius_mm = float(tube_radius_mm)

    def _sdf_local(self, P):
        q = np.linalg.norm(P[..., :2], axis=-1) - self.ring_radius_mm
        return np.hypot(q, P[..., 2]) - self.tube_radius_mm

    def _bounds_local(self):
        R, t = self.ring_radius_mm, self.tube_radius_mm
        return np.array([-R - t, -R - t, -t]), np.array([R + t, R + t, t])


class Annulus(SDF):
    """Flat ring (washer) in the local xy-plane."""

    def __init__(self, inner_radius_mm: float, outer_radius_mm: float,
                 thickness_mm: float):
        super().__init__()
        self.inner_radius_mm = float(inner_radius_mm)
        self.outer_radius_mm = float(outer_radius_mm)
        self.thickness_mm = float(thickness_mm)

    def _sdf_local(self, P):
        r = np.linalg.norm(P[..., :2], axis=-1)
        mid = 0.5 * (self.inner_radius_mm + self.outer_radius_mm)
        half = 0.5 * (self.outer_radius_mm - self.inner_radius_mm)
        d_r = np.abs(r - mid) - half
        d_z = np.abs(P[..., 2]) - 0.5 * self.thickness_mm
        d = np.stack([d_r, d_z], -1)
        return (np.minimum(np.max(d, -1), 0.0)
                + np.linalg.norm(np.maximum(d, 0.0), axis=-1))

    def _bounds_local(self):
        R, t = self.outer_radius_mm, 0.5 * self.thickness_mm
        return np.array([-R, -R, -t]), np.array([R, R, t])


class SphericalShell(SDF):
    """Spherical shell of mid-surface radius ``radius_mm`` and given thickness."""

    def __init__(self, radius_mm: float, thickness_mm: float):
        super().__init__()
        self.radius_mm = float(radius_mm)
        self.thickness_mm = float(thickness_mm)

    def _sdf_local(self, P):
        r = np.linalg.norm(P, axis=-1)
        return np.abs(r - self.radius_mm) - 0.5 * self.thickness_mm

    def _bounds_local(self):
        R = self.radius_mm + 0.5 * self.thickness_mm
        return np.full(3, -R), np.full(3, R)


class SphericalBand(SDF):
    """Band of a spherical shell between two polar angles about local +z.

    ``theta_min_deg = 0`` gives a cap (contact-lens dome); a non-zero
    ``theta_min_deg`` gives an annular band (ring on a curved surface).
    """

    def __init__(self, radius_mm: float, thickness_mm: float,
                 theta_max_deg: float, theta_min_deg: float = 0.0):
        super().__init__()
        self.radius_mm = float(radius_mm)
        self.thickness_mm = float(thickness_mm)
        self.theta_min_deg = float(theta_min_deg)
        self.theta_max_deg = float(theta_max_deg)

    def _sdf_local(self, P):
        r = np.linalg.norm(P, axis=-1)
        d_shell = np.abs(r - self.radius_mm) - 0.5 * self.thickness_mm
        rs = np.maximum(r, _EPS)
        theta = np.degrees(np.arccos(np.clip(P[..., 2] / rs, -1.0, 1.0)))
        mid = 0.5 * (self.theta_min_deg + self.theta_max_deg)
        half = 0.5 * (self.theta_max_deg - self.theta_min_deg)
        # angular deviation converted to arc length on the mid-surface
        d_ang = np.radians(np.abs(theta - mid) - half) * self.radius_mm
        d = np.stack([d_shell, d_ang], -1)
        return (np.minimum(np.max(d, -1), 0.0)
                + np.linalg.norm(np.maximum(d, 0.0), axis=-1))

    def _bounds_local(self):
        R = self.radius_mm + 0.5 * self.thickness_mm
        smax = math.sin(math.radians(min(self.theta_max_deg, 90.0))) if \
            self.theta_max_deg <= 90 else 1.0
        zmax = R * math.cos(math.radians(self.theta_min_deg))
        zmin = R * math.cos(math.radians(self.theta_max_deg))
        rad = R * max(smax, math.sin(math.radians(self.theta_min_deg)))
        return (np.array([-rad, -rad, min(zmin, zmax)]),
                np.array([rad, rad, max(zmin, zmax)]))


def SphericalCap(radius_mm: float, thickness_mm: float, half_angle_deg: float) -> SphericalBand:
    """Contact-lens dome: shell cap of given half-angle about local +z."""
    return SphericalBand(radius_mm, thickness_mm, half_angle_deg, 0.0)


class Needle(SDF):
    """Shaft (capsule) along local +z with a conical tip pointing to -z.

    The tip apex sits at the local origin, so ``place(origin=tip, axis=...)``
    puts the tip exactly where you want it and the shaft runs back along +z.
    """

    def __init__(self, radius_mm: float, length_mm: float, tip_length_mm: float = 0.0):
        super().__init__()
        self.radius_mm = float(radius_mm)
        self.length_mm = float(length_mm)
        self.tip_length_mm = float(tip_length_mm)

    def _sdf_local(self, P):
        r = np.linalg.norm(P[..., :2], axis=-1)
        z = P[..., 2]
        tl, L, a = self.tip_length_mm, self.length_mm, self.radius_mm
        # local radius profile: 0 at z=0 growing to a at z=tl, then constant
        if tl > 0:
            prof = np.clip(z / tl, 0.0, 1.0) * a
        else:
            prof = np.full_like(z, a)
        d_r = r - prof
        d_z = np.maximum(-z, z - L)
        d = np.stack([d_r, d_z], -1)
        return (np.minimum(np.max(d, -1), 0.0)
                + np.linalg.norm(np.maximum(d, 0.0), axis=-1))

    def _bounds_local(self):
        a, L = self.radius_mm, self.length_mm
        return np.array([-a, -a, 0.0]), np.array([a, a, L])


class Cuff(SDF):
    """Open cylindrical cuff about local +z, with an angular gap."""

    def __init__(self, inner_radius_mm: float, thickness_mm: float,
                 length_mm: float, gap_deg: float = 0.0):
        super().__init__()
        self.inner_radius_mm = float(inner_radius_mm)
        self.thickness_mm = float(thickness_mm)
        self.length_mm = float(length_mm)
        self.gap_deg = float(gap_deg)

    def _sdf_local(self, P):
        r = np.linalg.norm(P[..., :2], axis=-1)
        mid = self.inner_radius_mm + 0.5 * self.thickness_mm
        d_r = np.abs(r - mid) - 0.5 * self.thickness_mm
        d_z = np.abs(P[..., 2]) - 0.5 * self.length_mm
        d = np.stack([d_r, d_z], -1)
        s = (np.minimum(np.max(d, -1), 0.0)
             + np.linalg.norm(np.maximum(d, 0.0), axis=-1))
        if self.gap_deg > 0:
            phi = np.degrees(np.arctan2(P[..., 1], P[..., 0]))
            half = 0.5 * self.gap_deg
            in_gap = np.abs(phi) <= half
            d_gap = np.radians(half - np.abs(phi)) * mid   # arc distance into gap
            s = np.where(in_gap, np.maximum(s, d_gap), s)
        return s

    def _bounds_local(self):
        R = self.inner_radius_mm + self.thickness_mm
        h = 0.5 * self.length_mm
        return np.array([-R, -R, -h]), np.array([R, R, h])


class Polyline(SDF):
    """Swept sphere along a polyline: wires, leads, curved arrays."""

    def __init__(self, points_mm, radius_mm: float):
        super().__init__()
        self.points_mm = np.asarray(points_mm, float).reshape(-1, 3)
        self.radius_mm = float(radius_mm)
        if len(self.points_mm) < 2:
            raise ValueError("Polyline needs at least two points")

    def _sdf_local(self, P):
        A = self.points_mm[:-1]
        B = self.points_mm[1:]
        AB = B - A
        L2 = np.maximum((AB ** 2).sum(-1), _EPS)
        best = None
        flat = P.reshape(-1, 3)
        for a, ab, l2 in zip(A, AB, L2):
            t = np.clip(((flat - a) @ ab) / l2, 0.0, 1.0)
            d = np.linalg.norm(flat - (a + t[:, None] * ab), axis=-1)
            best = d if best is None else np.minimum(best, d)
        return (best - self.radius_mm).reshape(P.shape[:-1])

    def _bounds_local(self):
        r = self.radius_mm
        return self.points_mm.min(0) - r, self.points_mm.max(0) + r


# ------------------------------------------------------------- rasterization
def rasterize(region: Region, grid: Grid, supersample: int = 3,
              threshold: float = 0.5, fraction: bool = False, lattice: int = 1):
    """Rasterize a region onto ``grid``.

    Returns ``(idx, values, box)`` where ``idx`` is an ``(N, 3)`` array of voxel
    indices, ``values`` is the boolean occupancy (or the occupancy fraction if
    ``fraction=True``), and ``box`` is the ``(lo, hi)`` index window examined.

    Only voxels inside the region's bounding box are evaluated, and each voxel
    is sampled on an ``s x s x s`` lattice of sub-points, so features thinner
    than a voxel (a 150 um wire in an 83 um grid) still rasterize sanely.

    ``lattice=n`` instead rasterizes onto an ``n``-times coarser grid covering
    the same physical space and expands the result back, so every ``n x n x n``
    block of the fine grid is uniformly filled. Lab electrodes are frequently
    hand-drawn this way (e.g. at a 166 um pitch inside an 83 um model);
    matching that authoring lattice is what lets a region generated from a
    fitted spec reproduce a hand-built electrode voxel-for-voxel rather than
    just approximately. Because regions are defined in millimetres, this needs
    no parameter rescaling -- the same region is simply evaluated on a
    physically coarser lattice.
    """
    if lattice is not None and int(lattice) > 1:
        n = int(lattice)
        coarse = Grid(tuple(int(math.ceil(s / n)) for s in grid.shape),
                      grid.dx_mm * n, grid.origin_mm)
        cidx, cvals, (clo, chi) = rasterize(region, coarse, supersample=supersample,
                                            threshold=threshold, fraction=fraction)
        shape = np.asarray(grid.shape)
        lo, hi = np.asarray(clo) * n, np.minimum(np.asarray(chi) * n, shape)
        if len(cidx) == 0:
            return (np.zeros((0, 3), np.int64),
                    np.zeros(0, dtype=float if fraction else bool), (lo, hi))
        offs = np.stack(np.meshgrid(*([np.arange(n)] * 3), indexing="ij"),
                        -1).reshape(-1, 3)
        idx = (cidx[:, None, :] * n + offs[None, :, :]).reshape(-1, 3)
        vals = np.repeat(cvals, len(offs)) if fraction \
            else np.ones(len(idx), dtype=bool)
        keep = np.all(idx < shape, axis=1)
        return idx[keep].astype(np.int64), vals[keep], (lo, hi)

    b = region.bounds_mm()
    if b is None:
        lo = np.zeros(3, dtype=np.int64)
        hi = np.asarray(grid.shape, dtype=np.int64)
    else:
        lo, hi = grid.clip_box(b[0], b[1])
    if np.any(hi <= lo):
        empty = np.zeros((0, 3), dtype=np.int64)
        return empty, np.zeros(0, dtype=bool if not fraction else float), (lo, hi)

    s = max(1, int(supersample))
    off = (np.arange(s) + 0.5) / s - 0.5                      # sub-voxel offsets
    ox, oy, oz = np.meshgrid(off, off, off, indexing="ij")
    subs = np.stack([ox, oy, oz], -1).reshape(-1, 3) * grid.dx_mm

    ii = np.arange(lo[0], hi[0]); jj = np.arange(lo[1], hi[1]); kk = np.arange(lo[2], hi[2])
    shape = (len(ii), len(jj), len(kk))
    acc = np.zeros(shape, dtype=np.float32)

    # chunk over x-slabs to bound peak memory
    max_pts = 4_000_000
    per_slab = max(1, int(max_pts / max(1, shape[1] * shape[2] * len(subs))))
    for start in range(0, shape[0], per_slab):
        stop = min(start + per_slab, shape[0])
        I, J, K = np.meshgrid(ii[start:stop], jj, kk, indexing="ij")
        C = grid.centers_mm(np.stack([I, J, K], -1))          # (a,b,c,3)
        hits = np.zeros(C.shape[:-1], dtype=np.float32)
        for d in subs:
            hits += region.contains(C + d)
        acc[start:stop] = hits / len(subs)

    frac = acc
    sel = frac >= threshold
    idx = np.argwhere(sel) + lo
    vals = frac[sel] if fraction else np.ones(len(idx), dtype=bool)
    return idx.astype(np.int64), vals, (lo, hi)
