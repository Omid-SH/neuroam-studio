"""Anatomical frames and surface operators for electrode registration.

Placing an electrode by hard-coded voxel index ranges (the legacy
``for i = 211:212 ... model_new(j,k,i) = 102`` pattern) does not survive a
change of resolution, a change of anatomy, or a rotation.  This module gives
electrodes an *anatomical* address instead:

- :class:`Frame` -- an origin and an orthonormal triad derived from the
  anatomy itself (e.g. the eye centre from a least-squares sphere fit to the
  retina shell, with +z along the corneal axis).  A spec written as
  "ring of radius 2.7 mm, 2.0 mm anterior of the globe centre" then maps to
  any model that has the same landmarks, at any voxel size.
- surface operators -- signed distance to a labelled tissue, ray-cast surface
  points, outward normals, snap-to-surface with a standoff, and
  :func:`conform`, which drapes a footprint onto a tissue surface so a patch
  or lens electrode follows the anatomy instead of floating in a box.

All coordinates are millimetres in the model's world frame unless a name says
``_vox``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

from .geometry import Grid, MaskRegion, Region, SDF, rotation_to

__all__ = ["Frame", "sphere_fit", "label_mask", "signed_distance_mm",
           "surface_band", "conform", "surface_point", "normal_at",
           "snap_to_surface"]


# ------------------------------------------------------------------- fitting
def sphere_fit(points: np.ndarray) -> Tuple[np.ndarray, float]:
    """Least-squares sphere through points ``(N, 3)``: returns (centre, radius)."""
    P = np.asarray(points, float)
    A = np.c_[2 * P, np.ones(len(P))]
    b = (P ** 2).sum(1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = sol[:3]
    return c, float(np.sqrt(max(sol[3] + (c ** 2).sum(), 0.0)))


def label_mask(model, labels: Iterable[int]) -> np.ndarray:
    labels = list(labels)
    return np.isin(model.labels, labels)


# --------------------------------------------------------------------- frame
@dataclass
class Frame:
    """An anatomical coordinate frame.

    ``axes`` rows are the unit vectors (e1, e2, e3); **e3 is the principal
    axis** (for an eye: the corneal / anterior axis).  Spherical coordinates
    use ``theta`` from +e3 and ``phi`` measured from +e1 towards +e2.
    """

    origin_mm: np.ndarray
    axes: np.ndarray
    grid: Grid
    name: str = "frame"
    radius_mm: Optional[float] = None       # fitted landmark radius, if any
    fit_rms_mm: Optional[float] = None

    # -- constructors -----------------------------------------------------
    @classmethod
    def from_axes(cls, origin_mm, axis, grid: Grid, roll_deg: float = 0.0,
                  name: str = "frame") -> "Frame":
        R = rotation_to(axis)                       # local +z -> axis
        if roll_deg:
            from .geometry import rotation_axis_angle
            R = R @ rotation_axis_angle((0, 0, 1), roll_deg)
        axes = np.stack([R[:, 0], R[:, 1], R[:, 2]])   # rows e1,e2,e3
        return cls(np.asarray(origin_mm, float), axes, grid, name)

    @classmethod
    def from_shell(cls, model, shell_labels: Iterable[int], axis=None,
                   axis_from_labels: Optional[Iterable[int]] = None,
                   axis_sign: float = 1.0, name: str = "frame") -> "Frame":
        """Fit a sphere to a shell-like structure and build a frame on it.

        ``axis`` may be given directly (world direction).  Otherwise it is the
        direction from the fitted centre to the centroid of
        ``axis_from_labels`` (e.g. the stimulating electrode, the lens, or the
        cornea), times ``axis_sign``.
        """
        grid = Grid.from_model(model)
        idx = np.argwhere(label_mask(model, shell_labels))
        if idx.size == 0:
            raise ValueError(f"no voxels with labels {list(shell_labels)}")
        P = grid.centers_mm(idx)
        c, r = sphere_fit(P)
        rms = float(np.sqrt(np.mean((np.linalg.norm(P - c, axis=1) - r) ** 2)))
        if axis is None:
            if axis_from_labels is None:
                raise ValueError("give axis= or axis_from_labels=")
            q = np.argwhere(label_mask(model, axis_from_labels))
            if q.size == 0:
                raise ValueError(f"no voxels with labels {list(axis_from_labels)}")
            axis = grid.centers_mm(q).mean(0) - c
        a = np.asarray(axis, float) * axis_sign
        f = cls.from_axes(c, a, grid, name=name)
        f.radius_mm = r
        f.fit_rms_mm = rms
        return f

    @classmethod
    def eye(cls, model, retina_labels=(77,), anterior=None,
            anterior_from_labels=None, name: str = "eye") -> "Frame":
        """Eye frame: origin at the globe centre, +e3 along the corneal axis.

        With neither ``anterior`` nor ``anterior_from_labels``, the anterior
        direction is taken as *away* from the centroid of the retina labels
        (the retina caps the posterior pole).
        """
        if anterior is None and anterior_from_labels is None:
            return cls.from_shell(model, retina_labels,
                                  axis_from_labels=retina_labels,
                                  axis_sign=-1.0, name=name)
        return cls.from_shell(model, retina_labels, axis=anterior,
                              axis_from_labels=anterior_from_labels, name=name)

    # -- geometry ---------------------------------------------------------
    @property
    def e1(self): return self.axes[0]

    @property
    def e2(self): return self.axes[1]

    @property
    def e3(self): return self.axes[2]

    @property
    def origin_vox(self):
        return self.origin_mm / self.grid.dx_mm - 0.5

    def direction(self, theta_deg: float, phi_deg: float = 0.0) -> np.ndarray:
        t, p = math.radians(theta_deg), math.radians(phi_deg)
        return (math.sin(t) * math.cos(p) * self.e1
                + math.sin(t) * math.sin(p) * self.e2
                + math.cos(t) * self.e3)

    def point(self, r_mm: float, theta_deg: float = 0.0, phi_deg: float = 0.0):
        """Point at spherical coordinates about the frame origin (world mm)."""
        return self.origin_mm + r_mm * self.direction(theta_deg, phi_deg)

    def along(self, axial_mm: float, lateral_mm: float = 0.0,
              phi_deg: float = 0.0):
        """Point ``axial_mm`` along +e3 and ``lateral_mm`` off-axis at ``phi``."""
        p = math.radians(phi_deg)
        return (self.origin_mm + axial_mm * self.e3
                + lateral_mm * (math.cos(p) * self.e1 + math.sin(p) * self.e2))

    def to_frame(self, P_world_mm) -> np.ndarray:
        return (np.asarray(P_world_mm, float) - self.origin_mm) @ self.axes.T

    def to_world(self, P_frame_mm) -> np.ndarray:
        return np.asarray(P_frame_mm, float) @ self.axes + self.origin_mm

    def spherical_of(self, P_world_mm):
        """(r, theta_deg, phi_deg) of world points in this frame."""
        L = self.to_frame(P_world_mm)
        r = np.linalg.norm(L, axis=-1)
        th = np.degrees(np.arccos(np.clip(L[..., 2] / np.maximum(r, 1e-12), -1, 1)))
        ph = np.degrees(np.arctan2(L[..., 1], L[..., 0]))
        return r, th, ph

    def place(self, sdf: SDF, r_mm: float = 0.0, theta_deg: float = 0.0,
              phi_deg: float = 0.0, roll_deg: float = 0.0,
              axis: Optional[Sequence[float]] = None) -> SDF:
        """Put an SDF at frame spherical coordinates, local +z along the radius.

        ``axis`` overrides the orientation (given in world coordinates).
        """
        origin = self.point(r_mm, theta_deg, phi_deg)
        a = self.direction(theta_deg, phi_deg) if axis is None else np.asarray(axis, float)
        return sdf.place(origin, a, roll_deg)

    # -- convenience shapes on a spherical surface -------------------------
    def ring(self, r_mm: float, theta_deg: float, tube_radius_mm: float,
             phi_deg: float = 0.0) -> SDF:
        """A wire ring lying on the sphere of radius ``r_mm`` at polar angle ``theta``.

        This is the natural address for an ocular ring electrode: ``r_mm`` is
        the distance from the globe centre (so the ring hugs the surface) and
        ``theta_deg`` slides it from the corneal apex (0) toward the equator
        (90).  ``phi_deg`` tilts the ring plane's normal off the frame axis.
        """
        from .geometry import Torus
        t = math.radians(theta_deg)
        axis = self.direction(phi_deg, 0.0) if phi_deg else self.e3
        centre = self.origin_mm + r_mm * math.cos(t) * axis
        return Torus(r_mm * math.sin(t), tube_radius_mm).place(centre, axis)

    def arc(self, r_mm: float, theta_deg: float, tube_radius_mm: float,
            phi0_deg: float = 0.0, span_deg: float = 360.0,
            n_points: int = 81) -> SDF:
        """A partial wire ring on the sphere of radius ``r_mm`` at ``theta``.

        The open-arc counterpart of :meth:`ring` — a thread that lies against
        the globe over ``span_deg`` of azimuth, centred on ``phi0_deg``.  This
        is the natural address for a DTL-type ocular thread electrode, which
        contacts an arc of the limbus rather than a closed loop.

        ``span_deg >= 360`` closes the loop (and is then just :meth:`ring`
        sampled as a polyline).  Arc length is ``r_mm * sin(theta) * span``.
        """
        from .geometry import Polyline
        span = min(float(span_deg), 360.0)
        closed = span >= 359.999
        phi = np.linspace(phi0_deg - span / 2.0, phi0_deg + span / 2.0,
                          int(n_points))
        if closed:
            phi = phi[:-1]
            phi = np.append(phi, phi[0])
        pts = [self.point(r_mm, theta_deg, float(p)).tolist() for p in phi]
        return Polyline(pts, tube_radius_mm)

    def arc_length_mm(self, r_mm: float, theta_deg: float,
                      span_deg: float) -> float:
        """Contact length of :meth:`arc` — what a clinical spec quotes in mm."""
        return r_mm * math.sin(math.radians(theta_deg)) * math.radians(
            min(float(span_deg), 360.0))

    def cap(self, r_mm: float, half_angle_deg: float, thickness_mm: float) -> SDF:
        """A contact-lens dome: spherical shell cap about the frame axis."""
        from .geometry import SphericalCap
        return SphericalCap(r_mm, thickness_mm, half_angle_deg).place(
            self.origin_mm, self.e3)

    def band(self, r_mm: float, theta_min_deg: float, theta_max_deg: float,
             thickness_mm: float) -> SDF:
        """An annular band of a spherical shell (a wide ring that follows curvature)."""
        from .geometry import SphericalBand
        return SphericalBand(r_mm, thickness_mm, theta_max_deg, theta_min_deg).place(
            self.origin_mm, self.e3)

    def describe(self) -> dict:
        return {"name": self.name, "origin_mm": self.origin_mm.tolist(),
                "origin_vox": np.round(self.origin_vox, 3).tolist(),
                "axes": self.axes.tolist(),
                "radius_mm": self.radius_mm, "fit_rms_mm": self.fit_rms_mm}


# ---------------------------------------------------------------- surfaces
def signed_distance_mm(mask: np.ndarray, dx_mm: float) -> np.ndarray:
    """Signed distance to the mask boundary: negative inside, positive outside."""
    from scipy import ndimage
    d_out = ndimage.distance_transform_edt(~mask)
    d_in = ndimage.distance_transform_edt(mask)
    return (d_out - d_in) * dx_mm


def surface_band(model, labels: Iterable[int], inner_mm: float, outer_mm: float,
                 name: str = "band", crop_center_mm: Optional[np.ndarray] = None,
                 crop_radius_mm: Optional[float] = None) -> MaskRegion:
    """Voxels whose signed distance to a tissue lies in ``[inner, outer]`` mm.

    ``inner_mm = 0, outer_mm = 0.2`` is a 200 um skin of tissue *outside* the
    structure -- where a surface electrode sits.  Negative values reach inside
    the tissue (a recessed or embedded contact).

    The returned mask always covers the whole model (``MaskRegion`` indexes
    into it with full-grid voxel coordinates), but the expensive part -- the
    distance transform -- is computed on a local crop when
    ``crop_center_mm``/``crop_radius_mm`` are given, and only that crop of the
    result is filled in (everywhere else is simply "not in the band", which
    is correct as long as the region this feeds -- typically a small local
    footprint via :func:`conform` -- never reaches past the crop). Omit both
    for the previous whole-grid behaviour, e.g. a genuinely global band.
    """
    grid = Grid.from_model(model)
    shape = np.asarray(grid.shape)
    if crop_center_mm is None:
        sd = signed_distance_mm(label_mask(model, labels), grid.dx_mm)
        return MaskRegion((sd >= inner_mm) & (sd <= outer_mm), grid, name=name)

    pad_mm = max(crop_radius_mm or 0.0, abs(outer_mm), abs(inner_mm)) + 1.0
    c = grid.index_of_mm(np.asarray(crop_center_mm, float))
    pad_vox = int(math.ceil(pad_mm / grid.dx_mm))
    lo = np.clip(c - pad_vox, 0, None)
    hi = np.clip(c + pad_vox + 1, None, shape)
    sub_labels = model.labels[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    sd = signed_distance_mm(np.isin(sub_labels, np.asarray(list(labels))), grid.dx_mm)
    band_local = (sd >= inner_mm) & (sd <= outer_mm)
    full = np.zeros(shape, dtype=bool)
    full[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = band_local
    return MaskRegion(full, grid, name=name)


def conform(footprint: Region, model, labels: Iterable[int],
            offset_mm: float = 0.0, thickness_mm: float = 0.1,
            name: str = "conformal") -> Region:
    """Drape a footprint onto a tissue surface.

    ``footprint`` is any region that carves out *where* on the body the
    contact goes (a cone from the eye centre, a cylinder, a box); the result
    is its intersection with a thin shell hugging the tissue, so the electrode
    follows curvature instead of cutting through it.

    Restricts the underlying distance transform to a crop around
    ``footprint`` itself (via its own ``bounds_mm()``) when available -- a
    conformal footprint is always small and local, so there is no reason to
    pay for a whole-model distance transform to drape it (see
    :func:`surface_band`'s ``crop_*`` parameters).
    """
    crop_center = crop_radius = None
    b = footprint.bounds_mm()
    if b is not None:
        lo_mm, hi_mm = np.asarray(b[0]), np.asarray(b[1])
        crop_center = 0.5 * (lo_mm + hi_mm)
        crop_radius = float(np.linalg.norm(hi_mm - lo_mm)) * 0.5
    band = surface_band(model, labels, offset_mm, offset_mm + thickness_mm, name=name,
                        crop_center_mm=crop_center, crop_radius_mm=crop_radius)
    return footprint & band


def surface_point(model, labels: Iterable[int], origin_mm, direction,
                  max_mm: Optional[float] = None, step_mm: Optional[float] = None):
    """March from ``origin`` along ``direction``; return the exit point of the tissue.

    Returns the last point inside the labelled tissue (world mm), or ``None``
    if the ray never enters it. Samples ``model.labels`` directly at each
    step rather than building a whole-grid boolean mask first (``label_mask``)
    -- a ray march only ever touches a few hundred points regardless of model
    size, so there is no reason to pay for a full-array pass first.
    """
    grid = Grid.from_model(model)
    labels_arr = np.asarray(list(labels))
    d = np.asarray(direction, float)
    d = d / max(np.linalg.norm(d), 1e-12)
    step = step_mm or 0.5 * grid.dx_mm
    if max_mm is None:
        max_mm = float(np.linalg.norm(np.asarray(grid.shape) * grid.dx_mm))
    n = int(max_mm / step)
    P = np.asarray(origin_mm, float) + np.outer(np.arange(n) * step, d)
    idx = grid.index_of_mm(P)
    ok = np.all((idx >= 0) & (idx < np.asarray(grid.shape)), axis=-1)
    inside = np.zeros(len(P), dtype=bool)
    i = idx[ok]
    inside[ok] = np.isin(model.labels[i[:, 0], i[:, 1], i[:, 2]], labels_arr)
    if not inside.any():
        return None
    return P[np.flatnonzero(inside)[-1]]


def normal_at(model, labels: Iterable[int], point_mm, smooth_vox: float = 1.5,
             crop_mm: float = 3.0):
    """Outward unit normal of a labelled surface near ``point`` (world mm).

    The normal is an inherently *local* quantity -- computed here from a
    small crop of the model around ``point_mm`` (``crop_mm`` on each side,
    padded for the smoothing kernel), not a whole-model distance transform.
    On a small crop model both cost the same; on a full head model (hundreds
    of millions of voxels) a whole-grid ``distance_transform_edt`` here would
    take minutes *per electrode* for a result that only ever depended on a
    neighbourhood a few voxels across -- ``crop_mm=3.0`` is generous for a
    typical ``smooth_vox<=3``; raise it only if a very large ``smooth_vox``
    genuinely needs more context.
    """
    from scipy import ndimage
    grid = Grid.from_model(model)
    shape = np.asarray(grid.shape)
    i = grid.index_of_mm(point_mm)
    pad_vox = int(math.ceil(crop_mm / grid.dx_mm)) + int(math.ceil(smooth_vox * 3)) + 2
    lo = np.clip(i - pad_vox, 0, None)
    hi = np.clip(i + pad_vox + 1, None, shape)
    sub = model.labels[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    sd = signed_distance_mm(np.isin(sub, np.asarray(list(labels))), grid.dx_mm)
    if smooth_vox:
        sd = ndimage.gaussian_filter(sd, smooth_vox)
    li = np.clip(i - lo, 1, np.asarray(sub.shape) - 2)
    g = np.array([
        sd[li[0] + 1, li[1], li[2]] - sd[li[0] - 1, li[1], li[2]],
        sd[li[0], li[1] + 1, li[2]] - sd[li[0], li[1] - 1, li[2]],
        sd[li[0], li[1], li[2] + 1] - sd[li[0], li[1], li[2] - 1]])
    n = np.linalg.norm(g)
    return g / n if n > 1e-12 else np.array([0.0, 0.0, 1.0])


def snap_to_surface(sdf: SDF, model, labels: Iterable[int], direction,
                    standoff_mm: float = 0.0, search_mm: float = 5.0,
                    tol_mm: Optional[float] = None) -> SDF:
    """Slide an SDF along ``direction`` until it just clears the tissue.

    Bisects on the translation so that the minimum gap between the electrode
    body and the tissue surface equals ``standoff_mm`` (0 = touching).
    """
    grid = Grid.from_model(model)
    tol = tol_mm or 0.1 * grid.dx_mm
    mask = label_mask(model, labels)
    idx = np.argwhere(mask)
    if idx.size == 0:
        raise ValueError("empty target tissue")
    d = np.asarray(direction, float)
    d = d / max(np.linalg.norm(d), 1e-12)

    b = sdf.bounds_mm()
    pad = search_mm + standoff_mm + 2 * grid.dx_mm
    lo, hi = grid.clip_box(b[0] - pad, b[1] + pad)
    sub = idx[np.all((idx >= lo) & (idx < hi), axis=1)]
    if sub.size == 0:
        sub = idx
    S = grid.centers_mm(sub)

    def gap(t):
        return float(np.min(sdf.translate(d * t).sdf(S)))

    # walk outwards until clear, then bisect
    t_lo, t_hi = 0.0, 0.0
    if gap(0.0) < standoff_mm:
        t_hi = tol
        while gap(t_hi) < standoff_mm and t_hi < search_mm:
            t_lo, t_hi = t_hi, t_hi * 2 if t_hi else tol
    else:
        t_lo = -search_mm
        while gap(t_lo) >= standoff_mm and t_lo < 0:
            t_lo += 0.25 * search_mm
        t_hi = t_lo + 0.25 * search_mm
    for _ in range(60):
        mid = 0.5 * (t_lo + t_hi)
        if gap(mid) < standoff_mm:
            t_lo = mid
        else:
            t_hi = mid
        if t_hi - t_lo < tol:
            break
    return sdf.translate(d * t_hi)
