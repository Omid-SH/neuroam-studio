"""Interactive 3D visualization of voxel models, electrodes and AM fields.

Design notes
------------
* **Identity is carried by three chromatic slots only** — stimulating electrode
  (red), return electrode (blue), target tissue (green) — validated all-pairs
  in light and dark. Everything else is neutral grey "context" at low opacity.
  This is both an accessibility decision and the right scientific emphasis:
  anatomy is context, the electrodes and the target structure are the message.
* Every surface carries a legend entry and a hover label, so identity is never
  colour-alone, and each legend entry is a **toggle** — that is the mechanism
  for "hide the rest of the head, show only the eye, then only the retina".
* Field magnitude uses a perceptually-uniform sequential ramp (monotone
  lightness, CVD-safe) — never a rainbow.

Output is a self-contained HTML file (plotly) plus optional static PNG
snapshots (matplotlib) for reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

# --------------------------------------------------------------------------
# palette (validated with the dataviz palette validator, --pairs all)
# --------------------------------------------------------------------------
PALETTE = {
    "light": {
        "surface": "#fcfcfb", "text": "#0b0b0b", "text2": "#52514e",
        "source": "#e34948", "ground": "#2a78d6", "target": "#1baf7a",
        "context": "#8a8a86", "insulation": "#4f4f4c", "grid": "#dcdcd8",
    },
    "dark": {
        "surface": "#1a1a19", "text": "#ffffff", "text2": "#c3c2b7",
        "source": "#e66767", "ground": "#3987e5", "target": "#199e70",
        "context": "#7d7d79", "insulation": "#a8a8a2", "grid": "#33332f",
    },
}

#: sequential ramps for field magnitude — monotone lightness, CVD-safe
FIELD_CMAP = "Magma"
FIELD_CMAP_LIGHT = "Viridis"

#: conventional roles -> palette slot
ROLE_COLOR = {"source": "source", "stim": "source", "anode": "source",
              "ground": "ground", "return": "ground", "cathode": "ground",
              "target": "target", "insulation": "insulation",
              "context": "context"}


def _require_plotly():
    try:
        import plotly.graph_objects as go  # noqa: F401
        return True
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "3D views need plotly:  pip install plotly") from exc


# --------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------
def _crop_slices(crop) -> Tuple[slice, slice, slice]:
    if crop is None:
        return (slice(None),) * 3
    (x0, x1), (y0, y1), (z0, z1) = crop
    return slice(x0, x1), slice(y0, y1), slice(z0, z1)


def isosurface_from_mask(mask: np.ndarray, step: int = 1,
                         origin=(0, 0, 0), spacing: float = 1.0,
                         smooth_iters: int = 0):
    """Marching-cubes surface of a boolean voxel mask.

    Returns ``(verts, faces)`` in physical units (``spacing`` per voxel),
    offset by ``origin`` (in voxels).  ``step`` decimates the volume before
    meshing — use it on large models.
    """
    from skimage import measure

    m = np.ascontiguousarray(mask[::step, ::step, ::step]).astype(np.float32)
    if m.max() == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)
    # pad so surfaces at the volume border close properly
    m = np.pad(m, 1, mode="constant", constant_values=0.0)
    try:
        verts, faces, _, _ = measure.marching_cubes(m, level=0.5)
    except (RuntimeError, ValueError):
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)
    verts = (verts - 1.0) * step + np.asarray(origin, dtype=float)
    if smooth_iters:
        verts = _laplacian_smooth(verts, faces, smooth_iters)
    return verts * spacing, faces


def _laplacian_smooth(verts: np.ndarray, faces: np.ndarray, iters: int = 2,
                      lam: float = 0.5) -> np.ndarray:
    """Light Laplacian smoothing to take the stair-steps off voxel surfaces."""
    n = len(verts)
    e0 = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    e1 = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    rows = np.concatenate([e0, e1])
    cols = np.concatenate([e1, e0])
    from scipy import sparse
    A = sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    deg = np.asarray(A.sum(1)).ravel()
    deg[deg == 0] = 1
    v = verts.copy()
    for _ in range(iters):
        v = (1 - lam) * v + lam * (A @ v) / deg[:, None]
    return v


def sample_at_vertices(field: np.ndarray, verts_vox: np.ndarray) -> np.ndarray:
    """Sample a per-voxel field at mesh vertices (vertex coords in voxels)."""
    from scipy.ndimage import map_coordinates
    c = np.clip(verts_vox - 0.5, 0, np.asarray(field.shape) - 1).T
    return map_coordinates(field.astype(float), c, order=1, mode="nearest")


# --------------------------------------------------------------------------
# scene
# --------------------------------------------------------------------------
@dataclass
class Scene:
    """An accumulating 3D scene rendered to a self-contained HTML file.

    Every ``add_*`` call becomes one legend entry the viewer can click to
    show/hide — that is how you go from "whole head" to "eye only" to
    "retina only" without regenerating anything.
    """

    dx: float = 1.0                     # voxel size (m)
    title: str = ""
    subtitle: str = ""
    theme: str = "dark"
    units: str = "mm"
    crop: Optional[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]] = None
    traces: List = _field(default_factory=list)
    _annotations: List[str] = _field(default_factory=list)

    # ---- unit handling ----
    @property
    def _scale(self) -> float:
        return {"m": 1.0, "mm": 1e3, "um": 1e6, "µm": 1e6,
                "voxel": None}.get(self.units, 1e3) or None

    def _spacing(self) -> float:
        return 1.0 if self._scale is None else self.dx * self._scale

    def _colors(self) -> Dict[str, str]:
        return PALETTE["dark" if self.theme == "dark" else "light"]

    # ---- structural surfaces -------------------------------------------
    def add_materials(self, labels: np.ndarray, ids: Sequence[int], name: str,
                      role: str = "context", color: Optional[str] = None,
                      opacity: float = 0.15, step: int = 1,
                      smooth: int = 1, visible: Union[bool, str] = True,
                      legendgroup: Optional[str] = None):
        """Add an isosurface around a set of material ids."""
        _require_plotly()
        import plotly.graph_objects as go

        sl = _crop_slices(self.crop)
        sub = labels[sl]
        origin = [s.start or 0 for s in sl]
        mask = np.isin(sub, np.asarray(list(ids)))
        n_vox = int(mask.sum())
        if n_vox == 0:
            return self
        verts, faces = isosurface_from_mask(mask, step=step, origin=origin,
                                            spacing=self._spacing(),
                                            smooth_iters=smooth)
        if len(verts) == 0:
            return self
        c = color or self._colors()[ROLE_COLOR.get(role, "context")]
        self.traces.append(go.Mesh3d(
            x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color=c, opacity=opacity, name=f"{name}",
            legendgroup=legendgroup or name, showlegend=True, visible=visible,
            hovertemplate=f"<b>{name}</b><br>materials {list(ids)}"
                          f"<br>{n_vox:,} voxels<extra></extra>",
            flatshading=False, lighting=dict(ambient=0.55, diffuse=0.8,
                                             specular=0.12, roughness=0.85),
        ))
        return self

    def add_electrode(self, labels: np.ndarray, ids: Sequence[int], name: str,
                      role: str = "source", opacity: float = 1.0,
                      step: int = 1, visible: Union[bool, str] = True):
        """Add an electrode surface (opaque, chromatic, always on top)."""
        return self.add_materials(labels, ids, name, role=role,
                                  opacity=opacity, step=step, smooth=0,
                                  visible=visible)

    # ---- field on a structure ------------------------------------------
    def add_field_on_surface(self, labels: np.ndarray, ids: Sequence[int],
                             field: np.ndarray, name: str,
                             cmap: Optional[str] = None, step: int = 1,
                             smooth: int = 1, opacity: float = 1.0,
                             cmin: Optional[float] = None,
                             cmax: Optional[float] = None,
                             log: bool = False, colorbar_title: str = "",
                             visible: Union[bool, str] = True):
        """Colour a tissue's surface by a per-voxel field.

        This is the workhorse for "current density **on the retina**": the
        geometry is the structure, the colour is the field.
        """
        _require_plotly()
        import plotly.graph_objects as go

        sl = _crop_slices(self.crop)
        sub = labels[sl]
        fsub = field[sl]
        origin = [s.start or 0 for s in sl]
        mask = np.isin(sub, np.asarray(list(ids)))
        if not mask.any():
            return self
        verts, faces = isosurface_from_mask(mask, step=step, origin=[0, 0, 0],
                                            spacing=1.0, smooth_iters=smooth)
        if len(verts) == 0:
            return self
        vals = sample_at_vertices(fsub, verts)
        if log:
            vals = np.log10(np.maximum(vals, np.finfo(float).tiny))
        vpos = (verts + np.asarray(origin, dtype=float)) * self._spacing()
        self.traces.append(go.Mesh3d(
            x=vpos[:, 0], y=vpos[:, 1], z=vpos[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            intensity=vals, colorscale=cmap or self._field_cmap(),
            cmin=cmin, cmax=cmax, opacity=opacity, name=name,
            legendgroup=name, showlegend=True, visible=visible,
            showscale=True,
            colorbar=dict(title=dict(text=colorbar_title or name, side="right"),
                          thickness=12, len=0.55, x=1.02,
                          tickfont=dict(size=10)),
            hovertemplate=f"<b>{name}</b><br>value %{{intensity:.3g}}<extra></extra>",
            lighting=dict(ambient=0.72, diffuse=0.6, specular=0.05),
        ))
        return self

    def _field_cmap(self) -> str:
        return FIELD_CMAP if self.theme == "dark" else FIELD_CMAP_LIGHT

    # ---- field in a volume ----------------------------------------------
    def add_field_isosurface(self, field: np.ndarray, levels: Sequence[float],
                             name: str, mask: Optional[np.ndarray] = None,
                             opacity: float = 0.35, step: int = 1,
                             smooth: int = 1, colors: Optional[Sequence] = None,
                             visible: Union[bool, str] = True):
        """Iso-contours of a field (e.g. ``|J|`` = 1, 5, 10 A/m²)."""
        _require_plotly()
        import plotly.graph_objects as go
        import plotly.express as px

        sl = _crop_slices(self.crop)
        f = np.asarray(field[sl], dtype=float)
        origin = [s.start or 0 for s in sl]
        if mask is not None:
            f = np.where(mask[sl], f, 0.0)
        ramp = colors or px.colors.sample_colorscale(
            self._field_cmap(),
            np.linspace(0.35, 0.9, len(levels)))
        for lev, col in zip(levels, ramp):
            verts, faces = isosurface_from_mask(f >= lev, step=step,
                                                origin=origin,
                                                spacing=self._spacing(),
                                                smooth_iters=smooth)
            if len(verts) == 0:
                continue
            self.traces.append(go.Mesh3d(
                x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                color=col, opacity=opacity, name=f"{name} ≥ {lev:.3g}",
                legendgroup=name, showlegend=True, visible=visible,
                hovertemplate=f"<b>{name}</b> ≥ {lev:.3g}<extra></extra>",
                lighting=dict(ambient=0.6, diffuse=0.7, specular=0.05)))
        return self

    def add_field_points(self, field: np.ndarray, name: str,
                         mask: Optional[np.ndarray] = None,
                         percentile: float = 99.0, max_points: int = 60_000,
                         size: float = 2.0, log: bool = False,
                         colorbar_title: str = "",
                         visible: Union[bool, str] = "legendonly"):
        """Scatter the hottest voxels of a field (dense-cloud view)."""
        _require_plotly()
        import plotly.graph_objects as go

        sl = _crop_slices(self.crop)
        f = np.asarray(field[sl], dtype=float)
        origin = np.array([s.start or 0 for s in sl], dtype=float)
        sel = np.ones(f.shape, dtype=bool) if mask is None else mask[sl].copy()
        if not sel.any():
            return self
        thr = np.percentile(f[sel], percentile)
        sel &= f >= thr
        idx = np.argwhere(sel)
        if len(idx) == 0:
            return self
        if len(idx) > max_points:
            keep = np.random.default_rng(0).choice(len(idx), max_points,
                                                   replace=False)
            idx = idx[keep]
        vals = f[idx[:, 0], idx[:, 1], idx[:, 2]]
        if log:
            vals = np.log10(np.maximum(vals, np.finfo(float).tiny))
        p = (idx + 0.5 + origin) * self._spacing()
        self.traces.append(go.Scatter3d(
            x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="markers",
            marker=dict(size=size, color=vals, colorscale=self._field_cmap(),
                        opacity=0.85, showscale=True,
                        colorbar=dict(title=dict(text=colorbar_title or name,
                                                 side="right"),
                                      thickness=12, len=0.55, x=1.02)),
            name=f"{name} (top {100-percentile:g}%)", legendgroup=name,
            showlegend=True, visible=visible,
            hovertemplate=f"<b>{name}</b> %{{marker.color:.3g}}<extra></extra>"))
        return self

    def add_slice(self, field: np.ndarray, axis: str, index: int, name: str,
                  mask: Optional[np.ndarray] = None, log: bool = False,
                  cmin: Optional[float] = None, cmax: Optional[float] = None,
                  opacity: float = 1.0, colorbar_title: str = "",
                  visible: Union[bool, str] = True):
        """A cut-plane through the volume, textured by the field."""
        _require_plotly()
        import plotly.graph_objects as go

        a = "xyz".index(axis)
        sl = list(_crop_slices(self.crop))
        f = np.asarray(field[tuple(sl)], dtype=float)
        origin = np.array([s.start or 0 for s in sl], dtype=float)
        loc = index - origin[a]
        if not (0 <= loc < f.shape[a]):
            return self
        take = [slice(None)] * 3
        take[a] = int(loc)
        plane = f[tuple(take)]
        if mask is not None:
            mplane = mask[tuple(sl)][tuple(take)]
            plane = np.where(mplane, plane, np.nan)
        if log:
            plane = np.log10(np.maximum(plane, np.finfo(float).tiny))
        rem = [i for i in range(3) if i != a]
        sp = self._spacing()
        u = (np.arange(f.shape[rem[0]]) + 0.5 + origin[rem[0]]) * sp
        v = (np.arange(f.shape[rem[1]]) + 0.5 + origin[rem[1]]) * sp
        U, V = np.meshgrid(u, v, indexing="ij")
        const = np.full(U.shape, (index + 0.5) * sp)
        coords = {}
        coords["xyz"[a]] = const
        coords["xyz"[rem[0]]] = U
        coords["xyz"[rem[1]]] = V
        self.traces.append(go.Surface(
            x=coords["x"], y=coords["y"], z=coords["z"],
            surfacecolor=plane, colorscale=self._field_cmap(),
            cmin=cmin, cmax=cmax, opacity=opacity, name=name,
            legendgroup=name, showlegend=True, visible=visible,
            showscale=True,
            colorbar=dict(title=dict(text=colorbar_title or name, side="right"),
                          thickness=12, len=0.55, x=1.02),
            hovertemplate=f"<b>{name}</b> %{{surfacecolor:.3g}}<extra></extra>",
            lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0)))
        return self

    def add_note(self, text: str):
        self._annotations.append(text)
        return self

    # ---- neuron morphology ----------------------------------------------
    def add_neuron(self, points_vox: np.ndarray, edges: np.ndarray, name: str,
                  values: Optional[np.ndarray] = None,
                  cmap: Optional[str] = None,
                  cmin: Optional[float] = None, cmax: Optional[float] = None,
                  colorbar_title: str = "", role: str = "target",
                  color: Optional[str] = None, line_width: float = 4.0,
                  visible: Union[bool, str] = True):
        """A morphology skeleton as connected line segments.

        ``points_vox``/``edges`` are typically an SWC trace (dense, real
        anatomy — see :mod:`neuroam.morphology`) or a NEURON section chain
        (:func:`neuroam.neuron_link.get_segment_edges`), in the same voxel
        frame as everything else in this scene. ``values`` colors the
        skeleton per point (e.g. simulated Vm nearest-neighbor-mapped onto
        the trace) with this scene's field colormap; without it, the whole
        skeleton is one flat structural color (``role``).
        """
        _require_plotly()
        import plotly.graph_objects as go

        p = np.asarray(points_vox, dtype=float) * self._spacing()
        e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
        if len(e) == 0:
            return self
        # one disjoint-segment polyline per plotly's NaN-break idiom
        xs = np.full(3 * len(e), np.nan)
        ys = np.full(3 * len(e), np.nan)
        zs = np.full(3 * len(e), np.nan)
        xs[0::3], xs[1::3] = p[e[:, 0], 0], p[e[:, 1], 0]
        ys[0::3], ys[1::3] = p[e[:, 0], 1], p[e[:, 1], 1]
        zs[0::3], zs[1::3] = p[e[:, 0], 2], p[e[:, 1], 2]

        line: Dict = dict(width=line_width)
        if values is not None:
            v = np.asarray(values, dtype=float)
            cs = np.full(3 * len(e), np.nan)
            cs[0::3], cs[1::3] = v[e[:, 0]], v[e[:, 1]]
            line.update(color=cs, colorscale=cmap or self._field_cmap(),
                       cmin=cmin, cmax=cmax, showscale=True,
                       colorbar=dict(title=dict(text=colorbar_title or name,
                                                side="right"),
                                    thickness=12, len=0.55, x=1.02))
        else:
            line["color"] = color or self._colors()[ROLE_COLOR.get(role, "context")]

        self.traces.append(go.Scatter3d(
            x=xs, y=ys, z=zs, mode="lines", line=line, name=name,
            legendgroup=name, showlegend=True, visible=visible,
            hoverinfo="skip"))
        return self

    # ---- output ---------------------------------------------------------
    def figure(self):
        _require_plotly()
        import plotly.graph_objects as go
        col = self._colors()
        ax_lab = "voxels" if self._scale is None else self.units
        axis_cfg = dict(showbackground=False, gridcolor=col["grid"],
                        zerolinecolor=col["grid"],
                        color=col["text2"], showspikes=False)
        sub = self.subtitle
        if self._annotations:
            sub = (sub + "<br>" if sub else "") + " · ".join(self._annotations)
        fig = go.Figure(data=self.traces)
        fig.update_layout(
            title=dict(
                text=(f"<b>{self.title}</b>" +
                      (f"<br><span style='font-size:12px;color:{col['text2']}'>"
                       f"{sub}</span>" if sub else "")),
                x=0.02, xanchor="left", font=dict(size=17, color=col["text"])),
            scene=dict(
                xaxis=dict(title=f"x ({ax_lab})", **axis_cfg),
                yaxis=dict(title=f"y ({ax_lab})", **axis_cfg),
                zaxis=dict(title=f"z ({ax_lab})", **axis_cfg),
                aspectmode="data",
                camera=dict(eye=dict(x=1.6, y=1.5, z=1.0)),
                bgcolor=col["surface"]),
            paper_bgcolor=col["surface"],
            font=dict(color=col["text"], size=12,
                      family="system-ui, -apple-system, Segoe UI, sans-serif"),
            legend=dict(title=dict(text="click to show / hide"),
                        bgcolor="rgba(0,0,0,0)", itemsizing="constant",
                        font=dict(color=col["text"], size=11),
                        x=0.0, y=0.98, xanchor="left", yanchor="top"),
            margin=dict(l=0, r=90, t=70 if not sub else 86, b=0),
        )
        return fig

    def write_html(self, path, include_plotlyjs: str = "cdn",
                   auto_open: bool = False) -> Path:
        fig = self.figure()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(path), include_plotlyjs=include_plotlyjs,
                       full_html=True, auto_open=auto_open,
                       config={"displaylogo": False,
                               "toImageButtonOptions": {"scale": 2}})
        return path


# --------------------------------------------------------------------------
# high-level convenience views
# --------------------------------------------------------------------------
def view_model(model, out_html, context_ids: Optional[Sequence[int]] = None,
               focus: Optional[Dict[str, Sequence[int]]] = None,
               source_ids=(101,), ground_ids=(100, 102),
               insulation_ids=(110,), step: int = 1,
               secondary_step: Optional[int] = None,
               context_step: Optional[int] = None,
               title: str = "", subtitle: str = "", theme: str = "dark",
               crop=None, units: str = "mm") -> Path:
    """Model + electrodes, ready to inspect before running a simulation.

    ``focus`` maps a display name to material ids, e.g.
    ``{"retina": [77], "vitreous": [55]}`` — each becomes a toggleable layer.
    Everything else defaults to a single ghosted "head tissue" surface.
    """
    labels = model.labels
    focus = focus or {}
    named = set()
    for ids in focus.values():
        named.update(ids)
    named.update(source_ids); named.update(ground_ids); named.update(insulation_ids)

    present = set(np.unique(labels).tolist())
    if context_ids is None:
        context_ids = sorted(present - named - {0})

    sc = Scene(dx=model.dx, title=title or f"{model.name} — model & electrodes",
               subtitle=subtitle, theme=theme, crop=crop, units=units)
    sstep = secondary_step if secondary_step is not None else max(step, 2)
    cstep = context_step if context_step is not None else max(step, 3)
    if context_ids:
        sc.add_materials(labels, context_ids, "head tissue (context)",
                         role="context", opacity=0.10, step=cstep, smooth=2)
    # target/focus structures: first gets the chromatic "target" slot,
    # the rest are lighter greys so identity stays legible.
    for i, (nm, ids) in enumerate(focus.items()):
        sc.add_materials(labels, ids, nm,
                         role="target" if i == 0 else "context",
                         opacity=0.9 if i == 0 else 0.22,
                         step=step if i == 0 else sstep, smooth=1)
    for nm, ids, role in (("stimulating electrode", source_ids, "source"),
                          ("return electrode", ground_ids, "ground"),
                          ("insulation", insulation_ids, "insulation")):
        ids = [i for i in ids if i in present]
        if ids:
            sc.add_electrode(labels, ids, nm, role=role,
                             opacity=1.0 if role != "insulation" else 0.55,
                             step=1)
    sc.add_note("legend entries toggle layers")
    return sc.write_html(out_html)


def view_current_density(model, Jmag, out_html,
                         focus: Optional[Dict[str, Sequence[int]]] = None,
                         context_ids: Optional[Sequence[int]] = None,
                         source_ids=(101,), ground_ids=(100, 102),
                         insulation_ids=(110,),
                         iso_levels: Optional[Sequence[float]] = None,
                         slices: Sequence[Tuple[str, int]] = (),
                         step: int = 1, secondary_step: Optional[int] = None,
                         context_step: Optional[int] = None, log: bool = True,
                         title: str = "", subtitle: str = "",
                         theme: str = "dark", crop=None,
                         units: str = "mm",
                         unit_label: str = "|J| (A/m²)") -> Path:
    """Current-density view: field painted on the focus structures.

    Layers produced (all toggleable):
      * ghosted head tissue (context)
      * each focus structure, **coloured by** ``|J|`` **on its surface**
      * ``|J|`` isosurfaces inside the first focus structure
      * optional cut-planes
      * electrodes
    """
    labels = model.labels
    focus = focus or {}
    named = set()
    for ids in focus.values():
        named.update(ids)
    named.update(source_ids); named.update(ground_ids); named.update(insulation_ids)
    present = set(np.unique(labels).tolist())
    if context_ids is None:
        context_ids = sorted(present - named - {0})

    sc = Scene(dx=model.dx, title=title or f"{model.name} — current density",
               subtitle=subtitle, theme=theme, crop=crop, units=units)
    sstep = secondary_step if secondary_step is not None else max(step, 2)
    cstep = context_step if context_step is not None else max(step, 3)
    if context_ids:
        sc.add_materials(labels, context_ids, "head tissue (context)",
                         role="context", opacity=0.07, step=cstep,
                         smooth=2, visible="legendonly")

    lab = ("log₁₀ " + unit_label) if log else unit_label
    for i, (nm, ids) in enumerate(focus.items()):
        m = np.isin(labels, np.asarray(list(ids)))
        if not m.any():
            continue
        vals = Jmag[m]
        lo, hi = np.percentile(vals, [2, 99.5])
        if log:
            lo, hi = np.log10(max(lo, 1e-30)), np.log10(max(hi, 1e-30))
        sc.add_field_on_surface(labels, ids, Jmag, f"{nm} — {unit_label}",
                                step=step if i == 0 else sstep, log=log,
                                cmin=lo, cmax=hi,
                                colorbar_title=lab, visible=(i == 0))

    if focus and iso_levels:
        first = list(focus.values())[0]
        sc.add_field_isosurface(Jmag, iso_levels, "|J| isosurface",
                                mask=np.isin(labels, np.asarray(list(first))),
                                step=step, visible="legendonly")
    for ax, ix in slices:
        sc.add_slice(Jmag, ax, ix, f"cut-plane {ax}={ix}", log=log,
                     colorbar_title=lab, visible="legendonly")

    for nm, ids, role in (("stimulating electrode", source_ids, "source"),
                          ("return electrode", ground_ids, "ground"),
                          ("insulation", insulation_ids, "insulation")):
        ids = [i for i in ids if i in present]
        if ids:
            sc.add_electrode(labels, ids, nm, role=role,
                             opacity=1.0 if role != "insulation" else 0.5)
    sc.add_note("legend entries toggle layers")
    return sc.write_html(out_html)


# --------------------------------------------------------------------------
# neuron registration views
# --------------------------------------------------------------------------
def view_neuron_context(model, points_vox: np.ndarray, edges: np.ndarray,
                        out_html, neuron_name: str = "registered neuron",
                        focus: Optional[Dict[str, Sequence[int]]] = None,
                        context_ids: Optional[Sequence[int]] = None,
                        source_ids=(101,), ground_ids=(100, 102),
                        insulation_ids=(110,), step: int = 2,
                        secondary_step: Optional[int] = None,
                        context_step: Optional[int] = None,
                        title: str = "", subtitle: str = "",
                        theme: str = "dark", units: str = "mm",
                        marker_radius_vox: Optional[float] = None) -> Path:
    """Where a registered neuron sits inside the full model.

    Identical layer set to :func:`view_model` (ghosted head, focus
    structures, electrodes) plus the morphology itself. At full-head scale
    the morphology alone is easy to lose, so a small sphere is added at its
    soma/root (``points_vox[0]``) in a distinct color — findable at any
    zoom, then toggle it off once you've zoomed in.
    """
    labels = model.labels
    focus = focus or {}
    named = set()
    for ids in focus.values():
        named.update(ids)
    named.update(source_ids); named.update(ground_ids); named.update(insulation_ids)
    present = set(np.unique(labels).tolist())
    if context_ids is None:
        context_ids = sorted(present - named - {0})

    sc = Scene(dx=model.dx, title=title or f"{model.name} — neuron location",
              subtitle=subtitle, theme=theme, units=units)
    sstep = secondary_step if secondary_step is not None else max(step, 2)
    cstep = context_step if context_step is not None else max(step, 3)
    if context_ids:
        sc.add_materials(labels, context_ids, "head tissue (context)",
                         role="context", opacity=0.08, step=cstep, smooth=2)
    for i, (nm, ids) in enumerate(focus.items()):
        sc.add_materials(labels, ids, nm,
                         role="target" if i == 0 else "context",
                         opacity=0.85 if i == 0 else 0.2,
                         step=step if i == 0 else sstep, smooth=1)
    for nm, ids, role in (("stimulating electrode", source_ids, "source"),
                          ("return electrode", ground_ids, "ground"),
                          ("insulation", insulation_ids, "insulation")):
        ids = [i for i in ids if i in present]
        if ids:
            sc.add_electrode(labels, ids, nm, role=role,
                             opacity=1.0 if role != "insulation" else 0.5)

    sc.add_neuron(points_vox, edges, neuron_name, role="target",
                 color="#ffd23f", line_width=3.5)
    r = marker_radius_vox or max(2.0, 0.01 * max(model.world))
    root = np.asarray(points_vox[0], dtype=float)
    p0 = root * sc._spacing()
    theta, phi = np.mgrid[0:np.pi:12j, 0:2 * np.pi:16j]
    sx = p0[0] + r * sc._spacing() * np.sin(theta) * np.cos(phi)
    sy = p0[1] + r * sc._spacing() * np.sin(theta) * np.sin(phi)
    sz = p0[2] + r * sc._spacing() * np.cos(theta)
    import plotly.graph_objects as go
    sc.traces.append(go.Surface(
        x=sx, y=sy, z=sz, surfacecolor=np.ones_like(sx),
        colorscale=[[0, "#ffd23f"], [1, "#ffd23f"]], showscale=False,
        name=f"{neuron_name} (locator)", legendgroup=f"{neuron_name} (locator)",
        showlegend=True, opacity=0.9,
        hovertemplate=f"<b>{neuron_name} soma</b><extra></extra>",
        lighting=dict(ambient=0.8, diffuse=0.4, specular=0.3)))
    sc.add_note("legend entries toggle layers — locator sphere marks the soma")
    return sc.write_html(out_html)


def view_neuron_placement_editor(model, points_vox: np.ndarray, edges: np.ndarray,
                                 out_html, neuron_name: str = "registered neuron",
                                 focus: Optional[Dict[str, Sequence[int]]] = None,
                                 context_ids: Optional[Sequence[int]] = None,
                                 source_ids=(101,), ground_ids=(100, 102),
                                 insulation_ids=(110,), step: int = 2,
                                 secondary_step: Optional[int] = None,
                                 context_step: Optional[int] = None,
                                 title: str = "", subtitle: str = "",
                                 theme: str = "dark", units: str = "mm",
                                 translate_range_vox: float = 60.0,
                                 plotlyjs_src: str = "cdn") -> Path:
    """Same layers as :func:`view_neuron_context`, plus live sliders to
    translate/rotate the morphology in the browser — so a placement can be
    found *visually*, before ever running NEURON.

    Plotly has no built-in drag/rotate gizmo for a single object, so this
    builds one out of ordinary HTML range inputs plus a small hand-written
    ``Plotly.restyle`` call: moving a slider recomputes every point's
    position client-side (the exact same rigid transform as
    :func:`neuroam.coupling.transform_coordinates` — translate + intrinsic
    Rx·Ry·Rz about the point cloud's own centroid) and pushes the new
    coordinates into the neuron trace and a soma marker, live. Nothing here
    talks back to Python — there is no server. The payoff is the live
    readout: it prints the exact ``translate=..., rotate_deg=..., pivot=...``
    call to make in Python once a placement looks right, so the number you
    saw is the number that actually runs.

    ``plotlyjs_src`` defaults to Plotly's own CDN (fine for a local file
    opened directly); pass an explicit URL — e.g. a pinned cdnjs build —
    when hosting this page somewhere with a stricter script allowlist.
    """
    import json

    _require_plotly()
    import plotly.graph_objects as go

    labels = model.labels
    focus = focus or {}
    named = set()
    for ids in focus.values():
        named.update(ids)
    named.update(source_ids); named.update(ground_ids); named.update(insulation_ids)
    present = set(np.unique(labels).tolist())
    if context_ids is None:
        context_ids = sorted(present - named - {0})

    sc = Scene(dx=model.dx, title=title or f"{model.name} — neuron placement editor",
              subtitle=subtitle, theme=theme, units=units)
    sstep = secondary_step if secondary_step is not None else max(step, 2)
    cstep = context_step if context_step is not None else max(step, 3)
    if context_ids:
        sc.add_materials(labels, context_ids, "head tissue (context)",
                         role="context", opacity=0.08, step=cstep, smooth=2)
    for i, (nm, ids) in enumerate(focus.items()):
        sc.add_materials(labels, ids, nm,
                         role="target" if i == 0 else "context",
                         opacity=0.85 if i == 0 else 0.2,
                         step=step if i == 0 else sstep, smooth=1)
    for nm, ids, role in (("stimulating electrode", source_ids, "source"),
                          ("return electrode", ground_ids, "ground"),
                          ("insulation", insulation_ids, "insulation")):
        ids = [i for i in ids if i in present]
        if ids:
            sc.add_electrode(labels, ids, nm, role=role,
                             opacity=1.0 if role != "insulation" else 0.5)

    neuron_trace_idx = len(sc.traces)
    sc.add_neuron(points_vox, edges, neuron_name, role="target",
                 color="#ffd23f", line_width=3.5)
    marker_trace_idx = len(sc.traces)
    sp = sc._spacing()
    p0 = np.asarray(points_vox[0], dtype=float) * sp
    sc.traces.append(go.Scatter3d(
        x=[p0[0]], y=[p0[1]], z=[p0[2]], mode="markers",
        marker=dict(size=7, color="#ffd23f", symbol="diamond"),
        name=f"{neuron_name} (soma)", legendgroup=f"{neuron_name} (soma)",
        showlegend=True, hovertemplate=f"<b>{neuron_name} soma</b><extra></extra>"))
    sc.add_note("drag the sliders below to preview a placement")

    fig = sc.figure()
    div_id = "neuron-placement-plot"
    plot_html = fig.to_html(full_html=False, include_plotlyjs=plotlyjs_src,
                            div_id=div_id, config={"displaylogo": False})

    col = PALETTE["dark" if theme == "dark" else "light"]
    pivot = np.asarray(points_vox, dtype=float).mean(axis=0)
    edges_arr = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    payload = dict(points=np.asarray(points_vox, dtype=float).tolist(),
                  edges=edges_arr.tolist(), pivot=pivot.tolist(), spacing=sp)

    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>{neuron_name} — placement editor</title>
<style>
  body {{ margin:0; background:{col['surface']}; color:{col['text']};
        font-family: system-ui, -apple-system, Segoe UI, sans-serif; }}
  #panel {{ position:fixed; top:0; left:0; right:0; z-index:10; padding:10px 16px;
          background:{col['surface']}; border-bottom:1px solid {col['grid']};
          display:flex; gap:22px; flex-wrap:wrap; align-items:flex-start; }}
  #plot {{ position:absolute; top:96px; left:0; right:0; bottom:0; }}
  .group {{ display:flex; flex-direction:column; gap:2px; }}
  .group h4 {{ margin:0 0 2px; font-size:11px; letter-spacing:.06em;
             text-transform:uppercase; color:{col['text2']}; font-weight:600; }}
  .row {{ display:flex; align-items:center; gap:6px; font-size:12px; }}
  .row label {{ width:14px; color:{col['text2']}; }}
  .row input[type=range] {{ width:150px; }}
  .row output {{ width:52px; text-align:right; font-variant-numeric:tabular-nums; }}
  #readout {{ font-family: ui-monospace, Consolas, monospace; font-size:12px;
             background:rgba(128,128,128,.12); border-radius:6px; padding:8px 10px;
             white-space:pre; max-width:520px; }}
  button {{ background:{col['context']}; color:{col['surface']}; border:none;
          border-radius:5px; padding:5px 10px; font-size:12px; cursor:pointer; }}
</style></head>
<body>
<div id="panel">
  <div class="group"><h4>translate (voxels)</h4>
    <div class="row"><label>x</label><input type="range" id="tx" min="{-translate_range_vox}" max="{translate_range_vox}" value="0" step="0.5"><output id="tx_v">0</output></div>
    <div class="row"><label>y</label><input type="range" id="ty" min="{-translate_range_vox}" max="{translate_range_vox}" value="0" step="0.5"><output id="ty_v">0</output></div>
    <div class="row"><label>z</label><input type="range" id="tz" min="{-translate_range_vox}" max="{translate_range_vox}" value="0" step="0.5"><output id="tz_v">0</output></div>
  </div>
  <div class="group"><h4>rotate (degrees, about centroid)</h4>
    <div class="row"><label>x</label><input type="range" id="rx" min="-180" max="180" value="0" step="1"><output id="rx_v">0</output></div>
    <div class="row"><label>y</label><input type="range" id="ry" min="-180" max="180" value="0" step="1"><output id="ry_v">0</output></div>
    <div class="row"><label>z</label><input type="range" id="rz" min="-180" max="180" value="0" step="1"><output id="rz_v">0</output></div>
  </div>
  <div class="group"><h4>copy into Python once it looks right</h4>
    <div id="readout"></div>
    <div style="margin-top:4px"><button id="reset">reset</button></div>
  </div>
</div>
<div id="plot">{plot_html}</div>
<script>
const DATA = {json.dumps(payload)};
const NEURON_TRACE = {neuron_trace_idx};
const MARKER_TRACE = {marker_trace_idx};
const SP = DATA.spacing;

function deg2rad(d) {{ return d * Math.PI / 180; }}

function matMul3(A, B) {{
  const C = [[0,0,0],[0,0,0],[0,0,0]];
  for (let i=0;i<3;i++) for (let j=0;j<3;j++) for (let k=0;k<3;k++) C[i][j]+=A[i][k]*B[k][j];
  return C;
}}
function matVec3(A, v) {{
  return [A[0][0]*v[0]+A[0][1]*v[1]+A[0][2]*v[2],
         A[1][0]*v[0]+A[1][1]*v[1]+A[1][2]*v[2],
         A[2][0]*v[0]+A[2][1]*v[1]+A[2][2]*v[2]];
}}

function rotationMatrix(rxD, ryD, rzD) {{
  const rx = deg2rad(rxD), ry = deg2rad(ryD), rz = deg2rad(rzD);
  const Rx = [[1,0,0],[0,Math.cos(rx),-Math.sin(rx)],[0,Math.sin(rx),Math.cos(rx)]];
  const Ry = [[Math.cos(ry),0,Math.sin(ry)],[0,1,0],[-Math.sin(ry),0,Math.cos(ry)]];
  const Rz = [[Math.cos(rz),-Math.sin(rz),0],[Math.sin(rz),Math.cos(rz),0],[0,0,1]];
  return matMul3(matMul3(Rz, Ry), Rx);
}}

function currentParams() {{
  return {{
    tx: parseFloat(document.getElementById('tx').value),
    ty: parseFloat(document.getElementById('ty').value),
    tz: parseFloat(document.getElementById('tz').value),
    rx: parseFloat(document.getElementById('rx').value),
    ry: parseFloat(document.getElementById('ry').value),
    rz: parseFloat(document.getElementById('rz').value),
  }};
}}

function applyTransform() {{
  const p = currentParams();
  const R = rotationMatrix(p.rx, p.ry, p.rz);
  const pivot = DATA.pivot;
  const translate = [p.tx, p.ty, p.tz];

  const transformed = DATA.points.map(pt => {{
    const rel = [pt[0]-pivot[0], pt[1]-pivot[1], pt[2]-pivot[2]];
    const rot = matVec3(R, rel);
    return [rot[0]+pivot[0]+translate[0], rot[1]+pivot[1]+translate[1], rot[2]+pivot[2]+translate[2]];
  }});

  const xs = [], ys = [], zs = [];
  for (const [a, b] of DATA.edges) {{
    xs.push(transformed[a][0]*SP, transformed[b][0]*SP, null);
    ys.push(transformed[a][1]*SP, transformed[b][1]*SP, null);
    zs.push(transformed[a][2]*SP, transformed[b][2]*SP, null);
  }}
  Plotly.restyle('{div_id}', {{x: [xs], y: [ys], z: [zs]}}, [NEURON_TRACE]);

  const soma = transformed[0];
  Plotly.restyle('{div_id}', {{x: [[soma[0]*SP]], y: [[soma[1]*SP]], z: [[soma[2]*SP]]}}, [MARKER_TRACE]);

  const pv = pivot.map(v => v.toFixed(2));
  document.getElementById('readout').textContent =
    `translate_vox = (${{p.tx.toFixed(2)}}, ${{p.ty.toFixed(2)}}, ${{p.tz.toFixed(2)}})\\n` +
    `rotate_deg    = (${{p.rx.toFixed(1)}}, ${{p.ry.toFixed(1)}}, ${{p.rz.toFixed(1)}})\\n` +
    `pivot_vox     = (${{pv[0]}}, ${{pv[1]}}, ${{pv[2]}})\\n\\n` +
    `coords_vox = transform_coordinates(\\n` +
    `    coords_vox,\\n` +
    `    translate=(${{p.tx.toFixed(2)}}, ${{p.ty.toFixed(2)}}, ${{p.tz.toFixed(2)}}),\\n` +
    `    rotate_deg=(${{p.rx.toFixed(1)}}, ${{p.ry.toFixed(1)}}, ${{p.rz.toFixed(1)}}),\\n` +
    `    pivot=(${{pv[0]}}, ${{pv[1]}}, ${{pv[2]}}))`;

  for (const id of ['tx','ty','tz','rx','ry','rz']) {{
    document.getElementById(id+'_v').textContent = document.getElementById(id).value;
  }}
}}

for (const id of ['tx','ty','tz','rx','ry','rz']) {{
  document.getElementById(id).addEventListener('input', applyTransform);
}}
document.getElementById('reset').addEventListener('click', () => {{
  for (const id of ['tx','ty','tz','rx','ry','rz']) document.getElementById(id).value = 0;
  applyTransform();
}});
applyTransform();
</script>
</body></html>""", encoding="utf-8")
    return out_html


def view_neuron_field(model, field: np.ndarray, points_vox: np.ndarray,
                      edges: np.ndarray, out_html,
                      neuron_name: str = "registered neuron",
                      pad_vox: float = 25.0, iso_levels: Optional[Sequence[float]] = None,
                      slices: Sequence[str] = ("z",), log: bool = True,
                      title: str = "", subtitle: str = "", theme: str = "dark",
                      units: str = "um", unit_label: str = "|J| (A/m²)",
                      values: Optional[np.ndarray] = None,
                      values_colorbar_title: str = "Vm (mV)") -> Path:
    """Zoomed-in view: the tissue field immediately around a registered
    neuron, cropped tight so the field's local structure (not the whole
    head) is what fills the screen.

    ``field`` is a per-voxel array already in the model's own grid (e.g.
    ``|J|`` from :func:`neuroam.fields.current_density`, or a raw component).
    If ``values`` is given (a Vm-like array matching ``points_vox``), the
    morphology is colored by it instead of a flat color — combining "the
    field it sat in" with "how it responded" in one figure.
    """
    lo = np.floor(points_vox.min(axis=0) - pad_vox).astype(int)
    hi = np.ceil(points_vox.max(axis=0) + pad_vox).astype(int)
    lo = np.clip(lo, 0, np.asarray(model.world))
    hi = np.clip(hi, 0, np.asarray(model.world))
    crop = tuple((int(lo[a]), int(hi[a])) for a in range(3))

    sc = Scene(dx=model.dx, title=title or f"{model.name} — field around neuron",
              subtitle=subtitle, theme=theme, crop=crop, units=units)
    labels = model.labels
    present_ids = sorted(set(np.unique(labels[crop[0][0]:crop[0][1],
                                              crop[1][0]:crop[1][1],
                                              crop[2][0]:crop[2][1]]).tolist()) - {0})
    if present_ids:
        vals = field[np.isin(labels, present_ids)]
        vals = vals[np.isfinite(vals) & (vals > 0)]
        clo, chi = (np.percentile(vals, [2, 99.5]) if len(vals)
                   else (None, None))
        if log and clo is not None:
            clo, chi = np.log10(max(clo, 1e-30)), np.log10(max(chi, 1e-30))
        sc.add_field_on_surface(labels, present_ids, field,
                                f"tissue — {unit_label}", log=log,
                                cmin=clo, cmax=chi, colorbar_title=unit_label,
                                opacity=0.55)
    if iso_levels:
        sc.add_field_isosurface(field, iso_levels, "|J| isosurface",
                                visible="legendonly")
    for ax in slices:
        mid = {"x": (crop[0][0] + crop[0][1]) // 2,
              "y": (crop[1][0] + crop[1][1]) // 2,
              "z": (crop[2][0] + crop[2][1]) // 2}[ax]
        sc.add_slice(field, ax, mid, f"cut-plane {ax}={mid}", log=log,
                    colorbar_title=unit_label, visible="legendonly")

    sc.add_neuron(points_vox, edges, neuron_name, values=values,
                 colorbar_title=values_colorbar_title if values is not None else "",
                 role="target", color="#ffd23f", line_width=5.0)
    sc.add_note("cropped to the neuron's bounding box + padding")
    return sc.write_html(out_html)


def animate_neuron_activity(points_vox: np.ndarray, edges: np.ndarray,
                            t_ms: np.ndarray, values_over_time: np.ndarray,
                            out_html, name: str = "Vm", units: str = "um",
                            dx: float = 1.0, cmap: Optional[str] = None,
                            cmin: Optional[float] = None,
                            cmax: Optional[float] = None,
                            colorbar_title: str = "Vm (mV)",
                            max_frames: int = 150, theme: str = "dark",
                            title: str = "", subtitle: str = "",
                            frame_ms: int = 60) -> Path:
    """The morphology, colored by a per-point value that changes over time —
    a play/scrub-able record of simulated activity across the real
    morphology, the way you'd want to *see* a wave of depolarization
    actually move.

    ``values_over_time``: (T, n_points) matching ``points_vox`` row-for-row
    (typically Vm, nearest-neighbor-mapped from simulated segments onto a
    dense SWC trace via :func:`neuroam.morphology.nearest_values`, once per
    saved time step). Downsampled to at most ``max_frames`` frames.
    """
    _require_plotly()
    import plotly.graph_objects as go

    T = values_over_time.shape[0]
    idx = np.unique(np.linspace(0, T - 1, min(max_frames, T)).astype(int))
    scale = {"m": 1.0, "mm": 1e3, "um": 1e6, "µm": 1e6}.get(units, 1e6) * dx
    p = np.asarray(points_vox, dtype=float) * scale
    e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)

    xs = np.full(3 * len(e), np.nan)
    ys = np.full(3 * len(e), np.nan)
    zs = np.full(3 * len(e), np.nan)
    xs[0::3], xs[1::3] = p[e[:, 0], 0], p[e[:, 1], 0]
    ys[0::3], ys[1::3] = p[e[:, 0], 1], p[e[:, 1], 1]
    zs[0::3], zs[1::3] = p[e[:, 0], 2], p[e[:, 1], 2]

    if cmin is None or cmax is None:
        cmin = float(np.nanmin(values_over_time)) if cmin is None else cmin
        cmax = float(np.nanmax(values_over_time)) if cmax is None else cmax

    def color_for(t_i):
        v = values_over_time[t_i]
        cs = np.full(3 * len(e), np.nan)
        cs[0::3], cs[1::3] = v[e[:, 0]], v[e[:, 1]]
        return cs

    col = PALETTE["dark" if theme == "dark" else "light"]
    base = go.Scatter3d(
        x=xs, y=ys, z=zs, mode="lines",
        line=dict(width=5, color=color_for(idx[0]),
                 colorscale=cmap or FIELD_CMAP, cmin=cmin, cmax=cmax,
                 showscale=True,
                 colorbar=dict(title=dict(text=colorbar_title, side="right"),
                              thickness=12, len=0.55, x=1.02)),
        name=name, hoverinfo="skip")

    frames = [go.Frame(name=str(i),
                       data=[go.Scatter3d(line=dict(color=color_for(i)))],
                       traces=[0])
             for i in idx]

    fig = go.Figure(data=[base], frames=frames)
    ax_lab = units
    axis_cfg = dict(showbackground=False, gridcolor=col["grid"],
                    zerolinecolor=col["grid"], color=col["text2"],
                    showspikes=False)
    fig.update_layout(
        title=dict(text=(f"<b>{title or name + ' over time'}</b>" +
                         (f"<br><span style='font-size:12px;color:{col['text2']}'>"
                          f"{subtitle}</span>" if subtitle else "")),
                  x=0.02, xanchor="left", font=dict(size=17, color=col["text"])),
        scene=dict(xaxis=dict(title=f"x ({ax_lab})", **axis_cfg),
                  yaxis=dict(title=f"y ({ax_lab})", **axis_cfg),
                  zaxis=dict(title=f"z ({ax_lab})", **axis_cfg),
                  aspectmode="data", bgcolor=col["surface"]),
        paper_bgcolor=col["surface"],
        font=dict(color=col["text"], size=12,
                 family="system-ui, -apple-system, Segoe UI, sans-serif"),
        margin=dict(l=0, r=90, t=70, b=60),
        updatemenus=[dict(
            type="buttons", showactive=False, x=0.02, y=0.02, xanchor="left",
            buttons=[
                dict(label="▶ play", method="animate",
                    args=[None, dict(frame=dict(duration=frame_ms, redraw=True),
                                     fromcurrent=True, transition=dict(duration=0))]),
                dict(label="⏸ pause", method="animate",
                    args=[[None], dict(frame=dict(duration=0, redraw=False),
                                       mode="immediate")]),
            ])],
        sliders=[dict(
            x=0.02, y=-0.02, len=0.94, xanchor="left",
            currentvalue=dict(prefix="t = ", suffix=" ms",
                             font=dict(color=col["text2"], size=11)),
            font=dict(color=col["text2"]),
            steps=[dict(method="animate", label=f"{t_ms[i]:.2f}",
                       args=[[str(i)], dict(mode="immediate",
                                            frame=dict(duration=0, redraw=True))])
                  for i in idx])],
    )
    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html), include_plotlyjs="cdn", full_html=True,
                  config={"displaylogo": False})
    return out_html


# --------------------------------------------------------------------------
# static snapshots (for reports / chat previews)
# --------------------------------------------------------------------------
def snapshot_png(path, model, field: Optional[np.ndarray] = None,
                 focus_ids: Optional[Sequence[int]] = None,
                 electrode_ids: Sequence[int] = (101, 102),
                 centre: Optional[Sequence[int]] = None,
                 log: bool = True, title: str = "",
                 unit_label: str = "|J| (A/m²)",
                 scale_ids: Optional[Sequence[int]] = None,
                 clip_percentiles: Tuple[float, float] = (1.0, 99.5),
                 mask_to_focus: bool = False) -> Path:
    """Three orthogonal panels + a 3D view, as a static PNG.

    The colour range is taken from **tissue only** (electrode metal is ~10⁷×
    more conductive and would otherwise flatten all tissue contrast).
    ``scale_ids`` overrides which materials set the range.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    labels = model.labels
    if centre is None:
        m = np.isin(labels, np.asarray(list(electrode_ids))) if \
            np.isin(labels, np.asarray(list(electrode_ids))).any() else labels > 0
        idx = np.argwhere(m)
        centre = idx.mean(0).astype(int) if len(idx) else \
            np.array(labels.shape) // 2
    centre = np.asarray(centre, int)

    fig = plt.figure(figsize=(12, 3.6), dpi=140)
    col = PALETTE["light"]
    fig.patch.set_facecolor(col["surface"])
    axes = [fig.add_subplot(1, 4, i + 1) for i in range(3)]
    ax3d = fig.add_subplot(1, 4, 4, projection="3d")

    show = field if field is not None else labels.astype(float)
    vmin = vmax = None
    if field is not None:
        if log:
            show = np.log10(np.maximum(show, 1e-30))
        if scale_ids is not None:
            sm = np.isin(labels, np.asarray(list(scale_ids)))
        else:                       # tissue only: drop metal + insulation + air
            sm = ~np.isin(labels, np.asarray(list(electrode_ids) + [0, 110]))
        if sm.any():
            vmin, vmax = np.percentile(show[sm], clip_percentiles)

    for a, (axis, ix) in zip(axes, zip("xyz", centre)):
        k = "xyz".index(axis)
        take = [slice(None)] * 3
        take[k] = int(ix)
        plane = show[tuple(take)].T
        lplane = labels[tuple(take)].T
        if field is None:
            im = a.imshow(plane, origin="lower", cmap="tab20",
                          interpolation="nearest")
        else:
            if mask_to_focus and focus_ids is not None:
                fm = np.isin(lplane, np.asarray(list(focus_ids)))
                plane = np.where(fm, plane, np.nan)
            im = a.imshow(plane, origin="lower", cmap="magma",
                          interpolation="nearest", vmin=vmin, vmax=vmax)
            if focus_ids is not None and not mask_to_focus:
                a.contour(np.isin(lplane, np.asarray(list(focus_ids))).astype(float),
                          levels=[0.5], colors=[col["target"]], linewidths=1.0)
            a.contour(np.isin(lplane, np.asarray(list(electrode_ids))).astype(float),
                      levels=[0.5], colors=[col["source"]], linewidths=1.0)
        rem = [c for c in "xyz" if c != axis]
        a.set_xlabel(rem[0], fontsize=8); a.set_ylabel(rem[1], fontsize=8)
        a.set_title(f"{axis} = {ix}", fontsize=9, color=col["text"])
        a.tick_params(labelsize=7, colors=col["text2"])
        cb = fig.colorbar(im, ax=a, fraction=0.046, pad=0.02)
        cb.ax.tick_params(labelsize=7)

    # 3D: decimated point cloud of the focus structure + electrodes
    step = max(1, int(max(labels.shape) / 60))
    if focus_ids is not None:
        pts = np.argwhere(np.isin(labels[::step, ::step, ::step],
                                  np.asarray(list(focus_ids)))) * step
        if len(pts):
            c = None
            if field is not None:
                c = show[pts[:, 0], pts[:, 1], pts[:, 2]]
            ax3d.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=2, c=c,
                         cmap="magma" if c is not None else None,
                         color=None if c is not None else col["target"],
                         alpha=0.85, linewidths=0)
    ep = np.argwhere(np.isin(labels[::step, ::step, ::step],
                             np.asarray(list(electrode_ids)))) * step
    if len(ep):
        ax3d.scatter(ep[:, 0], ep[:, 1], ep[:, 2], s=4, c=col["source"],
                     alpha=0.9, linewidths=0)
    ax3d.set_xlabel("x", fontsize=7); ax3d.set_ylabel("y", fontsize=7)
    ax3d.set_zlabel("z", fontsize=7)
    ax3d.tick_params(labelsize=6)
    ax3d.set_title("3D", fontsize=9)

    fig.suptitle(title or f"{model.name}" +
                 (f" — {unit_label}" if field is not None else " — materials"),
                 fontsize=11, color=col["text"], x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(path, facecolor=col["surface"])
    plt.close(fig)
    return Path(path)
