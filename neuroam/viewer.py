"""Interactive 3D viewer for NeuroAM voxel models.

PyVista is imported lazily so solver-only and headless installations do not
need VTK.  The default scene contains a translucent non-void anatomy actor, a
movable categorical tissue slice, and one opaque actor per detected electrode
label.  Additional material labels can be highlighted as full 3D surfaces.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from .model import VoxelModel


_PALETTE = (
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
    "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC",
)

# Names established by the bundled RatCC configs and model notes.  The legacy
# .in format itself stores only numeric material IDs, so unknown IDs stay
# deliberately generic rather than being guessed from conductivity.
_KNOWN_LABEL_NAMES = {
    9: "muscle",
    12: "periorbital fat",
    13: "skin",
    24: "optic nerve",
    33: "sclera",
    66: "cornea",
    77: "retina",
    110: "electrode insulation",
}


def require_pyvista():
    """Import PyVista or raise an installation error with an exact command."""
    try:
        import pyvista as pv
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError(
            "The 3D viewer requires PyVista/VTK. Install it with:\n"
            "  python -m pip install -e \".[viewer]\""
        ) from exc
    return pv


def load_view_model(path, model_path=None) -> VoxelModel:
    """Load either a legacy ``.in`` model or a NeuroAM JSON config."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".in":
        return VoxelModel.from_legacy(path, model_path=model_path)
    if suffix == ".json":
        if model_path is not None:
            raise ValueError("--model can only be used with a legacy .in file")
        from .pipeline import build_model, load_config
        return build_model(load_config(path))
    raise ValueError("viewer input must be a legacy .in file or a config .json")


def _labels_touching_node(model: VoxelModel, node: Sequence[int]) -> set[int]:
    """Return voxel labels sharing a lattice node."""
    out: set[int] = set()
    nx, ny, nz = model.world
    x, y, z = (int(v) for v in node)
    for ix in (x - 1, x):
        for iy in (y - 1, y):
            for iz in (z - 1, z):
                if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
                    out.add(int(model.labels[ix, iy, iz]))
    return out


def electrode_labels(model: VoxelModel) -> Dict[int, str]:
    """Detect electrode labels and give them source/ground descriptions.

    Legacy files do not explicitly map terminals to material IDs.  Detection
    therefore combines terminal-adjacent labels with the conventional metal
    resistivity (all axes <= 1e-4 ohm m).
    """
    detected: Dict[int, str] = {}
    metal = {
        int(mat.id) for mat in model.materials
        if max(abs(float(v)) for v in mat.rho) <= 1e-4
    }
    present = {int(v) for v in np.unique(model.labels)}
    for label in sorted(metal & present):
        detected[label] = "electrode"

    for source in model.sources:
        for label in _labels_touching_node(model, source.node) & metal:
            detected[label] = f"source: {source.name}"
    for node in model.ground_nodes:
        for label in _labels_touching_node(model, node) & metal:
            detected[label] = "ground"

    # v0.2 terminal metadata is authoritative when it is available.
    for terminal in model.terminals:
        label = getattr(terminal, "label", None)
        if label is not None and int(label) in present:
            detected[int(label)] = f"{terminal.role}: {terminal.name}"
    return detected


def _material_name(model: VoxelModel, label: int) -> str:
    try:
        name = model.materials[int(label)].name
    except KeyError:
        name = ""
    return name or _KNOWN_LABEL_NAMES.get(int(label), f"material {label}")


def _color(label: int, electrode: bool = False) -> str:
    if electrode:
        return "#FF334F" if label % 2 else "#00CFE8"
    return _PALETTE[abs(int(label)) % len(_PALETTE)]


def material_label_report(model: VoxelModel) -> str:
    """Return the slice color key as compact terminal text."""
    electrodes = electrode_labels(model)
    labels = sorted(int(v) for v in np.unique(model.labels) if int(v) != 0)
    return "\n".join(
        f"  {label:>5}: {electrodes.get(label, _material_name(model, label))}"
        for label in labels
    )


def _label_lut(pv, present: Sequence[int], electrodes: Dict[int, str]):
    """Build an integer-indexed lookup table with transparent material 0."""
    max_label = max(present, default=0)
    if min(present, default=0) < 0 or max_label > 65535:
        raise ValueError("viewer material labels must be in the range 0..65535")
    rgba = np.zeros((max_label + 1, 4), dtype=np.uint8)
    for label in present:
        rgba[label, :3] = pv.Color(_color(label, label in electrodes)).int_rgb
        rgba[label, 3] = 0 if label == 0 else 255
    lut = pv.LookupTable(n_values=max_label + 1)
    lut.values = rgba
    lut.scalar_range = (0.0, float(max(max_label, 1)))
    return lut


class _VisibilityCallback:
    def __init__(self, actor):
        self.actor = actor

    def __call__(self, visible: bool) -> None:
        self.actor.SetVisibility(bool(visible))


def _add_visibility_control(plotter, actor, text: str, color: str,
                            row: int, visible: bool = True) -> None:
    """Add a compact checkbox and label down the upper-left of the window."""
    y = 820 - row * 34
    if y < 20:
        return
    plotter.add_checkbox_button_widget(
        _VisibilityCallback(actor), value=visible, position=(12, y), size=24,
        border_size=1, color_on=color, color_off="#555555",
        background_color="#222222",
    )
    plotter.add_text(text, position=(45, y + 2), font_size=9,
                     color="#F2F2F2", shadow=True)


def build_plotter(model: VoxelModel, labels: Optional[Iterable[int]] = None,
                  opacity: float = 0.10, background: str = "#20242B",
                  slice_axis: Optional[str] = "z",
                  slice_index: Optional[int] = None,
                  show_legend: bool = False,
                  off_screen: bool = False):
    """Build a PyVista scene without displaying it.

    Returned values are ``(plotter, grid)``.  Keeping this separate from
    :func:`show_model` makes screenshots and automated smoke tests possible.
    """
    if not 0.0 <= opacity <= 1.0:
        raise ValueError("opacity must be between 0 and 1")

    pv = require_pyvista()
    nx, ny, nz = model.world
    dx_mm = float(model.dx) * 1e3
    grid = pv.ImageData(
        dimensions=(nx + 1, ny + 1, nz + 1),
        spacing=(dx_mm, dx_mm, dx_mm),
        origin=(0.0, 0.0, 0.0),
    )
    # VTK cell IDs advance x first; Fortran order maps labels[x,y,z] to that.
    grid.cell_data["material"] = np.asarray(model.labels, dtype=np.int32).ravel(
        order="F")

    plotter = pv.Plotter(off_screen=off_screen, window_size=(1400, 900))
    plotter.set_background(background)
    display_name = model.name if len(model.name) <= 38 else model.name[:35] + "..."
    plotter.add_title(f"NeuroAM — {display_name}", font_size=13,
                      color="#F2F2F2")

    present = {int(v) for v in np.unique(model.labels)}
    nonvoid = sorted(v for v in present if v != 0)
    electrodes = electrode_labels(model)
    row = 0
    if nonvoid:
        # Material 0 is the legacy outside-world/void convention.
        context = grid.threshold(0.5, scalars="material", preference="cell",
                                 method="upper")
        context_actor = plotter.add_mesh(
            context, name="anatomy", color="#B8C2CC", opacity=opacity,
            smooth_shading=False, show_edges=False, pickable=False,
        )
        _add_visibility_control(plotter, context_actor, "anatomy context",
                                "#B8C2CC", row)
        row += 1

    # A colored section is the memory-safe way to inspect all internal tissue
    # boundaries, including on the 912x900x504 full-head model. Start it at a
    # source node when possible so the first view intersects the target.
    if slice_axis is not None:
        if slice_axis not in "xyz":
            raise ValueError("slice_axis must be x, y, z, or None")
        axis_i = "xyz".index(slice_axis)
        if slice_index is None:
            slice_index = (model.sources[0].node[axis_i] if model.sources
                           else model.world[axis_i] // 2)
        if not 0 <= int(slice_index) <= model.world[axis_i]:
            raise ValueError(
                f"slice index {slice_index} outside {slice_axis}=0.."
                f"{model.world[axis_i]}")
        origin = np.asarray(model.world, dtype=float) * dx_mm / 2.0
        origin[axis_i] = int(slice_index) * dx_mm
        lut = _label_lut(pv, sorted(present), electrodes)
        slice_actor = plotter.add_mesh_slice(
            grid, normal=slice_axis, origin=origin,
            scalars="material", preference="cell", cmap=lut,
            show_scalar_bar=False,
            widget_color="#F2F2F2", interaction_event="end",
        )
        _add_visibility_control(
            plotter, slice_actor,
            f"tissue slice ({slice_axis}={int(slice_index)}; drag plane)",
            "#F2F2F2", row)
        row += 1
        if show_legend:
            legend = [
                (f"{label}: {electrodes.get(label, _material_name(model, label))}",
                 _color(label, label in electrodes))
                for label in nonvoid
            ]
            plotter.add_legend(
                legend, bcolor=background, border=False, loc="lower right",
                size=(0.15, min(0.68, 0.021 * len(legend))),
                background_opacity=0.82,
            )

    requested = [] if labels is None else [int(v) for v in labels]
    missing = sorted(set(requested) - present)
    if missing:
        raise ValueError(f"material label(s) not present in model: {missing}")

    show_labels = list(electrodes)
    show_labels.extend(v for v in requested if v not in electrodes)
    counts = dict(zip(*np.unique(model.labels, return_counts=True)))
    for label in show_labels:
        part = grid.threshold((label - 0.5, label + 0.5),
                              scalars="material", preference="cell")
        is_electrode = label in electrodes
        color = _color(label, electrode=is_electrode)
        actor = plotter.add_mesh(
            part, name=f"label-{label}", color=color,
            opacity=1.0 if is_electrode else 0.72,
            smooth_shading=False, show_edges=False,
        )
        desc = electrodes.get(label, _material_name(model, label))
        text = f"{label}: {desc} ({int(counts[label]):,} vox)"
        _add_visibility_control(plotter, actor, text, color, row)
        row += 1

    plotter.add_axes(line_width=2, labels_off=False)
    plotter.show_bounds(
        mesh=grid, location="outer", all_edges=True, color="#D7DEE7",
        xtitle="X (mm)", ytitle="Y (mm)", ztitle="Z (mm)",
    )
    plotter.enable_trackball_style()
    plotter.view_isometric()
    plotter.reset_camera()
    return plotter, grid


def show_model(model: VoxelModel, labels: Optional[Iterable[int]] = None,
               opacity: float = 0.10, background: str = "#20242B",
               slice_axis: Optional[str] = "z",
               slice_index: Optional[int] = None,
               show_legend: bool = False,
               screenshot=None, off_screen: bool = False) -> None:
    """Open the interactive 3D window, or render an off-screen screenshot."""
    if off_screen and screenshot is None:
        raise ValueError("--off-screen requires --screenshot")
    plotter, _ = build_plotter(model, labels=labels, opacity=opacity,
                               background=background, slice_axis=slice_axis,
                               slice_index=slice_index,
                               show_legend=show_legend,
                               off_screen=off_screen)
    shot = None
    if screenshot is not None:
        shot = Path(screenshot)
        shot.parent.mkdir(parents=True, exist_ok=True)
    print("3D controls: drag=rotate, shift+drag=pan, wheel=zoom, r=reset; "
          "drag the white plane to inspect internal tissues")
    if slice_axis is not None:
        print("Tissue-slice color key (use --legend to also show it in-window):")
        print(material_label_report(model))
    plotter.show(title=f"NeuroAM — {model.name}", screenshot=shot)
