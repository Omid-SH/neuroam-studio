"""A real 3D render (not a flat 2D silhouette) of electrode placement for
every configuration in examples/39's comparison -- the surfaces of the
model tissue and the electrodes themselves, coloured by role (source vs.
ground/return), shot from one consistent camera: from in front of the
animal's eye, looking in, tilted slightly upward.

Replaces examples/40's top-down dot map with an actual isosurface render
per config (neuroam.viz3d.Scene). Ships as **interactive Plotly HTML**, not
a static PNG -- a first pass rasterised each panel through kaleido and
collaged them into one PNG, but kaleido's still-image export visibly loses
the smooth, connected marching-cubes contours the project's other 3D views
(e.g. examples/20's ``01_model_and_electrodes.html``) show live in a
browser. Reusing the real thing sidesteps that entirely.

Run:  python examples/42_electrode_3d_placement.py
Writes: samples/ratcc_eye_83um/verification/views/electrode_3d/<config>.html
       (one interactive scene per config)
       samples/ratcc_eye_83um/verification/views/electrode_placement_3d_grid.html
       (index page, all 15 embedded as a grid of iframes)
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from neuroam import viz3d
from _sweep_common import CachedMontage, CONFIGS as ORIG_CONFIGS

REPO = Path(__file__).resolve().parents[1]
SAMP = REPO / "samples/ratcc_eye_83um"
OUT3D = SAMP / "verification" / "views" / "electrode_3d"
VIEWS = SAMP / "verification" / "views"
OUT3D.mkdir(parents=True, exist_ok=True)

SKIN, CORNEA, SCLERA, RETINA = 13, 66, 33, 77
DX = 83e-6
PANEL_PX = 420

# import examples/38 despite its filename starting with a digit
_spec = importlib.util.spec_from_file_location(
    "clinical38", Path(__file__).resolve().parent / "38_solve_clinical_configs.py")
clinical38 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(clinical38)   # __name__ != "__main__", its own CLI block never runs

# (name, source_labels, ground_labels)
CLINICAL_ELECTRODES = {
    "TESGPS": ([101], [102]),
    "VIRON_periorbital_open": ([121], [122]),
    "VIRON_periorbital_closed": ([121], [122]),
    "VIRON6_open": ([121, 122, 123, 124, 125, 126], [127]),
    "VIRON6_closed": ([121, 122, 123, 124, 125, 126], [127]),
    "TpES_open": ([131], [132]),
    "TpES_closed": ([131], [132]),
    "TcES": ([141], [132]),
    "SCL_Wrist": ([101], [150]),
}

# corneal axis, world mm (docs/CLINICAL_ELECTRODES.md sec4) -- shared by every
# config (same base anatomy). "Front of the animal" = camera placed outside
# the eye along +e3, looking back toward the head; a small +Z adds the
# "little bit up" tilt (Z is the model's dorsal-ventral axis -- confirmed by
# examples/40's Z-projection reading as a recognisable top-down head outline).
E3 = np.array([-0.7071, -0.7071, 0.0])
UP_TILT = np.array([0.0, 0.0, 0.35])
CAMERA_DIR = E3 + UP_TILT
CAMERA_DIR = CAMERA_DIR / np.linalg.norm(CAMERA_DIR)


def build_figure(labels: np.ndarray, source_labels, ground_labels, step: int = 1):
    # crops are tight (electrode bbox + 25 vox), so step=1 (no decimation)
    # is affordable. Context is the union of *every* non-focus, non-electrode
    # material (examples/20's view_model() convention) -- almost solid tissue,
    # so it reads as a coherent body silhouette instead of the gappy, ridged
    # surface a single thin SKIN shell alone produces.
    all_ids = list(source_labels) + list(ground_labels)
    idx = np.argwhere(np.isin(labels, all_ids))
    lo = np.clip(idx.min(0) - 25, 0, None)
    hi = np.clip(idx.max(0) + 25, None, np.array(labels.shape))
    crop = tuple((int(lo[k]), int(hi[k])) for k in range(3))
    sl = tuple(slice(int(lo[k]), int(hi[k])) for k in range(3))

    focus_ids = [CORNEA, SCLERA]
    present = set(np.unique(labels[sl]).tolist())
    context_ids = sorted(present - set(focus_ids) - set(all_ids) - {0})

    sc = viz3d.Scene(dx=DX, crop=crop, units="mm", theme="light", title="", subtitle="")
    if context_ids:
        sc.add_materials(labels, context_ids, "head tissue", role="context",
                         opacity=0.10, step=step, smooth=2)
    sc.add_materials(labels, [CORNEA], "cornea", role="context", opacity=0.35,
                     step=step, smooth=2)
    sc.add_materials(labels, [SCLERA], "sclera", role="context", opacity=0.25,
                     step=step, smooth=2)
    sc.add_materials(labels, source_labels, "source", role="source", opacity=1.0,
                     step=step, smooth=2)
    sc.add_materials(labels, ground_labels, "ground / return", role="ground",
                     opacity=1.0, step=step, smooth=2)

    fig = sc.figure()
    # camera "eye" is in normalised scene units around the data's own bounding
    # box (plotly's convention with aspectmode="data"); a few units out along
    # CAMERA_DIR reliably clears the crop regardless of its absolute size.
    d = CAMERA_DIR * 1.7
    no_axis = dict(visible=False, showbackground=False)
    fig.update_layout(
        scene=dict(xaxis=no_axis, yaxis=no_axis, zaxis=no_axis),
        scene_camera=dict(eye=dict(x=float(d[0]), y=float(d[1]), z=float(d[2])),
                          up=dict(x=0, y=0, z=1)),
        showlegend=True,
        legend=dict(x=0.0, y=0.0, xanchor="left", yanchor="bottom",
                   bgcolor="rgba(255,255,255,0.75)", font=dict(size=9)),
        width=PANEL_PX, height=PANEL_PX, margin=dict(l=0, r=0, t=4, b=0))
    return fig


def render_one(name: str, labels: np.ndarray, source_labels, ground_labels) -> Path:
    fig = build_figure(labels, source_labels, ground_labels)
    out = OUT3D / f"{name}.html"
    fig.write_html(str(out), include_plotlyjs="cdn", full_html=True,
                   config={"displaylogo": False})
    return out


GRID_CSS = """
body { margin:0; background:#fcfcfb; color:#0b0b0b;
      font-family: system-ui, -apple-system, Segoe UI, sans-serif; }
header { padding: 18px 22px 4px; }
header h1 { font-size: 19px; margin: 0 0 6px; }
header p { font-size: 12.5px; color: #52514e; margin: 0 0 14px; }
.grid { display:grid; grid-template-columns: repeat(5, 1fr); gap: 14px;
       padding: 0 18px 24px; }
.cell { text-align:center; }
.cell .name { font-size: 12.5px; font-weight: 600; margin-bottom: 4px; }
.cell iframe { width:100%; aspect-ratio: 1 / 1; border: 1px solid #e1e0d9;
              border-radius: 6px; background:#fcfcfb; }
@media (max-width: 1100px) { .grid { grid-template-columns: repeat(3, 1fr); } }
@media (max-width: 640px)  { .grid { grid-template-columns: repeat(2, 1fr); } }
"""


def write_grid_html(panels: list[tuple[str, Path]]) -> Path:
    cells = "\n".join(
        f'    <div class="cell"><div class="name">{name}</div>'
        f'<iframe src="electrode_3d/{path.name}" loading="lazy" '
        f'scrolling="no"></iframe></div>'
        for name, path in panels)
    html = f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>Electrode placement, 3D</title>
<style>{GRID_CSS}</style></head>
<body>
<header>
  <h1>Electrode placement, 3D: original 6 RatCC montages + 9 literature-derived clinical configs</h1>
  <p>Red = source, blue = ground/return. Same camera (front of the eye, looking in, tilted
     slightly up) in every panel. Each panel is live -- drag to rotate, scroll to zoom,
     click legend entries to toggle layers.</p>
</header>
<div class="grid">
{cells}
</div>
</body></html>"""
    out = VIEWS / "electrode_placement_3d_grid.html"
    out.write_text(html, encoding="utf-8")
    return out


def main():
    t0 = time.time()
    panels = []

    for cfg in ORIG_CONFIGS:
        m = CachedMontage(cfg)
        p = render_one(cfg, m.model.labels, [101], [102])
        panels.append((cfg, p))
        print(f"[orig] {cfg} -> {p.name} ({time.time()-t0:.0f}s)", flush=True)
        del m

    # base full-head model loaded ONCE and reused across the 8 shared-anatomy
    # clinical configs (examples/38's own --report batch does the same) --
    # each call still gets a pristine labels array via clean_labels reset.
    shared_keys = [k for k in CLINICAL_ELECTRODES if k != "SCL_Wrist"]
    model0 = clinical38.load_base()
    frame0 = clinical38.Frame.eye(model0, (clinical38.RETINA,))
    clean0 = model0.labels.copy()
    print(f"[clinical base] loaded once, reused for {len(shared_keys)} configs "
         f"({time.time()-t0:.0f}s)", flush=True)
    for key in shared_keys:
        src, gnd = CLINICAL_ELECTRODES[key]
        model, frame, specs, qcs = clinical38.main(
            key, do_solve=False, model=model0, frame=frame0, clean_labels=clean0,
            return_model=True)
        p = render_one(key, model.labels, src, gnd)
        panels.append((key, p))
        print(f"[clinical] {key} -> {p.name} ({time.time()-t0:.0f}s)", flush=True)

    # SCL_Wrist keeps SCL-ON's own contact-lens ring (keep_scl_ring=True), so
    # it can't share clean0 (ring already stripped there) -- its own base load.
    src, gnd = CLINICAL_ELECTRODES["SCL_Wrist"]
    model, frame, specs, qcs = clinical38.main("SCL_Wrist", do_solve=False, return_model=True)
    p = render_one("SCL_Wrist", model.labels, src, gnd)
    panels.append(("SCL_Wrist", p))
    print(f"[clinical] SCL_Wrist -> {p.name} ({time.time()-t0:.0f}s)", flush=True)

    out_grid = write_grid_html(panels)
    print(f"wrote {out_grid}")


def test_one(cfg: str):
    """Render a single config and stop -- for validating the camera angle
    before committing to the full (slow) 15-panel batch."""
    if cfg in ORIG_CONFIGS:
        m = CachedMontage(cfg)
        p = render_one(cfg, m.model.labels, [101], [102])
    else:
        src, gnd = CLINICAL_ELECTRODES[cfg]
        model, frame, specs, qcs = clinical38.main(cfg, do_solve=False, return_model=True)
        p = render_one(cfg, model.labels, src, gnd)
    print(f"wrote {p}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_one(sys.argv[2])
    else:
        main()
