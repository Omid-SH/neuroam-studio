"""One figure: where every electrode actually sits, for the original 6 RatCC
montages and the 9 literature-derived clinical configs (examples/38) --
a visual, not just a numeric, check on the placements behind
examples/39's comparison.

Each panel is a top-down (X,Y) silhouette of the full head (any non-air
voxel, projected with a boolean OR along Z -- the model's thinnest axis, so
this reads as a recognisable head/eye outline) with that config's own
source electrode(s) (red star) and ground/return (blue circle) marked at
their true centroid position. One consistent projection and axis frame
across all 15 panels, so relative electrode distances are directly
comparable panel to panel -- e.g. TES-GPS's forehead ground sits close to
its own source, while SCL-Wrist's wrist-proxy return sits about as far from
the eye as this crop reaches.

Run:  python examples/40_electrode_placement_map.py
Requires: examples/38's registration reports (clinical configs) and the
         original 6 montages' own .in files (already in samples/).
Writes: samples/ratcc_eye_83um/verification/views/electrode_placement_map.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from neuroam.model import VoxelModel
from _sweep_common import CONFIGS as ORIG_CONFIGS

REPO = Path(__file__).resolve().parents[1]
SAMP = REPO / "samples/ratcc_eye_83um"
BASE_STEM = "RatCC_full_CLStim_JGND_912_900_504_Ring_Res_83um_with_retina"
CLIN_DIR = SAMP / "verification" / "clinical"
VIEWS = SAMP / "verification" / "views"

CLINICAL_KEYS = ["TESGPS", "VIRON_periorbital_open", "VIRON_periorbital_closed",
                "VIRON6_open", "VIRON6_closed", "TpES_open", "TpES_closed",
                "TcES", "SCL_Wrist"]

INK, INK2, SURF, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e1e0d9"
SOURCE_COLOR, GROUND_COLOR, OUTLINE_COLOR = "#e31a1c", "#1f78b4", "#c9c6bd"


def head_outline_xy(labels: np.ndarray) -> np.ndarray:
    """Boolean (X,Y) silhouette: any tissue present at any Z."""
    return np.any(labels > 0, axis=2)


def original_electrode_centroids(cfg: str, fname: str):
    model = VoxelModel.from_legacy(SAMP / f"{fname}.in")
    src = np.argwhere(model.labels == 101).mean(0)
    gnd = np.argwhere(model.labels == 102).mean(0)
    return [("source", src)], [("ground", gnd)]


def clinical_electrode_centroids(config_key: str):
    report = json.loads((CLIN_DIR / f"{config_key}_registration_report.json").read_text())
    sources, grounds = [], []
    for e in report["electrodes"]:
        if e["n_voxels"] == 0:
            continue
        c = np.array(e["centroid_vox"])
        (sources if e["role"] == "source" else grounds).append((e["name"], c))
    return sources, grounds


def draw_panel(ax, outline, sources, grounds, title):
    ax.imshow(outline.T, origin="lower", cmap="Greys", alpha=0.35,
             interpolation="nearest", vmin=0, vmax=1.8)
    for name, c in grounds:
        ax.scatter([c[0]], [c[1]], s=70, marker="o", facecolor=GROUND_COLOR,
                  edgecolor="white", linewidths=0.6, zorder=3)
    for name, c in sources:
        ax.scatter([c[0]], [c[1]], s=110, marker="*", facecolor=SOURCE_COLOR,
                  edgecolor="white", linewidths=0.6, zorder=4)
    ax.set_title(title, fontsize=9, color=INK)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(GRID)


def main():
    print("loading shared base anatomy for the head outline ...", flush=True)
    base_model = VoxelModel.from_legacy(SAMP / f"{BASE_STEM}.in")
    outline = head_outline_xy(base_model.labels)
    print(f"outline: {outline.sum():,} of {outline.size:,} XY cells occupied", flush=True)

    panels = []
    for cfg, fname in ORIG_CONFIGS.items():
        s, g = original_electrode_centroids(cfg, fname)
        panels.append((cfg, s, g))
        print(f"[orig] {cfg}: source {s[0][1].round(1).tolist()} "
             f"ground {g[0][1].round(1).tolist()}", flush=True)
    for key in CLINICAL_KEYS:
        s, g = clinical_electrode_centroids(key)
        panels.append((key, s, g))
        print(f"[clinical] {key}: {len(s)} source(s), {len(g)} ground(s)", flush=True)

    n = len(panels)
    ncols = 5
    nrows = -(-n // ncols)
    plot_h = 3.3 * nrows
    header_h = 1.6
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.1 * ncols, plot_h + header_h), dpi=140)
    fig.patch.set_facecolor(SURF)
    axes_flat = np.asarray(axes).reshape(-1)
    for ax, (title, s, g) in zip(axes_flat, panels):
        draw_panel(ax, outline, s, g, title)
    for ax in axes_flat[n:]:
        ax.axis("off")

    # legend, once, in the first unused cell (or as a figure-level legend)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="*", color="none", markerfacecolor=SOURCE_COLOR,
                      markeredgecolor="white", markersize=13, label="source"),
              Line2D([0], [0], marker="o", color="none", markerfacecolor=GROUND_COLOR,
                    markeredgecolor="white", markersize=9, label="ground / return")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=10,
              frameon=False, labelcolor=INK, bbox_to_anchor=(0.5, 0.0))

    total_h = plot_h + header_h
    fig.suptitle("Electrode placement: original 6 RatCC montages + 9 literature-derived "
                "clinical configs", fontsize=13, color=INK, y=1 - 0.35 / total_h)
    fig.text(0.5, 1 - 0.85 / total_h,
             "Top-down (X,Y) silhouette of the full head (any tissue, projected across Z) -- "
             "same frame in every panel, so relative source/ground distance is comparable.\n"
             "Grounds far off this projection sit near the edge/outside the crop "
             "(documented approximations -- see docs/CLINICAL_ELECTRODES.md).",
             fontsize=8.5, color=INK2, ha="center", va="top")
    fig.tight_layout(rect=(0, 0.03, 1, plot_h / total_h))
    out = VIEWS / "electrode_placement_map.png"
    fig.savefig(out, facecolor=SURF)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
