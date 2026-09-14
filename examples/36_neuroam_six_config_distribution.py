"""The six-configuration current-density comparison, from NeuroAM's own
solves this time -- not the OEC-RGC legacy `_J.raw` archive that
``examples/24_six_configuration_figure.py`` reads directly.

Same regions and visual language as examples/24 (central/peripheral retina
split at 5 deg, occipital cortex), plus a fourth panel examples/24 computed
the mask for but never actually plotted: the optic nerve itself (material
24, the two hand-traced tube sections from ``calculate_retina_opticnerve_EF.m``).
Reuses the already-solved, already-cached ``.neuroam_cache`` basis fields
(no re-solving -- same fields ``examples/28`` produced and the stimulation
sweeps in examples/31-35 already reused), scaled to the 200 uA amplitude
this whole project's OEC-RGC-derived convention uses -- the same scaling
``examples/33_verify_against_oecrgc.py`` confirmed matches the legacy
pipeline's own archived fields to ~1%.

Run:  python examples/36_neuroam_six_config_distribution.py
Writes: samples/ratcc_eye_83um/verification/views/
       six_configuration_distribution_neuroam.png
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from neuroam.frames import Frame
from _sweep_common import CachedMontage, CONFIGS

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
V = REPO / "samples/ratcc_eye_83um/verification"

RETINA, OPTIC_NERVE, BRAIN = 77, 24, 18
SPLIT_DEG = 5.0
CUR1_AMP_A = 200e-6          # see examples/33's cross-validation

INK, INK2, MUTED, SURF, GRID = "#0b0b0b", "#52514e", "#898781", "#fcfcfb", "#e1e0d9"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]  # slots 1-6


def _box_strip(ax, arrays, title, ylabel, log=True):
    rng = np.random.default_rng(0)
    positions = np.arange(1, len(arrays) + 1)
    for pos, arr, color in zip(positions, arrays, SLOTS):
        x = pos + rng.uniform(-0.14, 0.14, size=len(arr))
        ax.scatter(x, arr, s=3, color=color, alpha=0.3, linewidths=0,
                  zorder=2, rasterized=True)
    bp = ax.boxplot(arrays, positions=positions, widths=0.6, showfliers=False,
                    patch_artist=True, zorder=3,
                    medianprops=dict(color=INK, linewidth=1.4),
                    whiskerprops=dict(color=INK2, linewidth=1.0),
                    capprops=dict(color=INK2, linewidth=1.0),
                    boxprops=dict(linewidth=1.0, edgecolor=INK2))
    for patch, color in zip(bp["boxes"], SLOTS):
        patch.set_facecolor(color)
        patch.set_alpha(0.28)
    if log:
        ax.set_yscale("log")
    ax.set_title(title, fontsize=10.5, color=INK)
    ax.set_ylabel(ylabel, fontsize=8.5, color=INK2)
    ax.set_xticks(positions)
    ax.set_xticklabels(list(CONFIGS), fontsize=6.3, color=INK2, rotation=20, ha="right")
    ax.tick_params(axis="y", labelsize=7.5, colors=INK2)
    ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)


def main():
    # masks/geometry are identical across all 6 configs (same anatomy,
    # only the electrode pair differs) -- compute once from any one of them
    print("loading SCL-ON for shared masks/geometry ...", flush=True)
    ref = CachedMontage("SCL-ON")
    labels = ref.model.labels

    retina_mask = labels == RETINA
    on_mask = labels == OPTIC_NERVE
    brain_mask = labels == BRAIN
    Yg, Zg = np.indices((labels.shape[1], labels.shape[2]))
    occ_plane = (Yg > 550) & (Zg > 300)          # same voxel rule as the MATLAB script
    occipital_mask = brain_mask & occ_plane[None, :, :]

    idx = np.argwhere(retina_mask)
    fr = Frame.eye(ref.model, (RETINA,))
    pts_mm = fr.grid.centers_mm(idx)
    _, theta, _ = fr.spherical_of(pts_mm)
    ecc = 180.0 - theta
    central = ecc <= SPLIT_DEG
    peripheral = ~central
    print(f"retina: {len(idx):,} voxels, central(n={central.sum()}) / "
         f"peripheral(n={peripheral.sum()}); optic nerve {on_mask.sum():,}; "
         f"occipital {occipital_mask.sum():,}", flush=True)

    store = {}
    J_central, J_peripheral, J_on, J_occ = [], [], [], []
    for cfg in CONFIGS:
        m = ref if cfg == "SCL-ON" else CachedMontage(cfg)
        J = m.jmag * CUR1_AMP_A          # A/m^2 at 200 uA, per examples/33's scaling
        Jr = J[retina_mask]
        c = Jr[central].astype(np.float32)
        p = Jr[peripheral].astype(np.float32)
        n = J[on_mask].astype(np.float32)
        o = J[occipital_mask].astype(np.float32)
        J_central.append(c); J_peripheral.append(p); J_on.append(n); J_occ.append(o)
        store[f"{cfg}_J_central"] = c
        store[f"{cfg}_J_peripheral"] = p
        store[f"{cfg}_J_optic_nerve"] = n
        store[f"{cfg}_J_occipital"] = o
        print(f"[{cfg}] central median={np.median(c):.4g} peripheral median={np.median(p):.4g} "
             f"optic-nerve median={np.median(n):.4g} occipital median={np.median(o):.4g}",
             flush=True)
    store["ecc_split_deg"] = np.array([SPLIT_DEG])
    store["cur1_amp_A"] = np.array([CUR1_AMP_A])
    np.savez_compressed(V / "six_configuration_distribution_neuroam.npz", **store)

    fig = plt.figure(figsize=(18.5, 5.6), dpi=140)
    fig.patch.set_facecolor(SURF)
    gs = fig.add_gridspec(1, 4, wspace=0.34, left=0.045, right=0.985,
                          top=0.80, bottom=0.24)

    ax1 = fig.add_subplot(gs[0, 0])
    _box_strip(ax1, J_central, f"Central retina (ecc <= {SPLIT_DEG:g}, n={central.sum()})",
              "|J| (A/m^2) at 200 uA")
    ax2 = fig.add_subplot(gs[0, 1])
    _box_strip(ax2, J_peripheral, f"Peripheral retina (ecc > {SPLIT_DEG:g}, n={peripheral.sum()})",
              "|J| (A/m^2) at 200 uA")
    ax3 = fig.add_subplot(gs[0, 2])
    _box_strip(ax3, J_on, f"Optic nerve (n={on_mask.sum()})",
              "|J| (A/m^2) at 200 uA")
    ax4 = fig.add_subplot(gs[0, 3])
    _box_strip(ax4, J_occ, f"Occipital cortex (n={occipital_mask.sum()})",
              "|J| (A/m^2) at 200 uA")

    fig.suptitle("Current density by region across the 6 electrode-pair montages -- "
                "NeuroAM's own solve", fontsize=13.5, color=INK, x=0.045, ha="left", y=0.97)
    fig.text(0.045, 0.885,
             "Full 912x900x504 RatCC model, 83 um, solved by NeuroAM (unit-current basis "
             "field, scaled to 200 uA) . box = median/IQR, dots = per-voxel values . "
             "central/peripheral split at 5 deg eccentricity . cross-validated against "
             "the legacy OEC-RGC pipeline to ~1% (docs/VALIDATION.md)",
             fontsize=8.5, color=INK2)

    out = V / "views" / "six_configuration_distribution_neuroam.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURF)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
