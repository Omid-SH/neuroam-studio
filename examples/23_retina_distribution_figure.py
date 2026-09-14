"""Retina current-density distribution figure (Retina-HFBlock Figure 2c style).

Reads ``verification/retina_distribution.npz`` (written by
``22_retina_distribution_study.py``) and renders box + strip distributions
of current density across the five CL electrode variants, for three
RGC-relevant regions -- central retina, peripheral retina, optic nerve -- plus
a second row giving the tangential E-field (an axon-activation-relevant
proxy) for the two retina regions.

Run (after 22_retina_distribution_study.py):
    python examples/23_retina_distribution_figure.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
V = HERE.parent / "samples" / "ratcc_eye_83um" / "verification"

# ---- validated categorical slots (neuroam/viz3d.py PALETTE, fixed order) ----
INK, INK2, MUTED, SURF, GRID = "#0b0b0b", "#52514e", "#898781", "#fcfcfb", "#e1e0d9"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]   # 1,2,3,4,5

CASES = ["A_shipped_node", "B_shipped_supernode", "C_ring_th53",
         "D_ring_th40", "E_cap_30"]
LABELS = ["A: shipped\n(node)", "B: shipped\n(supernode)", "C: ring\n53.3°",
          "D: ring\n40.0°", "E: cap\n30°"]
CUR1_AMP_A = 0.0002

# central/peripheral cut, degrees of eccentricity from the posterior pole.
# The OEC-RGC MATLAB reference uses 2.5 deg for its foveal cap; this crop's
# retina is small (0-120 deg total, 27455 voxels) so 5 deg is used here to
# keep a workable voxel count in "central" -- change freely, no re-solve.
SPLIT_DEG = 5.0


def _box_strip(ax, arrays, title, ylabel, log=True):
    rng = np.random.default_rng(0)
    positions = np.arange(1, len(arrays) + 1)
    for pos, arr, color in zip(positions, arrays, SLOTS):
        x = pos + rng.uniform(-0.14, 0.14, size=len(arr))
        ax.scatter(x, arr, s=3, color=color, alpha=0.35, linewidths=0,
                   zorder=2, rasterized=True)
    bp = ax.boxplot(arrays, positions=positions, widths=0.5, showfliers=False,
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
    ax.set_xticklabels(LABELS, fontsize=7, color=INK2)
    ax.tick_params(axis="y", labelsize=7.5, colors=INK2)
    ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)


def main():
    z = np.load(V / "retina_distribution.npz")
    ecc_split = SPLIT_DEG

    def split(field_key):
        """Per-config arrays for `field_key`, cut at SPLIT_DEG eccentricity."""
        central, peripheral = [], []
        for c in CASES:
            ecc = z[f"{c}_ecc_deg"]
            vals = z[f"{c}_{field_key}"].astype(float) * CUR1_AMP_A
            m = ecc <= ecc_split
            central.append(vals[m])
            peripheral.append(vals[~m])
        return central, peripheral

    def scaled(key):
        return [z[f"{c}_{key}"].astype(float) * CUR1_AMP_A for c in CASES]

    J_central, J_peripheral = split("J_retina")
    Etan_central, Etan_peripheral = split("Etan_retina")
    n_central = len(J_central[0])
    n_peripheral = len(J_peripheral[0])

    fig = plt.figure(figsize=(13.5, 8.4), dpi=140)
    fig.patch.set_facecolor(SURF)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.5, wspace=0.32,
                          left=0.06, right=0.98, top=0.88, bottom=0.10)

    ax1 = fig.add_subplot(gs[0, 0])
    _box_strip(ax1, J_central, f"Central retina (ecc ≤ {ecc_split:g}°, n={n_central})",
              "|J| (µA/mm²)")
    ax2 = fig.add_subplot(gs[0, 1])
    _box_strip(ax2, J_peripheral, f"Peripheral retina (ecc > {ecc_split:g}°, n={n_peripheral})",
              "|J| (µA/mm²)")
    ax3 = fig.add_subplot(gs[0, 2])
    _box_strip(ax3, scaled("J_optic_nerve"), "Optic nerve",
              "|J| (µA/mm²)")

    ax4 = fig.add_subplot(gs[1, 0])
    _box_strip(ax4, Etan_central, "Central retina",
              "tangential E (V/m, per 200 µA eq.)")
    ax5 = fig.add_subplot(gs[1, 1])
    _box_strip(ax5, Etan_peripheral, "Peripheral retina",
              "tangential E (V/m, per 200 µA eq.)")

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    ax6.text(0.0, 1.0,
            "Tangential E-field\n(RGC-axon proxy)",
            fontsize=10.5, color=INK, fontweight="bold", va="top", ha="left",
            transform=ax6.transAxes)
    ax6.text(0.0, 0.82,
            "RGC axons run tangentially in the nerve-fibre layer, so the\n"
            "component of E tangent to the globe is a more relevant proxy\n"
            "for axonal activation than |E| alone. A literal activating-\n"
            "function (2nd spatial derivative along the axon path) needs\n"
            "actual axon morphology, which is not modelled here.",
            fontsize=8, color=INK2, va="top", ha="left", transform=ax6.transAxes,
            linespacing=1.5)
    ax6.text(0.0, 0.28,
            f"Central/peripheral split: this rat retina is afoveate, so\n"
            f"there is no anatomical foveal boundary in the model. \"Central\"\n"
            f"here = within {ecc_split:g}° of the posterior pole (the OEC-RGC\n"
            f"MATLAB reference uses 2.5° on the full model); \"peripheral\"\n"
            f"= everything beyond that, out to the retina's own edge (120°).",
            fontsize=8, color=INK2, va="top", ha="left", transform=ax6.transAxes,
            linespacing=1.5)

    fig.suptitle("Retina & optic-nerve current density across CL electrode variants",
                fontsize=13.5, color=INK, x=0.06, ha="left", y=0.975)
    fig.text(0.06, 0.925,
             "83 µm RatCC eye crop (SCL-ON montage only) · unit-current solve "
             "scaled to the bundled Cur1 waveform (200 µA) · box = median/IQR, "
             f"dots = per-voxel values (central n={n_central}, "
             f"peripheral n={n_peripheral}, optic nerve n=2176)",
             fontsize=9, color=INK2)

    out = V / "views" / "retina_current_density_distribution.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURF)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
