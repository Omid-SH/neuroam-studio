"""Reproduce the OEC-RGC Figure 2c six-configuration comparison, for real.

Unlike ``22_retina_distribution_study.py`` / ``23_retina_distribution_figure.py``
(which vary only the SCL/contact-lens electrode's own geometry, holding the
SCL-ON pair fixed), this script reproduces your *original* six montages --
each a different pair of anatomical electrode sites on the same full
912x900x504 rat head model:

    SCL-ON                     RatCC_full_CLStim_JGND
    SCL-IntraCranial           RatCC_full_CLStim_NeedleGND
    SCL-TransCranial           RatCC_full_CLStim_PatchGND
    ON-IntraCranial            RatCC_full_JStim_NeedleGND
    ON-TransCranial            RatCC_full_JStim_PatchGND
    IntraCranial-TransCranial  RatCC_full_NeedleStim_PlateGND

No new solving is needed: the six ``_J.raw`` current-density volumes were
already computed by the lab's legacy pipeline and sit in
``D:\\OEC-RGC\\Results\\with retina 83um``. This script only reads them,
masks them by retina eccentricity (central/peripheral, same 5 deg split as
23_retina_distribution_figure.py) and by an occipital-cortex ROI (brain
tissue, label 18, restricted to y>550 & z>300 -- the same voxel condition
``calculate_retina_opticnerve_EF.m`` uses), and boxes them up.

Units: the *_J.raw files are read as-is -- whatever amplitude the original
lab solve used, not necessarily NeuroAM's 1 A unit-current convention. If you
know that amplitude, rescale before comparing to the CL-geometry figure.

Run:  python examples/24_six_configuration_figure.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from neuroam.frames import Frame
from neuroam.model import VoxelModel

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
FULL_IN = REPO / "samples/ratcc_eye_83um/RatCC_full_CLStim_JGND_912_900_504_Ring_Res_83um_with_retina.in"
RESULTS = Path(r"D:\OEC-RGC\Results\with retina 83um")
V = REPO / "samples/ratcc_eye_83um/verification"

NX, NY, NZ = 912, 900, 504
RETINA, OPTIC_NERVE, BRAIN = 77, 24, 18
SPLIT_DEG = 5.0        # same central/peripheral cut as 23_retina_distribution_figure.py

CONFIGS = {
    "SCL-ON": "RatCC_full_CLStim_JGND_912_900_504_Ring_Res_83um_with_retina",
    "SCL-IntraCranial": "RatCC_full_CLStim_NeedleGND_912_900_504_Ring_Res_83um_with_retina",
    "SCL-TransCranial": "RatCC_full_CLStim_PatchGND_912_900_504_Ring_Res_83um_with_retina",
    "ON-IntraCranial": "RatCC_full_JStim_NeedleGND_912_900_504_Ring_Res_83um_with_retina",
    "ON-TransCranial": "RatCC_full_JStim_PatchGND_912_900_504_Ring_Res_83um_with_retina",
    "IntraCranial-TransCranial": "RatCC_full_NeedleStim_PlateGND_912_900_504_Res_83um_with_retina",
}

INK, INK2, MUTED, SURF, GRID = "#0b0b0b", "#52514e", "#898781", "#fcfcfb", "#e1e0d9"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]  # slots 1-6


def read_J_raw(path) -> np.ndarray:
    """Read a legacy ``_J.raw`` volume into a (nx, ny, nz) array, x-fastest.

    Mirrors ``calculate_retina_opticnerve_EF.m``:
    ``Vof = reshape(a, nz, ny, nx); J = permute(Vof, [3, 2, 1]);`` -- MATLAB's
    column-major reshape is reproduced with ``order="F"``, then the same axis
    permutation gives a (nx, ny, nz) array indexed [x, y, z].
    """
    a = np.fromfile(path, dtype="<f4", count=NZ * NY * NX)
    if a.size != NZ * NY * NX:
        raise ValueError(f"{path}: expected {NZ*NY*NX} floats, got {a.size}")
    Vof = a.reshape((NZ, NY, NX), order="F")
    return np.transpose(Vof, (2, 1, 0))


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
    print("loading full 912x900x504 model for masks (labels only) ...", flush=True)
    model = VoxelModel.from_legacy(FULL_IN)
    assert model.world == (NX, NY, NZ), model.world

    retina_mask = model.labels == RETINA
    on_mask = model.labels == OPTIC_NERVE
    brain_mask = model.labels == BRAIN
    Yg, Zg = np.indices((NY, NZ))
    occ_plane = (Yg > 550) & (Zg > 300)          # same voxel rule as the MATLAB script
    occipital_mask = brain_mask & occ_plane[None, :, :]

    idx = np.argwhere(retina_mask)
    fr = Frame.eye(model, (RETINA,))
    pts_mm = fr.grid.centers_mm(idx)
    _, theta, _ = fr.spherical_of(pts_mm)
    ecc = 180.0 - theta
    central = ecc <= SPLIT_DEG
    peripheral = ~central
    print(f"retina: {len(idx):,} voxels, central(n={central.sum()}) / "
          f"peripheral(n={peripheral.sum()}); optic nerve {on_mask.sum():,}; "
          f"occipital {occipital_mask.sum():,}", flush=True)

    store = {}
    J_central, J_peripheral, J_occ = [], [], []
    for cfg, fname in CONFIGS.items():
        path = RESULTS / f"{fname}_J.raw"
        print(f"reading {path.name} ...", flush=True)
        J = read_J_raw(path)
        Jr = J[retina_mask]
        c = Jr[central].astype(np.float32)
        p = Jr[peripheral].astype(np.float32)
        o = J[occipital_mask].astype(np.float32)
        J_central.append(c); J_peripheral.append(p); J_occ.append(o)
        store[f"{cfg}_J_central"] = c
        store[f"{cfg}_J_peripheral"] = p
        store[f"{cfg}_J_occipital"] = o
        del J, Jr
    store["ecc_split_deg"] = np.array([SPLIT_DEG])
    np.savez_compressed(V / "six_configuration_distribution.npz", **store)

    fig = plt.figure(figsize=(14.5, 5.6), dpi=140)
    fig.patch.set_facecolor(SURF)
    gs = fig.add_gridspec(1, 3, wspace=0.32, left=0.055, right=0.98,
                          top=0.80, bottom=0.24)

    ax1 = fig.add_subplot(gs[0, 0])
    _box_strip(ax1, J_central, f"Central retina (ecc ≤ {SPLIT_DEG:g}°, n={central.sum()})",
              "|J| (raw units, A/m² ≈ µA/mm²)")
    ax2 = fig.add_subplot(gs[0, 1])
    _box_strip(ax2, J_peripheral, f"Peripheral retina (ecc > {SPLIT_DEG:g}°, n={peripheral.sum()})",
              "|J| (raw units, A/m² ≈ µA/mm²)")
    ax3 = fig.add_subplot(gs[0, 2])
    _box_strip(ax3, J_occ, f"Occipital (n={occipital_mask.sum()})",
              "|J| (raw units, A/m² ≈ µA/mm²)")

    fig.suptitle("Current density by region across your original 6 electrode-pair "
                "montages", fontsize=13.5, color=INK, x=0.055, ha="left", y=0.97)
    fig.text(0.055, 0.885,
             "Full 912x900x504 RatCC model, 83 µm · box = median/IQR, dots = per-voxel "
             "values · units as stored in each _J.raw (amplitude not independently "
             "known -- rescale if you have it) · central/peripheral split at "
             f"{SPLIT_DEG:g}° eccentricity from the posterior pole",
             fontsize=8.5, color=INK2)

    out = V / "views" / "six_configuration_distribution.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURF)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
