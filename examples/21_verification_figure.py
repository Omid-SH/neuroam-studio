"""Reference vs parametric placement: geometry and retinal |J|, side by side.

Renders the figure that answers "do the generated electrodes reproduce the
hand-built montage?" — electrode geometry overlap, retinal |J| for both arms,
and their per-voxel difference. The Dice-score result this figure is built
from (0.893 CL ring / 0.975 J lead / 0.952 insulation) is already written up
in docs/AM_VERIFICATION.md; this script re-renders the figure from the raw
per-voxel data, not the summary numbers.

Needs the same ``fields.npz`` + ``agreement_tissue_matched.json`` inputs as
``examples/20_view_model_and_field.py`` — see that script's docstring for
how to regenerate them; ``agreement_tissue_matched.json`` is tracked in
this repo, ``fields.npz`` is not.

Run:
    python examples/21_verification_figure.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

HERE = Path(__file__).resolve().parent
V = HERE.parent / "samples" / "ratcc_eye_83um" / "verification"

INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
REF_C, PAR_C = "#2a78d6", "#e34948"          # validated categorical slots
# diverging ramp with a neutral grey midpoint (never a hue at the middle)
DIVERGING = LinearSegmentedColormap.from_list(
    "neuroam_div", [REF_C, "#e9e9e6", PAR_C])


def main():
    z = np.load(V / "fields.npz")
    lr, lp = z["labels_ref"], z["labels_param"]
    Ja, Jc = z["Jmag_A"].astype(float), z["Jmag_C"].astype(float)
    rows = json.loads((V / "agreement_tissue_matched.json").read_text())
    retina = (lr == 77) & (lr == lp)

    fig = plt.figure(figsize=(13.5, 7.4), dpi=140)
    fig.patch.set_facecolor(SURF)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1.0], hspace=0.32,
                          wspace=0.28, left=0.06, right=0.97, top=0.87,
                          bottom=0.09)

    # ---- row 1: electrode geometry, three orthogonal MIPs ----------------
    ref_e = np.isin(lr, [101, 102])
    par_e = np.isin(lp, [101, 102])
    for col, axis in enumerate("xyz"):
        ax = fig.add_subplot(gs[0, col])
        k = "xyz".index(axis)
        R, P = ref_e.any(axis=k).T, par_e.any(axis=k).T
        img = np.zeros(R.shape + (3,))
        img[...] = 1.0
        both = R & P
        img[R & ~P] = [0.16, 0.47, 0.84]        # reference only
        img[P & ~R] = [0.89, 0.29, 0.28]        # generated only
        img[both] = [0.42, 0.42, 0.40]          # agreement
        ax.imshow(img, origin="lower", interpolation="nearest")
        rem = [c for c in "xyz" if c != axis]
        ax.set_xlabel(rem[0], fontsize=8, color=INK2)
        ax.set_ylabel(rem[1], fontsize=8, color=INK2)
        ax.set_title(f"electrodes, projected along {axis}", fontsize=9.5,
                     color=INK)
        ax.tick_params(labelsize=7, colors=INK2)
    fig.text(0.06, 0.905,
             "hand-built (blue) · NeuroAM-generated (red) · agreement (grey)",
             fontsize=9, color=INK2)

    # ---- row 2a: retinal |J|, both arms, same scale ----------------------
    idx = np.argwhere(lr == 77)
    zc = int(np.median(idx[:, 2]))
    sl = (slice(None), slice(None), zc)
    vals = np.log10(np.maximum(Ja[retina], 1e-30))
    lo, hi = np.percentile(vals, [2, 98])
    for col, (arr, nm) in enumerate(((Ja, "hand-built montage"),
                                     (Jc, "NeuroAM-generated"))):
        ax = fig.add_subplot(gs[1, col])
        plane = np.where((lr == 77)[sl], np.log10(np.maximum(arr[sl], 1e-30)),
                         np.nan).T
        im = ax.imshow(plane, origin="lower", cmap="magma", vmin=lo, vmax=hi,
                       interpolation="nearest")
        ax.set_title(f"retinal |J| — {nm}", fontsize=9.5, color=INK)
        ax.set_xlabel("x", fontsize=8, color=INK2)
        ax.set_ylabel("y", fontsize=8, color=INK2)
        ax.tick_params(labelsize=7, colors=INK2)
        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
        cb.set_label("log₁₀ |J| (A/m² per A)", fontsize=7.5, color=INK2)
        cb.ax.tick_params(labelsize=7)

    # ---- row 2b: per-voxel difference ------------------------------------
    ax = fig.add_subplot(gs[1, 2])
    with np.errstate(divide="ignore", invalid="ignore"):
        d = 100.0 * (Jc - Ja) / np.maximum(Ja, 1e-30)
    plane = np.where((lr == 77)[sl], d[sl], np.nan).T
    lim = float(np.nanpercentile(np.abs(plane), 98)) or 1.0
    im = ax.imshow(plane, origin="lower", cmap=DIVERGING,
                   norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim),
                   interpolation="nearest")
    ax.set_title("difference (generated − hand-built)", fontsize=9.5, color=INK)
    ax.set_xlabel("x", fontsize=8, color=INK2)
    ax.set_ylabel("y", fontsize=8, color=INK2)
    ax.tick_params(labelsize=7, colors=INK2)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("% of reference |J|", fontsize=7.5, color=INK2)
    cb.ax.tick_params(labelsize=7)

    ret = [r for r in rows if r["region"] == "retina"][0]
    fig.suptitle("Electrode placement verification — rat eye, 83 µm, "
                 "CL-stim / J-ground", fontsize=13, color=INK, x=0.06,
                 ha="left", y=0.975)
    fig.text(0.06, 0.938,
             f"retina: mean |J| {ret['J_mean_pct']:+.1f}% · "
             f"E p95 {ret['E_p95_pct']:+.1f}% · r = {ret['pearson_r']:.3f} · "
             f"median per-voxel {ret['median_abs_rel_pct']:.1f}% · "
             f"only 736 of 6.34 M voxels differ in material",
             fontsize=9, color=INK2)

    out = V / "views" / "verification_figure.png"
    fig.savefig(out, facecolor=SURF)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
