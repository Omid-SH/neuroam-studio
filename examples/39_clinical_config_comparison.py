"""Compare the 9 literature-derived non-invasive clinical configs
(examples/38) against the original 6 RatCC montages (examples/28+), at the
same three regions: central retina, peripheral retina, occipital cortex --
the direct answer to "how do these compare to our 6 configurations".

Per-voxel *distributions* (box + strip, matching
examples/36_neuroam_six_config_distribution.py's style), not just medians --
the shape of the distribution (how much of the region is near the peak vs.
near the floor) is exactly what a single summary number hides.

Each clinical config is scaled to *its own* paper-cited clinical amplitude
(neuroam.safety's CLINICAL_DRIVE, same table examples/38 scores safety
against), not the 200 uA convention used for the original 6 -- these are
different devices at different real doses, and pretending otherwise would
misrepresent every one of them. Un-scaled (per-ampere) values are also kept,
for a dose-independent comparison.

Run:  python examples/39_clinical_config_comparison.py
Writes: samples/ratcc_eye_83um/verification/clinical/config_comparison.npz
       samples/ratcc_eye_83um/verification/clinical/config_comparison_summary.json
       samples/ratcc_eye_83um/verification/views/clinical_config_comparison.png
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from neuroam.cache import ResultCache
from neuroam.frames import Frame
from neuroam.model import VoxelModel

from _sweep_common import CachedMontage, CONFIGS as ORIG_CONFIGS

# import examples/38 despite its filename starting with a digit -- reused so
# this script's cache lookups use the exact same geometry-aware key a solve
# wrote (see field_cache_key's docstring: keying on config_key alone let a
# corrected electrode position silently reuse a stale field from the old one)
_spec = importlib.util.spec_from_file_location(
    "clinical38", Path(__file__).resolve().parent / "38_solve_clinical_configs.py")
clinical38 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(clinical38)   # __name__ != "__main__", its own CLI block never runs

REPO = Path(__file__).resolve().parents[1]
SAMP = REPO / "samples/ratcc_eye_83um"
BASE_STEM = "RatCC_full_CLStim_JGND_912_900_504_Ring_Res_83um_with_retina"
VER = SAMP / "verification" / "clinical"
VIEWS = SAMP / "verification" / "views"

RETINA, BRAIN = 77, 18
SPLIT_DEG = 5.0

CLINICAL_DRIVE = {
    "TESGPS": ("TES", 1e-3, 5e-3),
    "VIRON_periorbital_open": ("rtACS", 300e-6, 1 / (10 * 2)),
    "VIRON_periorbital_closed": ("rtACS", 300e-6, 1 / (10 * 2)),
    "VIRON6_open": ("E1", 300e-6, 1 / (10 * 2)),
    "VIRON6_closed": ("E1", 300e-6, 1 / (10 * 2)),
    "TpES_open": ("TpES", 4.8e-3, 10e-3),
    "TpES_closed": ("TpES", 4.8e-3, 10e-3),
    "TcES": ("TcES", 1e-3, 10e-3),
    "SCL_Wrist": ("SCL", 200e-6, 5e-3),
}
CLINICAL_KEYS = list(CLINICAL_DRIVE)

INK, INK2, MUTED, SURF, GRID = "#0b0b0b", "#52514e", "#898781", "#fcfcfb", "#e1e0d9"
SLOTS6 = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
SLOTS9 = ["#6a3d9a", "#b15928", "#a6cee3", "#1f78b4", "#33a02c", "#fb9a99",
         "#e31a1c", "#ff7f00", "#cab2d6"]
SLOTS = SLOTS6 + SLOTS9
ALL_LABELS = list(ORIG_CONFIGS) + CLINICAL_KEYS


def load_clinical_field(config_key: str, qcs):
    cache = ResultCache(REPO / ".neuroam_cache")
    key = clinical38.field_cache_key(config_key, qcs)
    full = cache.get_array("clinical_field", key)
    if full is None:
        raise RuntimeError(f"{config_key}: no cached field -- run "
                           f"examples/38_solve_clinical_configs.py {config_key} first")
    return full["jmag"].astype(np.float64)


def _summary(a: np.ndarray) -> dict:
    return {"median": float(np.median(a)), "p95": float(np.percentile(a, 95)),
           "max": float(a.max()), "n": int(a.size)}


def _fmt_dose(amp_A: float) -> str:
    ua = amp_A * 1e6
    return f"{ua/1000:g} mA" if ua >= 1000 else f"{ua:g} uA"


def _tick_labels(dose_by_label: dict) -> list:
    return [f"{name}\n({_fmt_dose(dose_by_label[name])})" for name in ALL_LABELS]


def _box_strip(ax, arrays, title, ylabel, colors, tick_labels, log=True, max_points=4000):
    rng = np.random.default_rng(0)
    positions = np.arange(1, len(arrays) + 1)
    for pos, arr, color in zip(positions, arrays, colors):
        pts = arr if len(arr) <= max_points else rng.choice(arr, max_points, replace=False)
        x = pos + rng.uniform(-0.16, 0.16, size=len(pts))
        ax.scatter(x, pts, s=2.5, color=color, alpha=0.25, linewidths=0,
                  zorder=2, rasterized=True)
    bp = ax.boxplot(arrays, positions=positions, widths=0.6, showfliers=False,
                    patch_artist=True, zorder=3,
                    medianprops=dict(color=INK, linewidth=1.2),
                    whiskerprops=dict(color=INK2, linewidth=0.9),
                    capprops=dict(color=INK2, linewidth=0.9),
                    boxprops=dict(linewidth=0.9, edgecolor=INK2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.30)
    if log:
        ax.set_yscale("log")
    ax.set_title(title, fontsize=10.5, color=INK)
    ax.set_ylabel(ylabel, fontsize=8, color=INK2)
    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, fontsize=6.0, color=INK2, rotation=65, ha="right")
    ax.tick_params(axis="y", labelsize=7.5, colors=INK2)
    ax.axvline(len(ORIG_CONFIGS) + 0.5, color=INK2, ls="--", lw=1.0, zorder=1)
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def gather_data():
    print("loading shared base anatomy for masks ...", flush=True)
    base_model = VoxelModel.from_legacy(SAMP / f"{BASE_STEM}.in")
    labels = base_model.labels
    retina_mask = labels == RETINA
    idx = np.argwhere(retina_mask)
    fr = Frame.eye(base_model, (RETINA,))
    pts_mm = fr.grid.centers_mm(idx)
    _, theta, _ = fr.spherical_of(pts_mm)
    ecc = 180.0 - theta
    central = ecc <= SPLIT_DEG
    peripheral = ~central
    Yg, Zg = np.indices((labels.shape[1], labels.shape[2]))
    occ_mask = (labels == BRAIN) & ((Yg > 550) & (Zg > 300))[None, :, :]
    print(f"retina: {len(idx):,} (central {central.sum()}, peripheral {peripheral.sum()}); "
         f"occipital: {occ_mask.sum():,}", flush=True)

    store: dict = {}
    summary: dict = {}

    for cfg in ORIG_CONFIGS:
        m = CachedMontage(cfg)
        J = m.jmag * 200e-6      # A/m^2 at 200 uA -- matches examples/36's convention
        c, p, o = J[retina_mask][central], J[retina_mask][peripheral], J[occ_mask]
        store[f"{cfg}_central"], store[f"{cfg}_peripheral"], store[f"{cfg}_occipital"] = \
            c.astype(np.float32), p.astype(np.float32), o.astype(np.float32)
        summary[cfg] = {"dose_A": 200e-6, "central": _summary(c),
                        "peripheral": _summary(p), "occipital": _summary(o)}
        print(f"[orig] {cfg}: central med={summary[cfg]['central']['median']:.4g} "
             f"peripheral med={summary[cfg]['peripheral']['median']:.4g} "
             f"occipital med={summary[cfg]['occipital']['median']:.4g} A/m^2 (200uA)",
             flush=True)
        del m, J, c, p, o

    # qcs (registered electrode positions) for each clinical config -- needed
    # to compute the same geometry-aware cache key a solve wrote. Shared base
    # model reused across the 8 configs that share anatomy (SCL_Wrist keeps
    # its own contact-lens ring, so it gets its own base load), same pattern
    # as examples/38's own --report batch and examples/42's render batch.
    shared_keys = [k for k in CLINICAL_KEYS if k != "SCL_Wrist"]
    model0 = clinical38.load_base()
    frame0 = clinical38.Frame.eye(model0, (clinical38.RETINA,))
    clean0 = model0.labels.copy()
    qcs_by_key = {}
    for key in shared_keys:
        _, _, _, qcs_by_key[key] = clinical38.main(
            key, do_solve=False, model=model0, frame=frame0, clean_labels=clean0,
            return_model=True)
    _, _, _, qcs_by_key["SCL_Wrist"] = clinical38.main(
        "SCL_Wrist", do_solve=False, return_model=True)

    for key in CLINICAL_KEYS:
        wf, amp_A, pw_s = CLINICAL_DRIVE[key]
        J1A = load_clinical_field(key, qcs_by_key[key])   # A/m^2 per amp (unit-current basis field)
        Jdose = J1A * amp_A
        c, p, o = (Jdose[retina_mask][central], Jdose[retina_mask][peripheral],
                  Jdose[occ_mask])
        store[f"{key}_central"], store[f"{key}_peripheral"], store[f"{key}_occipital"] = \
            c.astype(np.float32), p.astype(np.float32), o.astype(np.float32)
        summary[key] = {"dose_A": amp_A, "central": _summary(c),
                        "peripheral": _summary(p), "occipital": _summary(o)}
        print(f"[clinical] {key}: central med={summary[key]['central']['median']:.4g} "
             f"peripheral med={summary[key]['peripheral']['median']:.4g} "
             f"occipital med={summary[key]['occipital']['median']:.4g} A/m^2 "
             f"(at {amp_A*1e6:.0f} uA)", flush=True)
        del J1A, Jdose, c, p, o

    VER.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(VER / "config_comparison.npz", **store)
    (VER / "config_comparison_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {VER / 'config_comparison.npz'} and config_comparison_summary.json")
    return store, summary


def load_saved_data():
    z = np.load(VER / "config_comparison.npz")
    store = {k: z[k] for k in z.files}
    summary = json.loads((VER / "config_comparison_summary.json").read_text())
    return store, summary


def make_figure(store: dict, summary: dict):
    # ---- figure: full per-voxel distributions, orig 6 (200 uA) vs 9 clinical
    # configs (each at its own paper-cited dose), box + strip, log-y
    fig, axes = plt.subplots(3, 1, figsize=(13, 15), dpi=140)
    fig.patch.set_facecolor(SURF)
    regions = [("central", "Central retina (ecc <= 5 deg)"),
              ("peripheral", "Peripheral retina (ecc > 5 deg)"),
              ("occipital", "Occipital cortex")]
    dose_by_label = {name: summary[name]["dose_A"] for name in ALL_LABELS}
    tick_labels = _tick_labels(dose_by_label)
    for ax, (field, title) in zip(axes, regions):
        arrays = [store[f"{c}_{field}"] for c in ALL_LABELS]
        _box_strip(ax, arrays, title, "|J| (A/m^2), each config at its own dose",
                  SLOTS, tick_labels)

    fig.suptitle("Original 6 RatCC montages (200 uA) vs. 9 literature-derived "
                "non-invasive configs (each at its own clinical dose, in parentheses)",
                fontsize=13, color=INK, x=0.01, ha="left", y=0.995)
    fig.text(0.01, 0.975,
             "Dashed line separates the two groups -- doses (shown under each name) are\n"
             "NOT matched between them. Box = median/IQR, dots = per-voxel values (4,000/config).\n"
             "_open/_closed pairs are the SAME full-head solve twice, not two anatomies -- the\n"
             "reused mesh is blind to the eyelid edit; see CLINICAL_ELECTRODES.md sec7.1 + "
             "examples/41.",
             fontsize=8, color=INK2, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.935))
    out_png = VIEWS / "clinical_config_comparison.png"
    fig.savefig(out_png, facecolor=SURF)
    plt.close(fig)
    print(f"wrote {out_png}")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--replot":
        store, summary = load_saved_data()
        print("re-plotting from the already-saved config_comparison.npz/.json "
             "(no data reload)", flush=True)
    else:
        store, summary = gather_data()
    make_figure(store, summary)


if __name__ == "__main__":
    main()
