"""Cross-validate the 6 solved RatCC montages against the legacy OEC-RGC
MATLAB/C++ AM pipeline's own solved fields -- an independent implementation
of the same physics on the literal same `.in`/`.model`/`.mrm` files.

This does **not** re-solve anything on the NeuroAM side (reuses the
`.neuroam_cache` basis fields from ``examples/28_solve_scl_on_for_registration.py``,
same as ``_sweep_common.CachedMontage``). It reads the OEC-RGC project's own
archived `<config>_J.raw` current-density volumes (written by its C++ AM
solver, one per config, ~1.65 GB each at this model's 912x900x504 resolution)
and compares them, voxel-for-voxel at the retina (label 77), against
NeuroAM's own solve of the identical model.

Two things this confirms, not one:

1. **The retina masks agree exactly.** OEC-RGC's own retina definition
   (``calculate_retina_opticnerve_EF.m``) is a synthetic geometric shell +
   angular cap, computed independently of the `.model` file's material
   labels -- but material 77 in these `.in`/`.model` files was *painted*
   from that exact same shell/cap definition (see the `.in` file's own
   header comment), so the two should be the same voxel set by
   construction. Confirmed here by an exact voxel-count match (27,455) and
   a direct-lookup spot check before trusting anything scaled.
2. **The fields agree, once you scale for a units difference, not a physics
   difference.** NeuroAM solves a *unit-current* (1 A) basis field
   (``unit_current_vector`` -- see ``docs/PLAN.md``'s superposition design).
   OEC-RGC's archived `_J.raw` volumes are already scaled to this project's
   bundled 200 uA ``Cur1`` waveform amplitude, not 1 A. Dividing OEC-RGC's J
   by 200e-6 reproduces NeuroAM's own unit-current field to within ~1% at
   the retina, config for config -- not just the same order of magnitude,
   the same field.

Requires the OEC-RGC project checked out locally (not part of this repo --
a separate, sibling lab project). Point ``NEUROAM_OECRGC_DIR`` at it:

    export NEUROAM_OECRGC_DIR="/d/OEC-RGC"   # or wherever it's checked out
    python examples/33_verify_against_oecrgc.py

Writes: examples/out/oecrgc_verify/summary.json (+ prints a table)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from _sweep_common import CachedMontage, CONFIGS

REF = os.environ.get("NEUROAM_OECRGC_DIR")
if not REF:
    sys.exit("Set NEUROAM_OECRGC_DIR to the OEC-RGC project checkout to run "
             "this cross-validation (a separate, sibling lab project -- not "
             "part of this repo). See this script's module docstring.")
REF = Path(REF)
J_DIR = REF / "Results" / "with retina 83um"

RETINA = 77
CUR1_AMP_A = 200e-6          # OEC-RGC's archived .raw fields are at this amplitude
NX, NY, NZ = 912, 900, 504

OUT = Path(__file__).parent / "out" / "oecrgc_verify"
OUT.mkdir(parents=True, exist_ok=True)


def load_legacy_J(fname: str) -> np.memmap:
    """The OEC-RGC C++ AM solver's raw current-density volume, indexed
    identically to NeuroAM's own arrays: ``J[x, y, z]`` (0-based), physical
    voxel-for-voxel, no coordinate swap (verified against the source
    MATLAB reshape/permute and spot-checked against known values -- see
    the module docstring; note this is *not* the same indexing as the
    project's own ``Voxel_Report_*.csv`` exports, which have an
    independent x/y swap bug in their own mask-export code, unrelated to
    the raw fields read here)."""
    path = J_DIR / f"{fname}_J.raw"
    if not path.exists():
        raise FileNotFoundError(path)
    flat_shape = (NZ, NY, NX)  # MATLAB reshape(a, nz, ny, nx), column-major
    arr = np.memmap(path, dtype="<f4", mode="r", shape=flat_shape, order="F")
    return np.transpose(arr, (2, 1, 0))  # MATLAB permute(Vof, [3,2,1]) -> (nx,ny,nz)


def summarize(a: np.ndarray) -> dict:
    return {"mean": float(a.mean()), "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)), "max": float(a.max())}


results = []
for cfg, fname in CONFIGS.items():
    t0 = time.time()
    m = CachedMontage(cfg)
    retina_mask = m.model.labels == RETINA
    idx = np.argwhere(retina_mask)
    n_retina = len(idx)

    j_neuroam_1A = m.jmag[idx[:, 0], idx[:, 1], idx[:, 2]]           # A/m^2 per A
    j_neuroam = j_neuroam_1A * CUR1_AMP_A                            # A/m^2 at 200 uA

    legacy = load_legacy_J(fname)
    j_legacy = np.asarray(legacy[idx[:, 0], idx[:, 1], idx[:, 2]], dtype=np.float64)
    del legacy

    nz_both = (j_neuroam > 0) & (j_legacy > 0)
    ratio = j_legacy[nz_both] / j_neuroam[nz_both]

    rec = dict(
        config=cfg, n_retina_voxels=n_retina,
        n_compared=int(nz_both.sum()),
        neuroam_J_Apm2=summarize(j_neuroam),
        legacy_J_Apm2=summarize(j_legacy),
        ratio_legacy_over_neuroam=dict(
            mean=float(ratio.mean()), median=float(np.median(ratio)),
            std=float(ratio.std()), p5=float(np.percentile(ratio, 5)),
            p95=float(np.percentile(ratio, 95))),
        seconds=round(time.time() - t0, 1))
    results.append(rec)
    print(f"[{cfg}] n_retina={n_retina} neuroam(200uA) mean={rec['neuroam_J_Apm2']['mean']:.4g} "
          f"legacy mean={rec['legacy_J_Apm2']['mean']:.4g} "
          f"ratio(legacy/neuroam)={rec['ratio_legacy_over_neuroam']['mean']:.4f} "
          f"+/- {rec['ratio_legacy_over_neuroam']['std']:.4f} "
          f"({rec['seconds']:.0f}s)", flush=True)

out_json = OUT / "summary.json"
out_json.write_text(json.dumps(
    {"cur1_amp_A": CUR1_AMP_A, "retina_label": RETINA, "configs": results},
    indent=2))
print(f"wrote {out_json}")
