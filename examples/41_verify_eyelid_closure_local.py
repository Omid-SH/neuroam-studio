"""Does closing the eye actually change the retinal dose? A *valid* test of
this, unlike the full-head "open"/"closed" pairs in examples/38.

Those reused SCL-ON's own pre-built ``.mrm`` multires mesh for every clinical
config. That mesh is a static file: ``assemble_mrm`` takes each mesh
element's material id from the mesh's own embedded ``records[:, 6]``, never
from ``model.labels`` -- so ``close_eyelid()`` painting ``model.labels``
provably cannot reach the bulk tissue conductivity through that path at all.
Confirmed directly: VIRON_periorbital_open and _closed have byte-identical
electrode footprints (same n_voxels, same centroid, per their own
registration reports), yet their solved fields still differed (up to
5,003 A/m^2, ~79M of 413M voxels) -- with *different* CG iteration counts
(140 vs. 142) and residuals between the two runs. That is the signature of
ordinary solver run-to-run non-determinism (AMG/CG on a numerically
identical system), not a physical effect of the anatomy change. The
"eye state doesn't move the bulk retina/occipital dose" conclusion in
docs/CLINICAL_ELECTRODES.md's §7.1 was reasoning about noise, not an anatomy
effect that never occurred in that experiment. This script redoes the
comparison somewhere it actually can occur.

On the small eye-only crop (``RatCC_eye_180_160_220_83um_with_retina``,
6.44M voxels), ``assemble_uniform`` builds G directly from ``model.labels``
every time -- no static mesh in between -- so painting the eyelid here
*does* reach the solve. Reuses the exact VIRON_periorbital electrode
geometry examples/07 already designed and validated on this crop.

Run:  python examples/41_verify_eyelid_closure_local.py
Writes: samples/ratcc_eye_83um/verification/eyelid_closure_local_check.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from neuroam import fields as F
from neuroam.assembly import assemble_uniform
from neuroam.electrodes import ElectrodeSpec, region_from_spec, register, close_eyelid
from neuroam.frames import Frame
from neuroam.model import VoxelModel
from neuroam.solver import solve, unit_current_vector

REPO = Path(__file__).resolve().parents[1]
SAMP = REPO / "samples/ratcc_eye_83um"
CROP_IN = SAMP / "RatCC_eye_180_160_220_83um_with_retina.in"
OUT = SAMP / "verification" / "eyelid_closure_local_check.json"

SKIN, CORNEA, SCLERA, RETINA = 13, 66, 33, 77
PHI_SUPERIOR = 45.0


def pad(theta_deg, phi_deg, radius_mm):
    return {"type": "cylinder", "radius_mm": radius_mm, "height_mm": 1.0,
            "frame": "eye",
            "at": {"theta_deg": theta_deg, "phi_deg": phi_deg,
                   "on_surface": {"labels": [SKIN], "standoff_mm": 0.0, "max_mm": 14.0}},
            "conform": {"labels": [SKIN], "offset_mm": -0.10, "thickness_mm": 0.30}}


def load_clean_model():
    model = VoxelModel.from_legacy(CROP_IN)
    for old, new in ((101, CORNEA), (102, 12), (110, 9)):   # strip shipped hardware
        model.labels[model.labels == old] = new
    return model


def run_one(closed: bool):
    t0 = time.time()
    model = load_clean_model()
    frame = Frame.eye(model, (RETINA,))
    if closed:
        n = close_eyelid(model, frame, apex_radius_mm=3.36, theta_max_deg=75.0,
                         thickness_mm=0.30, skin_label=SKIN)
        print(f"[{'closed' if closed else 'open'}] eyelid: painted {n:,} voxels", flush=True)

    frames = {"eye": frame}
    specs = [
        ElectrodeSpec(name="E_sup", role="source", label=121, waveform="rtACS",
                      terminal="supernode", overwrite=[0, SKIN], min_voxels=150,
                      region=region_from_spec(pad(85.0, PHI_SUPERIOR, 1.3), model, frames)),
        ElectrodeSpec(name="RET", role="ground", label=122, terminal="supernode",
                      overwrite=[0, SKIN], min_voxels=300,
                      region=region_from_spec(pad(85.0, 165.0, 2.6), model, frames)),
    ]
    _, qcs = register(model, specs)
    for qc in qcs:
        print(f"[{'closed' if closed else 'open'}] {qc.name}: {qc.n_voxels} vox, "
             f"centroid {np.round(qc.centroid_vox, 1).tolist()}", flush=True)

    system = assemble_uniform(model)
    print(f"[{'closed' if closed else 'open'}] assembled: n={system.n:,} "
         f"({time.time()-t0:.0f}s)", flush=True)
    r = solve(system, unit_current_vector(system, "rtACS"), method="cg",
             rtol=1e-7, precond="amg")
    print(f"[{'closed' if closed else 'open'}] solved: {r.iterations} iters, "
         f"{r.seconds:.0f}s, residual {r.residual:.2e}", flush=True)

    grid = F.node_grid(system, r.v)
    grid = F.fill_hanging_nodes(grid, system)
    Ex, Ey, Ez, Emag = F.efield(grid, model.dx)
    _, _, _, Jmag = F.current_density(model, Ex, Ey, Ez)

    retina_mask = model.labels == RETINA
    Jr = Jmag[retina_mask].astype(np.float64)
    result = {"n_retina_voxels": int(retina_mask.sum()),
             "median_A_per_A": float(np.median(Jr)), "p95_A_per_A": float(np.percentile(Jr, 95)),
             "max_A_per_A": float(Jr.max()), "iterations": r.iterations,
             "residual": r.residual, "seconds_total": round(time.time() - t0, 1)}
    return result, Jmag


def main():
    res_open, Jopen = run_one(closed=False)
    res_closed, Jclosed = run_one(closed=True)

    diff = np.abs(Jopen - Jclosed)
    comparison = {
        "open": res_open, "closed": res_closed,
        "raw_field_max_abs_diff": float(diff.max()),
        "raw_field_n_differing_gt_1e-9": int((diff > 1e-9).sum()),
        "retina_median_ratio_closed_over_open": res_closed["median_A_per_A"] / res_open["median_A_per_A"],
    }
    print(json.dumps(comparison, indent=2))
    OUT.write_text(json.dumps(comparison, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
