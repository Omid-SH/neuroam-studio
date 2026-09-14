"""Retina current-density *distribution* across CL electrode variants.

Companion to ``06_cl_variant_study.py``: that script keeps only scalar ROI
summaries (mean/median/p95/max) per config, which is enough for a table but
not for a distribution figure. This script re-solves the same five electrode
variants and keeps the *whole* retina's per-voxel |J|, |E|, tangential E, and
eccentricity from the posterior pole, plus the optic nerve's |J| and |E|, so
a box/strip distribution figure (like Retina-HFBlock Figure 2c) -- with any
central/peripheral eccentricity cut -- can be built without re-solving again.

Note on what these 5 configs are and are not: all five keep the *same*
SCL(source)-ON(ground) electrode pair as the OEC-RGC "SCL-ON" montage (this
crop's .in file is that exact montage) and only vary the SCL/contact-lens
electrode's own geometry (ring angle, dome vs ring, node vs supernode
terminal). They do not reproduce the other 5 montages in that MATLAB study
(SCL-IntraCranial, SCL-TransCranial, ON-IntraCranial, ON-TransCranial,
IntraCranial-TransCranial) -- those pair different anatomical electrode
sites and would need the IntraCranial/TransCranial electrode geometries and
the full 912x900x504 model, neither of which is in this repo yet.

Eccentricity split for retina: this rat model has no true fovea (rats are
afoveate), so there is no anatomical foveal boundary to read off the model.
The OEC-RGC MATLAB reference (``calculate_retina_opticnerve_EF.m``) defines
its "foveal" cap as within 2.5 deg of the posterior pole on the same
full-head model this crop comes from. This script keeps the *raw per-voxel
eccentricity* for every retina voxel instead of baking in one split, so the
central/peripheral boundary (5 deg, 2.5 deg, whatever) can be changed in
``23_retina_distribution_figure.py`` with no re-solve.

Also keeps the retina's E-field split into radial and tangential components
at each voxel (relative to the globe centre): RGC axons run *tangentially*
in the nerve-fiber layer, so the tangential component is a more relevant
proxy for axonal activation than |E| alone (a real activating-function
calculation needs the actual axon path, which this repo does not model).

Run:  python examples/22_retina_distribution_study.py
Budget: same as 06_cl_variant_study.py -- ~30 min / ~4 GB for all five solves.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neuroam import fields as F
from neuroam.assembly import assemble_uniform
from neuroam.electrodes import ElectrodeSpec, register
from neuroam.frames import Frame
from neuroam.model import VoxelModel
from neuroam.solver import solve, unit_current_vector

HERE = Path(__file__).resolve().parents[1]
IN = HERE / "samples/ratcc_eye_83um/RatCC_eye_180_160_220_83um_with_retina.in"
VER = HERE / "samples/ratcc_eye_83um/verification"
OUT_NPZ = VER / "retina_distribution.npz"
OUT_JSON = VER / "retina_distribution_summary.json"

RETINA, OPTIC_NERVE = 77, 24
R_SURF, TUBE = 3.360, 0.12         # mm: eye-surface radius, ring wire radius
CUR1_AMP_A = 0.0002                # bundled waveform amplitude (200 uA)


# --------------------------------------------------------- electrode variants
# same five arms as 06_cl_variant_study.py
def shipped_node(model):
    return {"geometry": "shipped label 101", "terminal": "node"}


def shipped_supernode(model):
    register(model, [
        ElectrodeSpec("CL", from_label=101, role="source", label=101,
                      terminal="supernode", waveform="Cur1"),
        ElectrodeSpec("J", from_label=102, role="ground", label=102,
                      terminal="supernode")])
    return {"geometry": "shipped label 101", "terminal": "supernode"}


def designed(name, make_region):
    def build(model):
        model.labels[model.labels == 101] = 66      # strip the shipped ring
        frame = Frame.eye(model, (RETINA,))
        _, qc = register(model, [
            ElectrodeSpec(name, region=make_region(frame), role="source",
                          label=101, terminal="supernode", waveform="Cur1",
                          min_voxels=200),
            ElectrodeSpec("J", from_label=102, role="ground", label=102,
                          terminal="supernode")])
        return {"geometry": name, "terminal": "supernode",
                "n_vox": qc[0].n_voxels, "area_mm2": qc[0].surface_area_mm2}
    return build


CONFIGS = [
    ("A_shipped_node", shipped_node),
    ("B_shipped_supernode", shipped_supernode),
    ("C_ring_th53", designed("ring_th53", lambda f: f.ring(R_SURF, 53.3, TUBE))),
    ("D_ring_th40", designed("ring_th40", lambda f: f.ring(R_SURF, 40.0, TUBE))),
    ("E_cap_30", designed("cap_30", lambda f: f.cap(R_SURF, 30.0, 0.15))),
]

store: dict = {}
summary: list = []
REFERENCE_SPLITS_DEG = (2.5, 5.0)   # just for the console/JSON preview


def stats(arr_A_per_A: np.ndarray, amp_A: float) -> dict:
    """Summary stats of a per-1-A field array, scaled to the waveform amp.

    A/m^2 per A, times A, is A/m^2 -- numerically identical to uA/mm^2
    (1 A/m^2 = 1 uA / 1e6 mm^2 * 1e6 = 1 uA/mm^2), matching the
    Retina-HFBlock figure's units.
    """
    a = arr_A_per_A.astype(float) * amp_A
    if a.size == 0:
        return {"voxels": 0}
    return {"voxels": int(a.size), "mean_uA_mm2": float(a.mean()),
            "median_uA_mm2": float(np.median(a)),
            "p95_uA_mm2": float(np.percentile(a, 95)),
            "max_uA_mm2": float(a.max())}


def run(case: str, build) -> None:
    t0 = time.time()
    model = VoxelModel.from_legacy(IN)
    info = build(model)
    system = assemble_uniform(model)
    print(f"[{case}] {system.n:,} unknowns "
          f"(merged={system.merge_inv is not None}), "
          f"assembled in {time.time() - t0:.0f}s", flush=True)
    r = solve(system, unit_current_vector(system, "Cur1"), method="cg",
              rtol=1e-7, precond="diag")
    print(f"[{case}] {r.iterations} iters, {r.seconds:.0f}s, "
          f"residual {r.residual:.2e}", flush=True)

    grid = F.node_grid(system, r.v)
    Ex, Ey, Ez, Emag = F.efield(grid, model.dx)
    _, _, _, Jmag = F.current_density(model, Ex, Ey, Ez)

    retina_mask = model.labels == RETINA
    idx = np.argwhere(retina_mask)
    fr = Frame.eye(model, (RETINA,))
    pts_mm = fr.grid.centers_mm(idx)
    _, theta, _ = fr.spherical_of(pts_mm)
    ecc = 180.0 - theta                      # 0 deg at the posterior pole

    # radial vs tangential E at each retina voxel (globe-centred)
    rhat = (pts_mm - fr.origin_mm)
    rhat /= np.maximum(np.linalg.norm(rhat, axis=1, keepdims=True), 1e-12)
    Evec = np.stack([Ex[retina_mask], Ey[retina_mask], Ez[retina_mask]], axis=1)
    Erad = np.abs(np.einsum("ij,ij->i", Evec, rhat))
    Etan = np.sqrt(np.maximum(Emag[retina_mask] ** 2 - Erad ** 2, 0.0))

    Jr, Er = Jmag[retina_mask], Emag[retina_mask]
    on_mask = model.labels == OPTIC_NERVE
    has_on = bool(on_mask.any())

    # store the *whole* retina, unsplit, plus each voxel's own eccentricity --
    # the central/peripheral cut is then a free choice at figure-time.
    store[f"{case}_ecc_deg"] = ecc.astype(np.float32)
    store[f"{case}_J_retina"] = Jr.astype(np.float32)
    store[f"{case}_E_retina"] = Er.astype(np.float32)
    store[f"{case}_Etan_retina"] = Etan.astype(np.float32)
    if has_on:
        store[f"{case}_J_optic_nerve"] = Jmag[on_mask].astype(np.float32)
        store[f"{case}_E_optic_nerve"] = Emag[on_mask].astype(np.float32)

    preview = {}
    for split in REFERENCE_SPLITS_DEG:
        c = ecc <= split
        preview[f"split_{split}deg"] = {
            "central": stats(Jr[c], CUR1_AMP_A),
            "peripheral": stats(Jr[~c], CUR1_AMP_A),
        }

    rec = {
        "case": case, "unknowns": int(system.n), "iters": int(r.iterations),
        "solve_s": round(r.seconds, 1), "residual": r.residual,
        "ecc_range_deg": [float(ecc.min()), float(ecc.max())],
        "retina_J_whole": stats(Jr, CUR1_AMP_A),
        "retina_Etan_whole": stats(Etan, CUR1_AMP_A),
        "optic_nerve_J": stats(Jmag[on_mask], CUR1_AMP_A) if has_on else None,
        "reference_splits": preview,
        **info,
    }
    summary.append(rec)
    OUT_JSON.write_text(json.dumps(
        {"cur1_amp_A": CUR1_AMP_A, "reference_splits_deg": list(REFERENCE_SPLITS_DEG),
         "cases": summary}, indent=2, default=str))
    np.savez_compressed(OUT_NPZ, **store)
    p5 = preview["split_5.0deg"]
    print(f"[{case}] at a 5 deg split: central J p95 "
          f"{p5['central']['p95_uA_mm2']:.3g} uA/mm^2 (n={p5['central']['voxels']}), "
          f"peripheral {p5['peripheral']['p95_uA_mm2']:.3g} uA/mm^2 "
          f"(n={p5['peripheral']['voxels']})\n", flush=True)


def main() -> int:
    VER.mkdir(parents=True, exist_ok=True)
    for case, build in CONFIGS:
        run(case, build)
    print(f"wrote {OUT_NPZ} and {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
