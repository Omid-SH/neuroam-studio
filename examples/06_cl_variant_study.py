"""Contact-lens electrode variants: how much does the design change the retina dose?

Run:  python examples/06_cl_variant_study.py

Five solves on the bundled RatCC 83 um eye crop (6.44 M unknowns; budget
~30 min and ~4 GB on a workstation, or run it on CARC):

  A  the shipped montage exactly as it is  -- legacy single-node terminals
  B  the same geometry with equipotential supernode terminals
  C  the shipped ring replaced by its parametric equivalent
  D  the ring slid 13 deg toward the corneal apex
  E  a 30 deg contact-lens dome instead of a ring

A and B isolate the terminal model; C validates the design module against the
hand-painted electrode; D and E are new designs.  Results in
``samples/ratcc_eye_83um/cl_study_results.json``.
"""
from __future__ import annotations

import gc
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
from neuroam.solver import electrode_impedance, solve, unit_current_vector

HERE = Path(__file__).resolve().parents[1]
IN = HERE / "samples/ratcc_eye_83um/RatCC_eye_180_160_220_83um_with_retina.in"
OUT = HERE / "samples/ratcc_eye_83um/cl_study_results.json"

R_SURF, TUBE = 3.360, 0.12      # mm: eye-surface radius, ring wire radius
results = []


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
    Jmag = F.current_density(model, Ex, Ey, Ez)[-1]
    retina = F.roi_metrics(F.mask_from_labels(model, [77]), Emag, Jmag, "retina")
    rec = {"case": case, "unknowns": int(system.n), "iters": int(r.iterations),
           "solve_s": round(r.seconds, 1), "residual": r.residual,
           "Z_ohm": electrode_impedance(system, r.v, "Cur1"),
           "retina": retina, **info}
    results.append(rec)
    OUT.write_text(json.dumps(results, indent=2, default=str))
    print(f"[{case}] retina E_mean {retina['E_mean_V_per_m']:.4g} V/m/A, "
          f"p95 {retina['E_p95_V_per_m']:.4g}, Z {rec['Z_ohm']:.1f} ohm\n",
          flush=True)
    del model, system, r, grid, Ex, Ey, Ez, Emag, Jmag
    gc.collect()


def shipped_node(model):
    """Untouched: the .in file's own single source and ground nodes."""
    return {"geometry": "shipped label 101", "terminal": "node",
            "source_node": list(model.sources[0].node),
            "ground_nodes": [list(g) for g in model.ground_nodes]}


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
        frame = Frame.eye(model, (77,))
        _, qc = register(model, [
            ElectrodeSpec(name, region=make_region(frame), role="source",
                          label=101, terminal="supernode", waveform="Cur1",
                          min_voxels=200),
            ElectrodeSpec("J", from_label=102, role="ground", label=102,
                          terminal="supernode")])
        return {"geometry": name, "terminal": "supernode",
                "n_vox": qc[0].n_voxels, "area_mm2": qc[0].surface_area_mm2,
                "warnings": qc[0].warnings}
    return build


def main() -> int:
    run("A_shipped_node", shipped_node)
    run("B_shipped_supernode", shipped_supernode)
    run("C_ring_th53", designed("ring_th53",
                                lambda f: f.ring(R_SURF, 53.3, TUBE)))
    run("D_ring_th40", designed("ring_th40",
                                lambda f: f.ring(R_SURF, 40.0, TUBE)))
    run("E_cap_30", designed("cap_30", lambda f: f.cap(R_SURF, 30.0, 0.15)))

    print(f"{'case':22s} {'E_mean':>10s} {'E_p95':>10s} {'Z ohm':>8s} {'iters':>6s}")
    for r in results:
        print(f"{r['case']:22s} {r['retina']['E_mean_V_per_m']:10.4g} "
              f"{r['retina']['E_p95_V_per_m']:10.4g} {r['Z_ohm']:8.1f} "
              f"{r['iters']:6d}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
