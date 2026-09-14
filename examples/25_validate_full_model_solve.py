"""Validate NeuroAM's solver against the lab's precomputed full-model result.

Solves ONE of the 6 OEC-RGC montages (SCL-ON, i.e. the CLStim_JGND file) on
the full 912x900x504 model using the multiresolution mesh produced by the
lab's mesher.exe (25.1M unknowns, vs. 420M for the brute uniform grid), and
compares central/peripheral retina + optic-nerve + occipital current density
against the lab's own precomputed ``_J.raw`` for the same montage.

Run:  python examples/25_validate_full_model_solve.py <config-key>
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neuroam import fields as F
from neuroam.assembly import assemble_mrm, read_mrm
from neuroam.electrodes import ElectrodeSpec, register
from neuroam.frames import Frame
from neuroam.model import VoxelModel
from neuroam.solver import solve, unit_current_vector

REPO = Path(__file__).resolve().parents[1]
SAMP = REPO / "samples/ratcc_eye_83um"
RESULTS = Path(r"D:\OEC-RGC\Results\with retina 83um")
V = SAMP / "verification"

NX, NY, NZ = 912, 900, 504
RETINA, OPTIC_NERVE, BRAIN = 77, 24, 18
SPLIT_DEG = 5.0

CONFIGS = {
    "SCL-ON": "RatCC_full_CLStim_JGND_912_900_504_Ring_Res_83um_with_retina",
    "SCL-IntraCranial": "RatCC_full_CLStim_NeedleGND_912_900_504_Ring_Res_83um_with_retina",
    "SCL-TransCranial": "RatCC_full_CLStim_PatchGND_912_900_504_Ring_Res_83um_with_retina",
    "ON-IntraCranial": "RatCC_full_JStim_NeedleGND_912_900_504_Ring_Res_83um_with_retina",
    "ON-TransCranial": "RatCC_full_JStim_PatchGND_912_900_504_Ring_Res_83um_with_retina",
    "IntraCranial-TransCranial": "RatCC_full_NeedleStim_PlateGND_912_900_504_Res_83um_with_retina",
}


def read_J_raw(path) -> np.ndarray:
    a = np.fromfile(path, dtype="<f4", count=NZ * NY * NX)
    Vof = a.reshape((NZ, NY, NX), order="F")
    return np.transpose(Vof, (2, 1, 0))


def region_stats(arr: np.ndarray, name: str) -> dict:
    if arr.size == 0:
        return {"name": name, "voxels": 0}
    return {"name": name, "voxels": int(arr.size), "mean": float(arr.mean()),
            "median": float(np.median(arr)), "p95": float(np.percentile(arr, 95)),
            "max": float(arr.max())}


def main(cfg: str) -> int:
    fname = CONFIGS[cfg]
    in_path = SAMP / f"{fname}.in"
    mrm_path = SAMP / f"{fname}.mrm"
    if not in_path.exists():
        raise FileNotFoundError(f"missing {in_path} -- write it first")
    if not mrm_path.exists():
        raise FileNotFoundError(f"missing {mrm_path} -- run mesher.exe -f {fname} first")

    t0 = time.time()
    model = VoxelModel.from_legacy(in_path)
    print(f"[{cfg}] model loaded ({time.time()-t0:.0f}s)", flush=True)

    register(model, [
        ElectrodeSpec("stim", from_label=101, role="source", label=101,
                      terminal="supernode", waveform="Cur1"),
        ElectrodeSpec("gnd", from_label=102, role="ground", label=102,
                      terminal="supernode"),
    ])

    t0 = time.time()
    records = read_mrm(mrm_path)
    system = assemble_mrm(model, records)
    print(f"[{cfg}] assembled: n={system.n:,} nnz={system.G.nnz:,} "
          f"({time.time()-t0:.0f}s)", flush=True)

    t0 = time.time()
    r = solve(system, unit_current_vector(system, "Cur1"), method="cg",
              rtol=1e-7, precond="amg")
    print(f"[{cfg}] solved: {r.iterations} iters, {r.seconds:.0f}s, "
          f"residual {r.residual:.2e}", flush=True)

    grid = F.node_grid(system, r.v)
    # Multires system: most lattice points are not real mesh nodes and
    # node_grid() leaves them at 0 -- fill_hanging_nodes() interpolates them
    # (local averaging) before any gradient is taken. Skipping this step
    # differences real values against phantom zeros and produces huge
    # spurious E/J spikes in coarsely-clustered (far-from-electrode) tissue
    # -- exactly what corrupted the first pass (see verification/
    # validate_SCL-ON.json's absurd optic_nerve/occipital ratios vs the
    # retina ratios, which were already correct). A no-op if every node is
    # already a real mesh node (the uniform-assembly path).
    grid = F.fill_hanging_nodes(grid, system)
    Ex, Ey, Ez, Emag = F.efield(grid, model.dx)
    _, _, _, Jmag = F.current_density(model, Ex, Ey, Ez)

    retina_mask = model.labels == RETINA
    on_mask = model.labels == OPTIC_NERVE
    brain_mask = model.labels == BRAIN
    Yg, Zg = np.indices((NY, NZ))
    occ_plane = (Yg > 550) & (Zg > 300)
    occipital_mask = brain_mask & occ_plane[None, :, :]

    idx = np.argwhere(retina_mask)
    fr = Frame.eye(model, (RETINA,))
    pts_mm = fr.grid.centers_mm(idx)
    _, theta, _ = fr.spherical_of(pts_mm)
    ecc = 180.0 - theta
    central = ecc <= SPLIT_DEG

    Jr = Jmag[retina_mask]

    # cache raw per-voxel arrays -- reprocessing/plotting later shouldn't
    # need another ~45 min solve
    np.savez_compressed(
        V / f"raw_{cfg.replace(' ', '_')}.npz",
        J_central=Jr[central].astype(np.float32),
        J_peripheral=Jr[~central].astype(np.float32),
        J_optic_nerve=Jmag[on_mask].astype(np.float32),
        J_occipital=Jmag[occipital_mask].astype(np.float32),
    )

    neuroam_result = {
        "central": region_stats(Jr[central], "central retina"),
        "peripheral": region_stats(Jr[~central], "peripheral retina"),
        "optic_nerve": region_stats(Jmag[on_mask], "optic nerve"),
        "occipital": region_stats(Jmag[occipital_mask], "occipital"),
        "solve": {"unknowns": int(system.n), "iters": int(r.iterations),
                  "seconds": r.seconds, "residual": r.residual},
    }

    # ---- compare against the lab's own precomputed _J.raw for this montage
    lab_path = RESULTS / f"{fname}_J.raw"
    lab_result = None
    if lab_path.exists():
        Jlab = read_J_raw(lab_path)
        Jrl = Jlab[retina_mask]
        lab_result = {
            "central": region_stats(Jrl[central], "central retina"),
            "peripheral": region_stats(Jrl[~central], "peripheral retina"),
            "optic_nerve": region_stats(Jlab[on_mask], "optic nerve"),
            "occipital": region_stats(Jlab[occipital_mask], "occipital"),
        }

    out = {"config": cfg, "neuroam": neuroam_result, "lab_legacy": lab_result}
    out_path = V / f"validate_{cfg.replace(' ', '_')}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[{cfg}] wrote {out_path}")

    if lab_result:
        for region in ("central", "peripheral", "optic_nerve", "occipital"):
            a, b = neuroam_result[region]["p95"], lab_result[region]["p95"]
            ratio = a / b if b else float("nan")
            print(f"  {region:12s} p95: NeuroAM {a:10.4g}  lab {b:10.4g}  "
                  f"ratio {ratio:.3f}")
    return 0


if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else "SCL-ON"
    raise SystemExit(main(cfg))
