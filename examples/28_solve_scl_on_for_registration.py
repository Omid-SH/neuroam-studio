"""Solve one of the 6 OEC-RGC RatCC montages, save it as a reusable field.

This is the expensive part (assemble + AMG solve on the full 912x900x504,
25.1M-unknown multires mesh -- ~15-20 min per montage) done ONCE per config.
Unlike examples/25 (which only ever saved region-specific dose *statistics*,
not a reusable field), the solved result is saved twice, for two different
purposes:

- ``<config>.vof`` -- the project's own legacy node-voltage format (see
  docs/FORMATS.md), sitting right next to the source ``.in``/``.model``/
  ``.mrm`` files it was solved from. This is the real, documented,
  lab-portable deliverable: any tool that already reads a ``.vof`` (this
  project's own ``read_vof``, or the lab's other tools) can use it, and it
  is not tied to any particular neuron placement -- it's the field, full
  stop. At this mesh size that is a genuinely large file (~700 MB, ~25M
  lines) -- write_vof is vectorized (numpy string ops, one bulk write)
  specifically because the naive per-row loop it replaced does not scale
  to a mesh this size in reasonable time.
- ``.neuroam_cache/scl_on_field-<key>.npz`` -- an internal, content-hash
  keyed *performance* cache (see neuroam.cache) of the same solve, so this
  script (or anything else asking a different question about this exact
  model+electrode config -- a different crop, a different placement) skips
  straight to a fast reload instead of a 15-20 min re-solve. This is a
  cache, not a deliverable -- the ``.vof`` is the artifact to keep, point
  other tools at, or hand to a colleague.

Two separate per-cell scripts (D1 and A2i can't share a NEURON process --
see neuroam.neuron_link.load_hoc_cell's docstring) then register into this
saved field without re-solving, at any location, not just the one this
script also crops for convenience.

The placement anchor is the central-retina centroid -- the same
central/peripheral eccentricity split (<=5 deg) used in the SCL-ON
validation (examples/25) -- a real, already-characterized retina location
near the stimulating electrode, not an arbitrary pick. All 6 configs are
the same anatomy with different electrode placements, so this is the same
anchor voxel in every one of them.

Run:  python examples/28_solve_scl_on_for_registration.py [config-key]
      (config-key one of SCL-ON, SCL-IntraCranial, SCL-TransCranial,
       ON-IntraCranial, ON-TransCranial, IntraCranial-TransCranial;
       default SCL-ON)
Writes: samples/ratcc_eye_83um/<config>.vof
       samples/ratcc_eye_83um/verification/<config-key>_registration_cache.npz
       .neuroam_cache/scl_on_field-<key>.npz  (fast-reload cache, not the deliverable)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from neuroam import fields as F
from neuroam.assembly import assemble_mrm, read_mrm
from neuroam.cache import ResultCache, hash_inputs
from neuroam.electrodes import ElectrodeSpec, register
from neuroam.frames import Frame
from neuroam.model import VoxelModel
from neuroam.solver import solve, unit_current_vector

REPO = Path(__file__).resolve().parents[1]
SAMP = REPO / "samples/ratcc_eye_83um"

CONFIGS = {
    "SCL-ON": "RatCC_full_CLStim_JGND_912_900_504_Ring_Res_83um_with_retina",
    "SCL-IntraCranial": "RatCC_full_CLStim_NeedleGND_912_900_504_Ring_Res_83um_with_retina",
    "SCL-TransCranial": "RatCC_full_CLStim_PatchGND_912_900_504_Ring_Res_83um_with_retina",
    "ON-IntraCranial": "RatCC_full_JStim_NeedleGND_912_900_504_Ring_Res_83um_with_retina",
    "ON-TransCranial": "RatCC_full_JStim_PatchGND_912_900_504_Ring_Res_83um_with_retina",
    "IntraCranial-TransCranial": "RatCC_full_NeedleStim_PlateGND_912_900_504_Res_83um_with_retina",
}

RETINA = 77
SPLIT_DEG = 5.0
PAD_VOX = 40          # generous crop around the placement site (~6.6 mm cube at 83 um)


def main(cfg: str) -> int:
    fname = CONFIGS[cfg]
    in_path = SAMP / f"{fname}.in"
    mrm_path = SAMP / f"{fname}.mrm"
    out_cache = SAMP / "verification" / f"{cfg.replace(' ', '_')}_registration_cache.npz"
    if not in_path.exists() or not mrm_path.exists():
        raise FileNotFoundError(f"missing {in_path} or {mrm_path}")

    t0 = time.time()
    model = VoxelModel.from_legacy(in_path)
    print(f"[{cfg}] model loaded ({time.time()-t0:.0f}s)", flush=True)

    # every one of the 6 montages labels its stimulating electrode 101 and
    # its ground electrode 102 regardless of which *physical* electrode
    # (CL ring / J lead / needle / patch) occupies that role -- the same
    # convention examples/25's validation already relies on for all 6.
    register(model, [
        ElectrodeSpec("stim", from_label=101, role="source", label=101,
                      terminal="supernode", waveform="Cur1"),
        ElectrodeSpec("gnd", from_label=102, role="ground", label=102,
                      terminal="supernode"),
    ])

    # cheap fingerprint (path + size + mtime, not hashing 600+ MB of file
    # content) of exactly what determines the solved field: which source
    # model files, which electrode config, which solve tolerance.
    def _fingerprint(p: Path) -> tuple:
        st = p.stat()
        return (str(p), st.st_size, int(st.st_mtime))

    cache = ResultCache(REPO / ".neuroam_cache")
    field_key = hash_inputs(_fingerprint(in_path), _fingerprint(mrm_path),
                            "stim=101,gnd=102,supernode,Cur1", "rtol=1e-7,amg")

    vof_path = SAMP / f"{fname}.vof"

    def _solve_full_field():
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

        t0 = time.time()
        F.write_vof(vof_path, system, r.v)
        print(f"[{cfg}] wrote {vof_path} ({vof_path.stat().st_size/1e6:.0f} MB, "
             f"{time.time()-t0:.0f}s) -- the real, portable deliverable", flush=True)

        grid = F.node_grid(system, r.v)
        grid = F.fill_hanging_nodes(grid, system)
        Ex, Ey, Ez, Emag = F.efield(grid, model.dx)
        _, _, _, Jmag = F.current_density(model, Ex, Ey, Ez)
        return {"grid": grid.astype(np.float32), "jmag": Jmag.astype(np.float32)}

    hit = cache.get_array("scl_on_field", field_key) is not None
    full = cache.get_or_compute("scl_on_field", field_key, _solve_full_field)
    print(f"[{cfg}] full field {'loaded from cache' if hit else 'solved and cached'} "
         f"(.neuroam_cache/scl_on_field-{field_key}.npz)", flush=True)
    grid, Jmag = full["grid"].astype(np.float64), full["jmag"].astype(np.float64)

    if hit and not vof_path.exists():
        # a cache hit means _solve_full_field() (where .vof is normally
        # written, right alongside the solve that produced it) never ran --
        # get it anyway, from the cached grid, without a fresh 15-20 min
        # solve: re-assemble (cheap, no solve) just for system.node_coords,
        # then pull each real node's value straight out of the cached grid.
        t0 = time.time()
        records = read_mrm(mrm_path)
        system = assemble_mrm(model, records)
        print(f"[{cfg}] re-assembled for .vof only: n={system.n:,} "
             f"({time.time()-t0:.0f}s)", flush=True)
        c = system.node_coords
        full_at_nodes = grid[c[:, 0], c[:, 1], c[:, 2]]
        is_ground = np.zeros(len(c), dtype=bool)
        is_ground[system.ground_all] = True
        names = np.char.add(np.char.add(
            np.char.zfill(c[:, 0].astype(str), 4),
            np.char.zfill(c[:, 1].astype(str), 4)),
            np.char.zfill(c[:, 2].astype(str), 4))
        names = np.where(is_ground, "0", names)
        vals = np.char.mod("%.10g", full_at_nodes)
        vof_path.write_text("\n".join(np.char.add(np.char.add(names, " "), vals).tolist()) + "\n")
        print(f"[{cfg}] wrote {vof_path} ({vof_path.stat().st_size/1e6:.0f} MB) "
             f"from the cached grid, no re-solve", flush=True)

    # ---- placement anchor: central-retina centroid (same anatomy, same
    # anchor voxel, in every one of the 6 configs)
    retina_mask = model.labels == RETINA
    idx = np.argwhere(retina_mask)
    fr = Frame.eye(model, (RETINA,))
    pts_mm = fr.grid.centers_mm(idx)
    _, theta, _ = fr.spherical_of(pts_mm)
    ecc = 180.0 - theta
    central = ecc <= SPLIT_DEG
    anchor_vox = idx[central].mean(axis=0)
    print(f"[{cfg}] placement anchor (central retina centroid): {anchor_vox} voxels "
         f"({int(central.sum())} central-retina voxels averaged)", flush=True)

    world = np.asarray(model.world)
    lo = np.clip(np.floor(anchor_vox - PAD_VOX).astype(int), 0, None)
    hi_vox = np.minimum(np.ceil(anchor_vox + PAD_VOX).astype(int) + 1, world)
    hi_grid = np.minimum(hi_vox + 1, np.asarray(grid.shape))

    grid_crop = grid[lo[0]:hi_grid[0], lo[1]:hi_grid[1], lo[2]:hi_grid[2]]
    jmag_crop = Jmag[lo[0]:hi_vox[0], lo[1]:hi_vox[1], lo[2]:hi_vox[2]]
    labels_crop = model.labels[lo[0]:hi_vox[0], lo[1]:hi_vox[1], lo[2]:hi_vox[2]]
    print(f"[{cfg}] cropped: grid {grid_crop.shape}, Jmag {jmag_crop.shape}, "
         f"origin_vox {lo.tolist()}", flush=True)

    out_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_cache, grid_crop=grid_crop.astype(np.float32),
        jmag_crop=jmag_crop.astype(np.float32),
        labels_crop=labels_crop, origin_vox=lo, anchor_vox=anchor_vox,
        dx_m=np.asarray(model.dx), world=world,
        config=np.asarray(fname))
    print(f"[{cfg}] wrote {out_cache} ({out_cache.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else "SCL-ON"
    if cfg not in CONFIGS:
        sys.exit(f"unknown config {cfg!r}; choose from {list(CONFIGS)}")
    raise SystemExit(main(cfg))
