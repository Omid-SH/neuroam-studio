"""Design an electrode, register it on an anatomical model, and QC it.

Run:  python examples/05_electrode_design.py [path/to/Model.in]

With no argument this uses the bundled RatCC 83 um eye crop and

1. builds the **eye frame** from the anatomy alone (least-squares sphere fit
   to the retina shell, corneal axis away from the retina centroid);
2. reproduces the montage's existing contact-lens ring *parametrically* and
   reports how closely the analytic ring matches the shipped label-101 voxels;
3. designs three variants in the same frame -- the ring slid along the globe,
   and a contact-lens dome -- and prints the QC report for each;
4. writes a sparse overlay per variant (the anatomy is never duplicated) and
   can bake one back to legacy ``.model``/``.in`` for the lab mesher.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neuroam.model import VoxelModel
from neuroam.frames import Frame
from neuroam.geometry import Grid, rasterize
from neuroam.electrodes import ElectrodeSpec, Montage, register

HERE = Path(__file__).resolve().parents[1]
DEFAULT = HERE / "samples/ratcc_eye_83um/RatCC_eye_180_160_220_83um_with_retina.in"

R_SURF = 3.360      # mm from globe centre to the eye surface at the ring
TUBE = 0.12         # mm wire radius (240 um diameter)


def compare(idx: np.ndarray, truth: np.ndarray, dx_mm: float) -> str:
    """Agreement between a designed electrode and an existing painted one."""
    from scipy import ndimage
    lo = np.minimum(idx.min(0), truth.min(0)) - 2
    hi = np.maximum(idx.max(0), truth.max(0)) + 3
    A = np.zeros(hi - lo, bool); A[tuple((truth - lo).T)] = True
    B = np.zeros(hi - lo, bool); B[tuple((idx - lo).T)] = True
    dA = ndimage.distance_transform_edt(~A)
    dB = ndimage.distance_transform_edt(~B)
    iou = (A & B).sum() / (A | B).sum()
    haus = max(dB[A].max(), dA[B].max())
    return (f"IoU {iou:.3f}; max surface separation {haus:.2f} voxel "
            f"({haus * dx_mm * 1e3:.0f} um); designed {B.sum():,} vs "
            f"existing {A.sum():,} voxels")


def main(in_path=DEFAULT) -> int:
    in_path = Path(in_path)
    model = VoxelModel.from_legacy(in_path)
    grid = Grid.from_model(model)
    print(model.summary(), "\n")

    frame = Frame.eye(model, (77,))
    print(f"eye frame: centre {np.round(frame.origin_vox, 2).tolist()} vox, "
          f"globe radius {frame.radius_mm:.3f} mm (sphere-fit RMS "
          f"{frame.fit_rms_mm * 1e3:.0f} um)")
    print(f"           corneal axis {np.round(frame.e3, 4).tolist()}\n")

    # -- 1. reproduce the montage's existing ring, parametrically -----------
    existing = np.argwhere(model.labels == 101)
    if len(existing):
        ring = frame.ring(R_SURF, 53.3, TUBE)
        idx, _, _ = rasterize(ring, grid, supersample=3)
        print("parametric reproduction of the shipped contact-lens ring")
        print("   frame.ring(r=3.360 mm, theta=53.3 deg, tube=0.12 mm)")
        print("   " + compare(idx, existing, grid.dx_mm) + "\n")

    # -- 2. design variants in the eye frame -------------------------------
    variants = {
        "ring_th40": frame.ring(R_SURF, 40.0, TUBE),
        "ring_th53": frame.ring(R_SURF, 53.3, TUBE),
        "ring_th65": frame.ring(R_SURF, 65.0, TUBE),
        "cap_30": frame.cap(R_SURF, 30.0, 0.15),
    }
    out = in_path.parent / "electrode_designs"
    for name, region in variants.items():
        m = VoxelModel.from_legacy(in_path)
        m.labels[m.labels == 101] = 66          # clear the shipped electrode
        mont = Montage(name, [
            ElectrodeSpec(name, region=region, role="source", label=101,
                          terminal="supernode", waveform="Cur1",
                          min_voxels=200),
            ElectrodeSpec("J", from_label=102, role="ground", label=102,
                          terminal="supernode"),
        ]).build(m)
        print(mont.report())
        mont.save(out)
        print()
    print(f"overlays + QC written to {out}")
    print("bake one for the legacy mesher with:  "
          "python -m neuroam electrodes configs/ratcc_cl_ring.json --legacy out/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT))
