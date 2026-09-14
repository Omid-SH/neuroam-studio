"""Register the TES-GPS and VIRON clinical electrodes on the rat head model.

Run:  python examples/07_clinical_analogues.py [path/to/Model.in]

Two glaucoma neuromodulation trials, two very different electrodes:

  TES-GPS  (Lorenz et al., BMJ Open 2026;16:e112879, NCT06682962)
      OkuStim 2 / OkuEl M -- a single-use silver thread laid on the ocular
      surface at the *inferior limbus*, counter electrodes on the forehead.
      Symmetric biphasic rectangular pulses, 5 ms/phase, 20 Hz, up to 1 mA.

  VIRON    (Schittkowski et al., BMJ Open 2025;15:e091705, DRKS00029129)
      Two electrodes on *periorbital skin*, nothing touching the eye.
      10 Hz sinusoid, 600 uA peak-to-peak. Group 2 individualises the
      electrode positions from an MRI-based FEM (SimNIBS/CHARM).

This script places both on the RatCC anatomy through the v0.2 electrode
module, prints the QC that decides whether a placement is usable, and writes
one sparse overlay per montage.  Everything it does is also expressible in
the three configs it mirrors:

    configs/ratcc_tes_gps.json
    configs/ratcc_viron_periorbital.json
    configs/ratcc_viron_orbital6.json

Scaling, provenance and the azimuth convention: docs/CLINICAL_ELECTRODES.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neuroam.model import VoxelModel
from neuroam.frames import Frame
from neuroam.electrodes import ElectrodeSpec, Montage, region_from_spec

HERE = Path(__file__).resolve().parents[1]
DEFAULT = HERE / "samples/ratcc_eye_83um/RatCC_eye_180_160_220_83um_with_retina.in"
OUT = HERE / "samples/ratcc_eye_83um/electrode_designs"

SKIN, CORNEA, SCLERA, RETINA = 13, 66, 33, 77

# Anatomical azimuths in the fitted eye frame -- see docs/CLINICAL_ELECTRODES.md
# section 4 for how these were derived from the tissue profiles, and confirm
# them on the full 912x900x504 head model before publishing numbers.
PHI_SUPERIOR, PHI_NASAL, PHI_INFERIOR, PHI_TEMPORAL = 45.0, 135.0, 225.0, 315.0

# Human -> rat geometric scale, by globe radius: 3.155 mm / 12.0 mm
SCALE = 0.263


def pad(theta_deg: float, phi_deg: float, radius_mm: float) -> dict:
    """A skin-surface contact of the given radius, draped on the skin."""
    return {"type": "cylinder", "radius_mm": radius_mm, "height_mm": 1.0,
            "frame": "eye",
            "at": {"theta_deg": theta_deg, "phi_deg": phi_deg,
                   "on_surface": {"labels": [SKIN], "standoff_mm": 0.0,
                                  "max_mm": 14.0}},
            "conform": {"labels": [SKIN], "offset_mm": -0.10,
                        "thickness_mm": 0.30}}


def build(model, frames, name, entries, clean):
    """Register one montage on a *pristine* copy of the anatomy.

    ``Montage.build`` paints into ``model.labels``; three montages on one model
    object would see each other's hardware, so the anatomy is restored first.
    """
    model.labels[...] = clean
    specs = []
    for e in entries:
        try:
            region = region_from_spec(e["geometry"], model, frames)
        except ValueError as exc:
            print(f"   !! {e['name']}: {exc}")
            continue
        specs.append(ElectrodeSpec(
            name=e["name"], region=region, role=e["role"], label=e["label"],
            terminal="supernode", waveform=e.get("waveform"),
            overwrite=e.get("overwrite", "any"), min_voxels=e.get("min_voxels", 100)))
    return Montage(name, specs).build(model)


def main(in_path=DEFAULT) -> int:
    model = VoxelModel.from_legacy(Path(in_path))
    for old, new in ((101, CORNEA), (102, 12), (110, 9)):      # strip lab hardware
        model.labels[model.labels == old] = new
    frame = Frame.eye(model, (RETINA,))
    frames = {"eye": frame}
    clean = model.labels.copy()

    print(f"eye frame: centre {np.round(frame.origin_vox, 2).tolist()} vox, "
          f"globe radius {frame.radius_mm:.3f} mm "
          f"(sphere-fit RMS {frame.fit_rms_mm * 1e3:.0f} um)")
    print(f"           corneal axis {np.round(frame.e3, 4).tolist()}")

    # -- the limbus, measured rather than assumed ---------------------------
    from neuroam.geometry import Grid
    grid = Grid.from_model(model)
    theta = {}
    for lab in (CORNEA, SCLERA):
        P = grid.centers_mm(np.argwhere(model.labels == lab))
        r, th, _ = frame.spherical_of(P)
        theta[lab] = th[(r > 3.15) & (r < 3.50)]          # the outer shell only
    edges = np.arange(40, 115, 5.0)
    nc, _ = np.histogram(theta[CORNEA], edges)
    ns, _ = np.histogram(theta[SCLERA], edges)
    limbus = float(edges[np.argmax(ns > nc)])
    print(f"           limbus (cornea/sclera crossover on the outer shell) "
          f"~{limbus:.0f} deg -> corneal chord radius "
          f"{3.36 * np.sin(np.radians(limbus)):.2f} mm\n")

    montages = {}

    # -- 1. TES-GPS: OkuEl M thread at the inferior limbus ------------------
    okuel = {"type": "arc", "frame": "eye", "r_mm": 3.46, "theta_deg": 74.0,
             "tube_radius_mm": 0.10, "phi0_deg": PHI_INFERIOR,
             "span_deg": 100.0, "n_points": 81}
    print("TES-GPS -- OkuEl M silver thread on the ocular surface, inferior limbus")
    print(f"   contact length {frame.arc_length_mm(3.46, 74.0, 100.0):.2f} mm "
          f"(scaled from the >=10 mm specified for the human OkuEl M)")
    montages["TESGPS_OkuEl"] = build(model, frames, "TESGPS_OkuEl", [
        {"name": "OkuEl", "role": "source", "label": 101, "waveform": "TES",
         "geometry": okuel, "overwrite": "any", "min_voxels": 120},
        {"name": "CTR", "role": "ground", "label": 102, "overwrite": [0, SKIN],
         "min_voxels": 300, "geometry": pad(75.0, PHI_SUPERIOR, 2.8)}], clean)
    print(montages["TESGPS_OkuEl"].report(), "\n")

    # -- 2. VIRON: one periorbital cup + distant return ---------------------
    print("VIRON -- periorbital skin cup (10 mm clinical -> "
          f"{2 * 1.3:.1f} mm at scale {SCALE})")
    montages["VIRON_periorbital"] = build(model, frames, "VIRON_periorbital", [
        {"name": "E_sup", "role": "source", "label": 121, "waveform": "rtACS",
         "overwrite": [0, SKIN], "min_voxels": 150,
         "geometry": pad(85.0, PHI_SUPERIOR, 1.3)},
        {"name": "RET", "role": "ground", "label": 122, "overwrite": [0, SKIN],
         "min_voxels": 300, "geometry": pad(85.0, 165.0, 2.6)}], clean)
    print(montages["VIRON_periorbital"].report(), "\n")

    # -- 3. VIRON group 2: the six-cup orbital array (basis fields) ---------
    print("VIRON group 2 -- six independent orbital cups (Hunold et al. 2025 E1-E6).")
    print("   Cups the model cannot host are reported and skipped:")
    entries = [{"name": f"E{i}", "role": "source", "label": 120 + i,
                "waveform": f"E{i}", "overwrite": [0, SKIN], "min_voxels": 150,
                "geometry": pad(85.0, phi, 1.3)}
               for i, phi in enumerate([45.0, 105.0, 165.0, 225.0, 285.0, 345.0], 1)]
    entries.append({"name": "C1", "role": "ground", "label": 127,
                    "overwrite": [0, SKIN], "min_voxels": 300,
                    "geometry": pad(85.0, 15.0, 2.6)})
    montages["VIRON_orbital6"] = build(model, frames, "VIRON_orbital6", entries, clean)
    print(montages["VIRON_orbital6"].report(), "\n")

    OUT.mkdir(parents=True, exist_ok=True)
    for mont in montages.values():
        mont.save(OUT)
        print(f"wrote {OUT / mont.name}.overlay.npz / .montage.json")

    warned = [q.name for m in montages.values() for q in m.qc if q.warnings]
    print("\nQC warnings on:", ", ".join(warned) if warned else "none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
