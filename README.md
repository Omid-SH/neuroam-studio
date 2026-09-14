# NeuroAM Studio

**Config-driven multiscale Admittance Method + NEURON pipeline** — an
ASCENT-style, scriptable successor to the lab's ModelTool → mesher → netgen →
solver → interp3 → hoc toolchain, built on the same physics and file formats.

- **Direct sparse assembly, no netlist.** The admittance matrix is built
  in-memory straight from the voxel model (uniform) or from a `.mrm`
  multiresolution mesh — verified *bit-for-bit identical* to the system the
  legacy `netgen` + solver pipeline builds (see `docs/VALIDATION.md`), without
  ever writing the multi-GB `.net` text file.
- **Electrodes as geometry, not index loops.** An electrode is a millimetre
  region (ring, dome, needle, cuff, swept wire, CSG of any of them) placed in
  an *anatomical frame* fitted from the model itself — `frame.ring(r=3.360,
  theta=53.3, tube=0.12)` puts a wire ring on the eye surface 53.3 deg off the
  corneal apex, at any voxel size, in any species. It connects through an
  explicit terminal model (equipotential **supernode** by default), paints into
  a **sparse overlay** instead of duplicating an 862 MB `.model` per montage,
  and is QC'd before it is solved. See `docs/ELECTRODES.md`.
- **Reusable basis fields.** Each electrode is solved once at unit current;
  arbitrary waveforms and multi-electrode patterns are superpositions —
  no re-solving per time step (linearity of the AM system).
- **Legacy-compatible I/O.** Reads/writes `.in`, `.model`, `.mrm`, `.net`,
  `.cur`, `.vof`, `.vavg`, raw volumes, coordinates and `.v` files — existing
  lab models and the existing hoc/NEURON flow keep working.
- **NEURON coupling.** Trilinear field sampling at compartment coordinates,
  `.v` matrix generation (Makewaveform2-equivalent), and an optional in-Python
  `e_extracellular` driver with recording and spike detection.
- **Reproducible runs.** One versioned JSON config per experiment; every run
  writes a `run_manifest.json` with hashes, solver diagnostics, ROI dose
  metrics, and the produced files. Content-hash caching: changing a waveform
  never re-solves the field.

## Install

```bash
conda env create -f environment.yml     # or: pip install -e .
conda activate neuroam
pip install neuron                      # optional, for the NEURON driver
pip install -e ".[viewer]"              # optional, interactive 3D viewer (PyVista)
pip install -e ".[viz3d]"               # optional, shareable HTML 3D scenes (plotly)
pip install -e ".[amg]"                 # optional, algebraic-multigrid preconditioner
```

## Quick start

```bash
# run the bundled demo (simplified eye + contact-lens electrode)
python examples/02_eye_demo.py

# run any config
python -m neuroam run configs/sphere_demo.json

# design-check electrodes (register + QC, no solve)
python -m neuroam electrodes configs/ratcc_cl_ring.json

# clinical trial electrodes on the rat anatomy (TES-GPS thread, VIRON periorbital)
python examples/07_clinical_analogues.py
python -m neuroam electrodes configs/ratcc_tes_gps.json

# summarize an existing lab model
python -m neuroam info "path/to/RatCC_full_...83um.in"

# open an interactive 3D anatomy/electrode viewer
python -m neuroam view "path/to/RatCC_full_...83um.in"

# or print the summary and immediately open the viewer
python -m neuroam info "path/to/RatCC_full_...83um.in" --view

# solve an existing lab model directly (optionally with its .mrm)
python examples/03_legacy_model.py path/to/Model.in [path/to/Model.mrm]
```

The viewer starts with a movable, color-coded tissue section through the
source electrode, so internal anatomy is visible rather than hidden behind the
outer surface. Use `--labels 24 33 66 77` to add persistent 3D surfaces for
the RatCC optic nerve, sclera, cornea and retina. The material color key prints
in the terminal; add `--legend` only when an in-window legend is useful.

Validation: `python -m pytest tests/` (26+ tests: analytic solutions, legacy
equivalence, real lab model, coupling, pipeline). Set
`NEUROAM_REFDATA=".../Admittance Method/C++ AM/ftdss2"` to enable the
real-model tests against `Sph_80`. Results: `docs/VALIDATION.md`.

## Pipeline config (ASCENT-style)

```json
{
  "neuroam": "0.2",
  "name": "my_experiment",
  "model":     { "world": [60,60,60], "unit_voxel_m": 2.5e-4,
                 "background": 1, "shapes": [ ... ] },
  "materials": { "inline": [ {"id": 1, "sigma": 0.3, "name": "tissue"} ] },
  "frames":    { "eye": {"type":"eye","retina_labels":[77]} },
  "electrodes":[ {"name":"CL","role":"source","label":101,"terminal":"supernode",
                  "waveform":"pulse",
                  "geometry":{"type":"torus","ring_radius_mm":2.695,
                              "tube_radius_mm":0.12,"frame":"eye",
                              "at":{"axial_mm":2.006}}},
                 {"name":"ret","role":"ground","label":102,"from_label":102} ],
  "waveforms": { "pulse": {"type":"biphasic","amp_A":1e-3,"ratio":4.0, ...} },
  "solve":     { "method": "cg", "rtol": 1e-8, "cache": true },
  "outputs":   { "save": ["npz","png","vof"], "rois": [ ... ] },
  "neuron":    { "coordinates": "cells/rgc_coords.txt" }
}
```

Legacy models plug in directly: `"model": {"legacy_in": "RatCC_full.in",
"mrm": "RatCC_full.mrm"}`.

## Layout

```
neuroam/     package (materials, model, mesh, assembly, netlist, solver,
             fields, waveforms, coupling, neuron_link, cache, pipeline, cli, viz,
             viewer, geometry, frames, electrodes)
configs/     example JSON configs
examples/    01 analytic validation · 02 eye demo · 03 legacy model ·
             04 NEURON coupling · 05 electrode design
tests/       automated validation suite
docs/        ARCHITECTURE.md (system map) · DESIGN.md (implementation deep dive) ·
             PLAN.md (roadmap) · FORMATS.md (legacy formats) ·
             VALIDATION.md / AM_VERIFICATION.md (results) ·
             ELECTRODES.md / CLINICAL_ELECTRODES.md (electrode design)
```

## Documentation

Start with [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the system
map (~10 min read), then [`docs/DESIGN.md`](docs/DESIGN.md) for the
module-by-module implementation walkthrough. A full API reference (built
from the package's own docstrings) plus all of the above is also set up
for Sphinx/Read the Docs — build it locally with:

```sh
pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
```

## Scope of v0.1 and roadmap

Resistive/quasi-static only (`Cap` entries ignored with a warning); electrodes
are current-driven (no Dirichlet/voltage terminals yet) and rasterized without
partial-volume blending; the
multiresolution *mesher* is consumed, not reimplemented (use the lab mesher or
`Mesher/mesher.py` from the unreleased repo to make `.mrm` files); workstation
scale (~1e8 voxels). Next: complex admittance, COMSOL matched validation,
native mesher integration, HDF5/Zarr result store, population runs and
threshold maps, and a full PySide6 experiment GUI. An interactive PyVista
model/electrode viewer is available now through `neuroam view`. Details in
`docs/PLAN.md`.

## Status / IP

Internal lab software. The AM formulation and formats derive from the
Cela/Lazzi BEML toolchain and the lab's unreleased PAM lineage — settle
licensing and authorship with the lab before public release or a software
publication.
