# NeuroAM Studio

**Config-driven multiscale Admittance Method + NEURON pipeline** — an
ASCENT-style, scriptable successor to the lab's ModelTool → mesher → netgen →
solver → interp3 → hoc toolchain, built on the same physics and file formats.
Solve bioelectric fields on voxel anatomy, design and register electrodes in
physical units, couple the result into real NEURON compartmental neurons, and
visualize all of it — from whole-head context down to per-compartment
membrane voltage over time.

## What's here

- **Direct sparse assembly, no netlist.** The admittance matrix is built
  in-memory straight from the voxel model (uniform) or from a `.mrm`
  multiresolution mesh — verified *bit-for-bit identical* to the system the
  legacy `netgen` + solver pipeline builds (see `docs/VALIDATION.md`), without
  ever writing the multi-GB `.net` text file.
- **Electrodes as geometry, not index loops.** An electrode is a millimetre
  region (ring, dome, needle, cuff, swept wire, CSG of any of them) placed in
  an *anatomical frame* fitted from the model itself — `frame.ring(r=3.360,
  theta=53.3, tube=0.12)` puts a wire ring on the eye surface 53.3° off the
  corneal apex, at any voxel size, in any species. It connects through an
  explicit terminal model (equipotential **supernode** by default), paints into
  a **sparse overlay** instead of duplicating an 800+ MB `.model` per montage,
  and is QC'd before it is solved. Generated electrodes reproduce hand-built
  reference montages to Dice **0.89–0.98** (`docs/AM_VERIFICATION.md`). See
  `docs/ELECTRODES.md`.
- **Reusable basis fields.** Each electrode is solved once at unit current;
  arbitrary waveforms and multi-electrode patterns are superpositions —
  no re-solving per time step (linearity of the AM system).
- **Legacy-compatible I/O.** Reads/writes `.in`, `.model`, `.mrm`, `.net`,
  `.cur`, `.vof`, `.vavg`, raw volumes, coordinates and `.v` files — existing
  lab models and the existing hoc/NEURON flow keep working.
- **Real-neuron registration.** Two reconstructed rat retinal ganglion cell
  morphologies (D1, A2i) register into any solved field, anatomically aligned
  by their own soma-to-dendrite axis to the local retinal radial direction —
  not dropped in at a fixed orientation. Shareable, self-contained 3D
  HTML views (`neuroam.viz3d`, no VTK needed) show placement, local field, and
  activity over time together.
- **NEURON coupling.** Trilinear field sampling at compartment coordinates,
  `.v` matrix generation (Makewaveform2-equivalent), and an optional in-Python
  `e_extracellular` driver with recording and spike detection — validated
  against a real stimulation-parameter study across location, orientation,
  amplitude, waveform, and continuous pulse trains
  (`docs/STIMULATION_PARAMETER_STUDY.md`).
- **Cross-validated against the legacy pipeline.** All 6 full-head RatCC
  montages solved by NeuroAM agree with the lab's own archived OEC-RGC
  MATLAB/C++ AM solve of the identical model files to **~1%** at the retina,
  voxel for voxel — not just the same order of magnitude
  (`docs/VALIDATION.md`).
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
pip install -e ".[viz3d]"               # optional, shareable HTML 3D scenes (plotly + scikit-image)
pip install -e ".[amg]"                 # optional, algebraic-multigrid preconditioner (pyamg)
pip install -e ".[dev]"                 # optional, pytest for the test suite
```

Nothing here is required to *import* `neuroam` — only `numpy`/`scipy`/
`matplotlib` are non-optional runtime dependencies; everything else (NEURON,
PyVista, plotly, pyamg) is imported lazily inside the functions that need it,
so a plain `pip install -e .` is enough for the solver, electrode design, and
config pipeline on their own.

## Quick start

```bash
# analytic sanity check: point source in saline vs. the closed-form solution
python examples/01_point_source_validation.py

# end-to-end demo: simplified eye, CL vs needle return, retinal dose
python examples/02_eye_demo.py

# run any pipeline config
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

# full multiscale demo: tissue field -> extracellular drive -> NEURON spike
python examples/04_neuron_coupling.py
```

The viewer starts with a movable, color-coded tissue section through the
source electrode, so internal anatomy is visible rather than hidden behind the
outer surface. Use `--labels 24 33 66 77` to add persistent 3D surfaces for
the RatCC optic nerve, sclera, cornea and retina. The material color key prints
in the terminal; add `--legend` only when an in-window legend is useful.

Validation: `python -m pytest tests/` — 103 passing, 4 skipped (each skip
needs an external lab reference dataset via `NEUROAM_REFDATA`, not a missing
dependency). Results: `docs/VALIDATION.md`.

## Examples

Grouped by what question each one answers — every script is runnable on its
own; see each file's own docstring for exact inputs/outputs.

| # | Script | What it shows |
|---|---|---|
| 01 | `01_point_source_validation.py` | Analytic validation: point source vs. closed-form V |
| 02 | `02_eye_demo.py` | End-to-end miniature: simplified eye, CL vs. needle return, retinal dose |
| 03 | `03_legacy_model.py` | Solve any existing lab `.in`/`.model` directly |
| 04 | `04_neuron_coupling.py` | Full multiscale demo: tissue field → NEURON ball-and-stick spike |
| 05 | `05_electrode_design.py` | Design, register, and QC an electrode on an anatomical model |
| 06 | `06_cl_variant_study.py` | How much does contact-lens electrode geometry change retina dose? |
| 07 | `07_clinical_analogues.py` | Register real clinical electrodes (TES-GPS, VIRON) on the rat head |
| 20–21 | `20_view_model_and_field.py`, `21_verification_figure.py` | Early single-montage 3D views (predate the 6-montage work below) |
| 22–25 | `22_retina_distribution_study.py` … `25_validate_full_model_solve.py` | Retina current-density distributions; full-model solve validated against the lab's own precomputed result |
| 26–30 | `26_register_neuron.py` … `30_run_remaining_configs.py` | Register real D1/A2i RGC morphologies into all 6 solved RatCC montages |
| 31–35 | `31_stimulation_parameter_sweep.py` … `35_dose_response_figure.py` | 216-trial stimulation-parameter study (location, orientation, amplitude, waveform), then a corrected pulse-train validation — see `docs/STIMULATION_PARAMETER_STUDY.md` |
| 33, 36–37 | `33_verify_against_oecrgc.py`, `36_neuroam_six_config_distribution.py`, `37_neuroam_3d_six_configs.py` | Cross-validate all 6 montages against the legacy OEC-RGC pipeline (~1% agreement) and visualize the result, including an interactive 3D scene |

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
neuroam/     package: materials, model, mesh, assembly, netlist, solver,
             fields, waveforms, coupling, neuron_link, cache, pipeline, cli —
             geometry, frames, electrodes (electrode design/registration) —
             morphology (SWC I/O, real-neuron placement) —
             viz (matplotlib PNGs), viewer (interactive PyVista),
             viz3d (shareable plotly HTML scenes)
configs/     example JSON configs (sphere/legacy demos, RatCC electrode
             designs, clinical electrode analogues)
examples/    25 runnable scripts (numbered 01-07, then 20-37 -- no 08-19):
             01-07 core physics/electrodes, 20-37 visualization, real-neuron
             registration, and the stimulation-parameter study — see the
             Examples table above
samples/     reference anatomy/electrode/neuron-model data the examples run
             against (the 6 full-head RatCC montage `.model`/`.mrm`/`.vof`
             files are excluded from version control — 600+ MB each; see
             .gitignore and docs/PLAN.md for how to regenerate them)
tests/       automated validation suite (103 passing, `pytest tests/`)
docs/        ARCHITECTURE.md (system map) · DESIGN.md (implementation deep
             dive) · PLAN.md (roadmap) · FORMATS.md (legacy formats) ·
             VALIDATION.md / AM_VERIFICATION.md (results, incl. the
             cross-validation against the legacy OEC-RGC pipeline) ·
             STIMULATION_PARAMETER_STUDY.md (216-trial parameter study +
             pulse-train validation) · ELECTRODES.md / CLINICAL_ELECTRODES.md
             (electrode design)
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

This builds clean with zero warnings (`sphinx-build ... -W`), so it's ready
to connect to a Read the Docs project (`.readthedocs.yaml` at the repo root
already points at it) — that's a one-time manual step on readthedocs.org's
own dashboard (import the repo, no config needed beyond what's already
committed), not something this repo can do on its own.

## Scope and roadmap

Resistive/quasi-static only (`Cap` entries ignored with a warning); electrodes
are current-driven (no Dirichlet/voltage terminals yet) and rasterized without
partial-volume blending; the multiresolution *mesher* is consumed, not
reimplemented (use the lab mesher or `Mesher/mesher.py` from the unreleased
repo to make `.mrm` files); workstation scale (~1e8 voxels). Electrode design,
real-neuron registration and visualization, and a full stimulation-parameter
study with legacy cross-validation are done (v0.2–v0.4, `docs/VALIDATION.md`,
`docs/STIMULATION_PARAMETER_STUDY.md`). Next: complex admittance, COMSOL
matched validation, native mesher integration, HDF5/Zarr result store,
population runs and threshold maps, and a full PySide6 experiment GUI.
Details in `docs/PLAN.md`.

## Status / IP

Internal lab software. The AM formulation and formats derive from the
Cela/Lazzi BEML toolchain and the lab's unreleased PAM lineage — settle
licensing and authorship with the lab before public release or a software
publication.
