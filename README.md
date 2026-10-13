# NeuroAM Studio

**A reproducible, multiscale Admittance Method + NEURON workflow.** NeuroAM
Studio solves bioelectric fields on voxel anatomy, registers electrodes in
physical units, couples extracellular potential to compartmental neuronal
models, and visualizes results from whole-anatomy context to membrane voltage
over time. It retains compatible AM file readers and writers while assembling
sparse systems directly in Python.

![NeuroAM Studio workflow](docs/_static/neuroam_workflow_hero.png)

## What's here

- **Direct sparse assembly, no netlist.** The admittance matrix is built
  in-memory straight from the voxel model (uniform) or from a `.mrm`
  multiresolution mesh—verified *bit-for-bit identical* to the system produced
  by a compatible `netgen` + solver workflow (see `docs/VALIDATION.md`)—without
  writing the multi-gigabyte `.net` text file.
- **Electrodes as geometry, not index loops.** An electrode is a millimetre
  region (ring, dome, needle, cuff, swept wire, CSG of any of them) placed in
  an *anatomical frame* fitted from the model itself — `frame.ring(r=3.360,
  theta=53.3, tube=0.12)` puts a wire ring on the eye surface 53.3° off the
  corneal apex, at any voxel size, in any species. It connects through an
  explicit terminal model (equipotential **supernode** by default), paints into
  a **sparse overlay** instead of duplicating an 800+ MB `.model` per montage,
  and is quality-checked before it is solved. Generated electrodes reproduce
  reference contacts with Dice **0.89–0.98** (`docs/VALIDATION.md`). See
  `docs/ELECTRODES.md`.
- **Reusable basis fields.** Each electrode is solved once at unit current;
  arbitrary waveforms and multi-electrode patterns are superpositions —
  no re-solving per time step (linearity of the AM system).
- **Legacy-compatible I/O.** Reads/writes `.in`, `.model`, `.mrm`, `.net`,
  `.cur`, `.vof`, `.vavg`, raw volumes, coordinates and `.v` files — existing
  established models and hoc/NEURON workflows remain interoperable.
- **Real-neuron registration.** Two reconstructed rat retinal ganglion cell
  morphologies (D1, A2i) register into any solved field, anatomically aligned
  by their own soma-to-dendrite axis to the local retinal radial direction —
  not dropped in at a fixed orientation. Shareable, self-contained 3D
  HTML views (`neuroam.viz3d`, no VTK needed) show placement, local field, and
  activity over time together.
- **NEURON coupling.** Trilinear field sampling at compartment coordinates,
  `.v` matrix generation, and an optional in-Python `e_extracellular` driver
  with recording and spike detection. Round-trip, analytic interpolation, and
  end-to-end regression tests cover the coupling path.
- **Cross-validated against an independent implementation.** All 6 full-head
  RatCC montages solved by NeuroAM agree with archived OEC-RGC MATLAB/C++ AM
  results for the identical model files to **~1%** at the retina,
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

# summarize an existing AM model
python -m neuroam info "path/to/RatCC_full_...83um.in"

# open an interactive 3D anatomy/electrode viewer
python -m neuroam view "path/to/RatCC_full_...83um.in"

# or print the summary and immediately open the viewer
python -m neuroam info "path/to/RatCC_full_...83um.in" --view

# solve an existing AM model directly (optionally with its .mrm)
python examples/03_legacy_model.py path/to/Model.in [path/to/Model.mrm]

# full multiscale demo: tissue field -> extracellular drive -> NEURON spike
python examples/04_neuron_coupling.py
```

The viewer starts with a movable, color-coded tissue section through the
source electrode, so internal anatomy is visible rather than hidden behind the
outer surface. Use `--labels 24 33 66 77` to add persistent 3D surfaces for
the RatCC optic nerve, sclera, cornea and retina. The material color key prints
in the terminal; add `--legend` only when an in-window legend is useful.

Validation: `python -m pytest tests/` — the 2026-10-13 snapshot is **106
passing, 4 skipped**. The skipped tests require a separately distributed
reference dataset selected by `NEUROAM_REFDATA`. Results and interpretation
limits are documented in `docs/VALIDATION.md`.

## Examples

Grouped by what question each one answers — every script is runnable on its
own; see each file's own docstring for exact inputs/outputs.

| # | Script | What it shows |
|---|---|---|
| 01 | `01_point_source_validation.py` | Analytic validation: point source vs. closed-form V |
| 02 | `02_eye_demo.py` | End-to-end miniature: simplified eye, CL vs. needle return, retinal dose |
| 03 | `03_legacy_model.py` | Solve an existing AM `.in`/`.model` directly |
| 04 | `04_neuron_coupling.py` | Full multiscale demo: tissue field → NEURON ball-and-stick spike |
| 05 | `05_electrode_design.py` | Design, register, and QC an electrode on an anatomical model |
| 06 | `06_cl_variant_study.py` | How much does contact-lens electrode geometry change retina dose? |
| 07 | `07_clinical_analogues.py` | Register real clinical electrodes (TES-GPS, VIRON) on the rat head |
| 20–21 | `20_view_model_and_field.py`, `21_verification_figure.py` | Early single-montage 3D views (predate the 6-montage work below) |
| 22–25 | `22_retina_distribution_study.py` … `25_validate_full_model_solve.py` | Retina current-density distributions and comparison with a reference full-model result |
| 26–30 | `26_register_neuron.py` … `30_run_remaining_configs.py` | Register real D1/A2i RGC morphologies into all 6 solved RatCC montages |
| 31–35 | `31_stimulation_parameter_sweep.py` … `35_dose_response_figure.py` | Exploratory location, orientation, amplitude, waveform, and pulse-train analyses; these are demonstrations, not clinical predictions |
| 33, 36–37 | `33_verify_against_oecrgc.py`, `36_neuroam_six_config_distribution.py`, `37_neuroam_3d_six_configs.py` | Cross-validate all 6 montages against the legacy OEC-RGC pipeline (~1% agreement) and visualize the result, including an interactive 3D scene |
| 38–41 | `38_solve_clinical_configs.py` … `41_verify_eyelid_closure_local.py` | Advanced registration and numerical checks for literature-derived electrode configurations; interpret only within the documented model limits |
| 42 | `42_electrode_3d_placement.py` | Three-dimensional electrode placement visualization |

## Pipeline configuration

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
examples/    runnable scripts for core physics, electrode registration,
             visualization, neuronal coupling, verification, and advanced
             modelling studies — see the Examples table above
samples/     reference anatomy/electrode/neuron-model data the examples run
             against (the 6 full-head RatCC montage `.model`/`.mrm`/`.vof`
             files are excluded from version control — 600+ MB each; see
             .gitignore; provide compatible external files to run those examples)
tests/       automated validation suite (`pytest tests/`)
docs/        GETTING_STARTED.md (installation and first run) · ARCHITECTURE.md
             (system map) · DESIGN.md (implementation) · FORMATS.md
             (interoperability) · VALIDATION.md (current evidence) ·
             LIMITATIONS.md (interpretation boundary) · ELECTRODES.md
             (electrode design and registration)
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

## Scope

The current solver is resistive and quasi-static; electrodes are
current-controlled and rasterized without partial-volume conductivity
blending. NeuroAM consumes existing multiresolution meshes and targets
workstation-scale in-memory problems. Neuronal examples are software
demonstrations, not clinical efficacy, threshold, or safety evidence. See
[`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) for the complete interpretation
boundary.
