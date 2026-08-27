# NeuroAM Studio

**Config-driven multiscale Admittance Method + NEURON pipeline** — an
ASCENT-style, scriptable successor to the lab's ModelTool → mesher → netgen →
solver → interp3 → hoc toolchain, built on the same physics and file formats.

- **Direct sparse assembly, no netlist.** The admittance matrix is built
  in-memory straight from the voxel model (uniform) or from a `.mrm`
  multiresolution mesh — verified *bit-for-bit identical* to the system the
  legacy `netgen` + solver pipeline builds (see `docs/VALIDATION.md`), without
  ever writing the multi-GB `.net` text file.
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
```

## Quick start

```bash
# run the bundled demo (simplified eye + contact-lens electrode)
python examples/02_eye_demo.py

# run any config
python -m neuroam run configs/sphere_demo.json

# summarize an existing lab model
python -m neuroam info "path/to/RatCC_full_...83um.in"

# solve an existing lab model directly (optionally with its .mrm)
python examples/03_legacy_model.py path/to/Model.in [path/to/Model.mrm]
```

Validation: `python -m pytest tests/` (26+ tests: analytic solutions, legacy
equivalence, real lab model, coupling, pipeline). Set
`NEUROAM_REFDATA=".../Admittance Method/C++ AM/ftdss2"` to enable the
real-model tests against `Sph_80`. Results: `docs/VALIDATION.md`.

## Pipeline config (ASCENT-style)

```json
{
  "neuroam": "0.1",
  "name": "my_experiment",
  "model":     { "world": [60,60,60], "unit_voxel_m": 2.5e-4,
                 "background": 1, "shapes": [ ... ] },
  "materials": { "inline": [ {"id": 1, "sigma": 0.3, "name": "tissue"} ] },
  "electrodes":[ {"name":"stim","role":"source","shape":{...},"waveform":"pulse"},
                 {"name":"ret","role":"ground","shape":{...}} ],
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
             fields, waveforms, coupling, neuron_link, cache, pipeline, cli, viz)
configs/     example JSON configs
examples/    01 analytic validation · 02 eye demo · 03 legacy model · 04 NEURON coupling
tests/       automated validation suite
docs/        PLAN.md (architecture/roadmap) · FORMATS.md (legacy formats) ·
             VALIDATION.md (results)
```

## Scope of v0.1 and roadmap

Resistive/quasi-static only (`Cap` entries ignored with a warning); the
multiresolution *mesher* is consumed, not reimplemented (use the lab mesher or
`Mesher/mesher.py` from the unreleased repo to make `.mrm` files); workstation
scale (~1e8 voxels). Next: complex admittance, COMSOL matched validation,
native mesher integration, HDF5/Zarr result store, population runs and
threshold maps, PySide6/PyVista viewer. Details in `docs/PLAN.md`.

## Status / IP

Internal lab software. The AM formulation and formats derive from the
Cela/Lazzi BEML toolchain and the lab's unreleased PAM lineage — settle
licensing and authorship with the lab before public release or a software
publication.
