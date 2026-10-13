# Getting started

This guide installs NeuroAM Studio, runs a self-contained verification
example, and introduces the configuration-driven pipeline. Python 3.9 or
newer is required.

## Install the core package

Create and activate an isolated environment, then install the project from the
repository root:

```bash
python -m venv .venv
# Linux or macOS
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -e .
```

The core installation includes NumPy, SciPy, and Matplotlib. Optional features
are available as extras:

```bash
python -m pip install -e ".[dev]"       # automated tests
python -m pip install -e ".[neuron]"    # NEURON coupling
python -m pip install -e ".[viewer]"    # interactive PyVista viewer
python -m pip install -e ".[viz3d]"     # self-contained Plotly scenes
python -m pip install -e ".[amg]"       # algebraic-multigrid preconditioner
```

The supplied `environment.yml` provides a Conda alternative.

## Verify the installation

Run the analytic point-source example:

```bash
python examples/01_point_source_validation.py
```

To run the automated suite:

```bash
python -m pytest -q
```

Tests that require optional software or separately distributed reference data
are skipped when those resources are unavailable. See [Validation](VALIDATION.md)
for the current test snapshot and numerical results.

## Run a configuration

The bundled sphere configuration exercises the main pipeline without external
anatomy files:

```bash
python -m neuroam run configs/sphere_demo.json
```

A configuration describes the anatomy, materials, electrodes, waveforms,
solver, outputs, and optional neuronal coupling. A minimal structure is:

```json
{
  "neuroam": "0.2",
  "name": "example",
  "model": {
    "world": [60, 60, 60],
    "unit_voxel_m": 0.00025,
    "background": 1,
    "shapes": []
  },
  "materials": {
    "inline": [
      {"id": 1, "sigma": 0.3, "name": "tissue"}
    ]
  },
  "electrodes": [],
  "solve": {"method": "cg", "rtol": 1e-8, "cache": true},
  "outputs": {"save": ["npz", "png"]}
}
```

For established AM datasets, point the model section at the existing files:

```json
{
  "model": {
    "legacy_in": "path/to/model.in",
    "mrm": "path/to/model.mrm"
  }
}
```

## Inspect models and electrodes

Use the command-line tools to inspect a model or validate electrode placement
before committing to a large solve:

```bash
python -m neuroam info path/to/model.in
python -m neuroam electrodes configs/ratcc_cl_ring.json
python -m neuroam view path/to/model.in
```

The electrode command reports painted volume, surface contact, connected
components, overlap, and terminal placement. Treat warnings as modelling
errors unless the configuration documents why they are acceptable.

## Choose the next guide

- [Architecture](ARCHITECTURE.md) explains the pipeline and package boundaries.
- [Detailed design](DESIGN.md) documents algorithms and implementation choices.
- [Electrode design](ELECTRODES.md) covers geometry, anatomical frames,
  registration, terminal models, and quality control.
- [Legacy file formats](FORMATS.md) documents interoperable AM and NEURON files.
- [Limitations](LIMITATIONS.md) defines the present modelling and validation
  boundaries.
- [API reference](api/index.md) lists the public Python interfaces.
