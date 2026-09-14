# NeuroAM Studio

Config-driven multiscale Admittance Method + NEURON pipeline: solve
bioelectric fields on voxel anatomy, couple the result into NEURON
compartmental models, and visualize all of it — from whole-head context
down to per-compartment membrane voltage over time.

Start with **[Architecture](ARCHITECTURE.md)** for the system map, then
**[Design](DESIGN.md)** for the implementation walkthrough. The
**[API Reference](api/index.md)** is generated from the package's own
docstrings.

```{toctree}
:maxdepth: 2
:caption: Guide

ARCHITECTURE
DESIGN
PLAN
```

```{toctree}
:maxdepth: 1
:caption: Reference

FORMATS
ELECTRODES
CLINICAL_ELECTRODES
```

```{toctree}
:maxdepth: 1
:caption: Validation

VALIDATION
AM_VERIFICATION
```

```{toctree}
:maxdepth: 2
:caption: API

api/index
```

## Quick orientation

- **New to the codebase?** Read [Architecture](ARCHITECTURE.md) end to end
  first (~10 minutes) — it's written to be skimmable and to point at
  [Design](DESIGN.md) for anything that needs more depth.
- **Looking for a file format** (`.in`, `.model`, `.mrm`, `.net`, `.vof`,
  `.v`)? [FORMATS.md](FORMATS.md).
- **Want to know what's actually validated, and against what?**
  [VALIDATION.md](VALIDATION.md) and [AM_VERIFICATION.md](AM_VERIFICATION.md).
- **Building or registering an electrode?** [ELECTRODES.md](ELECTRODES.md)
  and [CLINICAL_ELECTRODES.md](CLINICAL_ELECTRODES.md).
- **Looking for a function or class?** [API Reference](api/index.md), or
  use the search box.
