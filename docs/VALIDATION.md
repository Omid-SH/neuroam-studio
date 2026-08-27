# Validation record — NeuroAM v0.1.0

All results below were produced by the automated suite (`pytest tests/`) and
the example scripts on 2026-08-27 (Linux, Python 3.11+, NumPy 2.4, SciPy 1.17,
NEURON 9.0.2). Re-run any time; the real-lab-model tests activate when
`NEUROAM_REFDATA` points at the folder containing the `Sph_80.*` files
(`Python-AM_NEURON-Unreleased/Admittance Method/C++ AM/ftdss2`).

## 1. Analytical

| check | result |
|---|---|
| Point source in saline, near-field V vs ρI/4πr (difference form) | median error **1.9 %**, max < 10 % (60³ demo; 32³ test asserts median < 5 %) |
| Isotropy (±x, ±y, ±z at equal r) | ptp/mean < 1e-9 |
| Two-layer slab between metal plates vs series-resistance solution | total drop and interface division exact to **< 1e-6** relative |
| Anisotropic ρ = (1, 4, 1) vs isotropic | conduction along y follows ρ_y (drop ratio > 2× as expected for 3D spreading) |
| Current conservation (KCL residual) | relative residual < 1e-9 (direct), < rtol (iterative) |
| CG vs direct solve | agree to < 1e-6 relative |

## 2. Legacy-toolchain equivalence

| check | result |
|---|---|
| netgen resistor formula R = 4ρs∥/(s⊥1s⊥2Δ) | reproduces the QuarterHead netlist values exactly (8e7 Ω size-1, 4e7 Ω size-2, ρ=5000, Δ=0.25 mm) |
| direct uniform assembly vs parse of our netgen-equivalent `.net` | **identical matrices** (same sparsity, values to 1e-9) |
| `.mrm` assembly on all-size-1 mesh vs uniform assembly | **identical matrices** (1e-12) |
| **Real lab model** `Sph_80` (80³, 50 µm, 272,623 meshed voxels): uniform assembly with void handling vs assembly from the lab mesher's actual `Sph_80.mrm` | **identical matrices**; mesh covers exactly the defined-material voxels |
| `Sph_80` unit-current solve (CG, diag precond) | converges, relative residual < 1e-6, monotone decay from source |
| Multires (2× coarsened far region) vs uniform, homogeneous medium | near-source potential differences agree < 5 % |

## 3. Coupling and pipeline

| check | result |
|---|---|
| Trilinear sampler vs analytic linear field | exact (1e-12) |
| Legacy `.model`/`.in` round-trip and layout vs the AM_v10.2 reshape | byte-equivalent labels array |
| `.cur`, coordinates, unit-field, `.v` round-trips | exact |
| SCB / ACB 1:4 waveform charge balance | < 1e-9 |
| End-to-end JSON pipeline (build → solve → cache → fields → ROI → `.v`) | runs; manifest written; second run reuses cached basis field |
| NEURON extracellular drive (ball-and-stick, HH) | resting stays at −65 mV undriven; focal cathodic pulse elicits propagated spike |
| Full multiscale demo (examples/04): 2 mm tissue cube, 200 µm surface electrode, axon 150 µm above | activation threshold **≈ 83 µA** for 200 µs cathodic-first biphasic — physiologically plausible order of magnitude |

## Not yet validated (roadmap)

- COMSOL matched-model comparison (planned; the slab and sphere cases are the
  analytic stand-ins until then).
- Capacitive/complex admittance (Cap entries are read and ignored).
- Multiresolution meshes produced by the lab mesher with size > 1 on
  heterogeneous models (the interface treatment is netgen-identical by
  construction; add a numerical comparison when a small mixed-size lab mesh
  is available).
