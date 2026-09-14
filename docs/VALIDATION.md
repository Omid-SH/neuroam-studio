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

---

## Electrode design and terminal models (v0.2, 2026-08-28)

Added: `tests/test_electrodes.py`, 24 tests (48 total in the suite, all passing).

### Geometry and registration

| check | result |
|---|---|
| Rasterized volume vs analytic (sphere, box, cylinder, torus, annulus) | within 5 % at supersample 4 |
| Volume invariance under rotation (3 arbitrary axes) | < 3 % |
| Bounding-box-limited rasterization vs brute force over the whole grid | identical voxel sets |
| SDF is a true distance after rigid motion | 1e-9 |
| Eye frame from a synthetic globe | centre within 0.6 vox, radius within 5 % |
| `snap_to_surface` standoff | within one voxel of the requested clearance |
| Overlay `save`/`load`/`apply` vs a baked label array | bit-identical |
| Legacy `.model`/`.in` export round-trip | labels identical; exported terminal node is inside the electrode |

QC catches, as regression tests: a declared terminal node that is not a node of
its electrode (the `sample.in` ground-node failure), electrodes sharing voxels,
and an `overwrite` policy blocking a paint.

### Terminal models

| check | result |
|---|---|
| Slab between two full-face supernodes vs `R = ρL/A` | rel. err **1e-13** |
| Disc on a half-space vs spreading resistance `ρ/(4a)` | 0.91 at a 32-voxel domain, rising toward 1 as the grounded walls recede |
| Supernode vs painted 1e-7 Ω·m metal, field away from the electrode | < 2 % |
| `distributed` terminal current vector | sums to 1 A, spread over the electrode's nodes |

### Real montage: RatCC 83 µm eye crop (6.44 M unknowns, CG + diag, rtol 1e-7)

Retina (label 77, 27,455 voxels) dose per ampere, and source-electrode access
impedance:

| case | geometry | terminal | E_mean (V/m/A) | E_p95 (V/m/A) | Z (Ω) | CG iters | solve |
|---|---|---|---|---|---|---|---|
| A | shipped label 101 | legacy single node | 1.8226e4 | 3.0283e4 | 552.3 | 2395 | 588 s |
| B | shipped label 101 | supernode | 1.8227e4 | 3.0283e4 | 552.3 | **1632** | **375 s** |
| C | `frame.ring(3.360, 53.3°, 0.12)` | supernode | 1.8167e4 | 3.0139e4 | 555.4 | 1628 | 365 s |
| D | `frame.ring(3.360, 40.0°, 0.12)` | supernode | 1.7092e4 | 2.9173e4 | 666.4 | 1624 | 326 s |
| E | `frame.cap(3.360, 30°, 0.15)` | supernode | 1.6567e4 | 2.9088e4 | 778.9 | 1630 | 368 s |

**A reproduces the archived result exactly** (E_mean 1.82e4, p95 3.03e4 from the
v0.1 run) — the new code paths do not disturb the legacy behaviour.

**B is the same physics at 32 % fewer iterations.** Making the electrodes
equipotential by construction changes the retina dose by 1.5e-4 relative and
the impedance by 2e-5 relative — the 1e-7 Ω·m metal bodies were already
equipotential — but removing those stiff resistors from the system improves the
conditioning enough to cut the solve from 588 s to 375 s. Supernode terminals
are therefore free accuracy *and* a 1.6× speedup on this class of model.

**C validates the design module end to end**: the parametric ring reproduces the
hand-painted electrode's retina dose to **0.3 %** (mean) and **0.5 %** (p95),
with a 0.6 % difference in access impedance, despite differing by up to 1.4
voxels in where the metal sits.

**D and E are new designs.** Sliding the ring 13° toward the corneal apex costs
6.2 % of retina E_mean and raises access impedance 21 %; a 30° contact-lens dome
costs 9.1 % despite 65 % more exposed area, at 41 % higher impedance. At the
200 µA Cur1 amplitude these are 3.63, 3.42 and 3.31 V/m in the retina
respectively (C, D, E).

Reproduce with `examples/06_cl_variant_study.py`; raw metrics in
`samples/ratcc_eye_83um/cl_study_results.json`.

### Cross-validation against the OEC-RGC legacy pipeline (v0.4, 2026-09-14)

The 6 full RatCC montages (912×900×504, 25.1M-unknown multires mesh —
`SCL-ON`, `SCL-IntraCranial`, `SCL-TransCranial`, `ON-IntraCranial`,
`ON-TransCranial`, `IntraCranial-TransCranial`) are the same `.in`/`.model`/
`.mrm` files the OEC-RGC project's own MATLAB/C++ AM pipeline solved and
archived (`calculate_retina_opticnerve_EF.m`, `<config>_J.raw`) — a
genuinely independent solver implementation on the identical anatomy and
electrode placement, not a re-derivation. `examples/33_verify_against_oecrgc.py`
compares them directly, voxel-for-voxel at the retina (material 77,
27,455 voxels):

**The retina masks agree exactly, not approximately.** OEC-RGC's own retina
definition is a synthetic geometric shell + angular cap, computed
independently of the `.model` file's material labels — but material 77 in
these files was *painted* from that exact shell/cap definition (per the
`.in` file's own header comment). Confirmed by an exact voxel-count match:
`labels==77` is 27,455 voxels; OEC-RGC's own `foveal_mask` (17 voxels) +
`peripheral_mask` (27,438 voxels) sum to the same 27,455, and spot-checked
coordinates land on identical voxels.

**The fields agree to ~1%, once scaled for units, not physics.** NeuroAM
solves a unit-current (1 A) basis field (`unit_current_vector`); OEC-RGC's
archived `_J.raw` volumes are scaled to this project's bundled 200 µA
`Cur1` waveform. Scaling NeuroAM's retina current density by 200 µA and
comparing directly against the legacy `_J.raw` value at every one of the
27,455 retina voxels, config by config:

| config | NeuroAM mean J (A/m², 200 µA) | OEC-RGC mean J (A/m²) | ratio (legacy/NeuroAM) mean ± std |
|---|---|---|---|
| SCL-ON | 37.23 | 37.30 | 1.0025 ± 0.0051 |
| SCL-IntraCranial | 21.74 | 21.63 | 0.9911 ± 0.0148 |
| SCL-TransCranial | 21.60 | 21.49 | 0.9900 ± 0.0165 |
| ON-IntraCranial | 11.30 | 11.26 | 1.0047 ± 0.0326 |
| ON-TransCranial | 11.91 | 11.87 | 1.0042 ± 0.0329 |
| IntraCranial-TransCranial | 0.688 | 0.681 | 0.9919 ± 0.0300 |

Every config's per-voxel ratio is within **1% of unity on average**, with a
per-voxel spread of 0.5–3.3% — small enough to attribute to solver
differences (iterative CG/AMG vs. the legacy solver's own method,
current-density finite-difference conventions at boundary voxels) rather
than a physics discrepancy. The per-config *ranking* also matches exactly:
`SCL-ON` (highest retina dose) ≫ `SCL-TransCranial` ≈ `SCL-IntraCranial` >
`ON-TransCranial` > `ON-IntraCranial` ≫ `IntraCranial-TransCranial`
(lowest) — the same ordering `docs/STIMULATION_PARAMETER_STUDY.md` §5.1
independently found from NeuroAM's own field-structure analysis alone
("only `SCL-ON` shows meaningful spatial structure"). Max values match to
4 significant figures against OEC-RGC's own archived
`Voxel_Report_MinMax.csv` summary (e.g. `SCL-ON` retina J_max: 383.85
here vs. 383.93 archived).

Requires the OEC-RGC project checked out locally (a separate, sibling lab
project, not part of this repo — set `NEUROAM_OECRGC_DIR` to its path).
Raw comparison: `examples/out/oecrgc_verify/summary.json`.

**Figures, from NeuroAM's own solve (not the legacy `_J.raw` archive):**

- `examples/36_neuroam_six_config_distribution.py` — the same box/strip
  comparison as `examples/24_six_configuration_figure.py` (which reads
  OEC-RGC's raw files directly), rebuilt from NeuroAM's own cached fields
  scaled to 200 µA, plus a fourth panel (optic nerve) the original script
  computed the mask for but never plotted. Writes
  `samples/ratcc_eye_83um/verification/views/six_configuration_distribution_neuroam.png`.
- `examples/37_neuroam_3d_six_configs.py` — one interactive 3D scene
  (`neuroam_six_config_current_density_3d.html`, ~33 MB, self-contained —
  open in a browser), all 6 montages' retina/optic-nerve/occipital current
  density at 200 µA in one crop, each region×config pair its own toggle
  (SCL-ON visible by default). Retina is a real isosurface; optic nerve and
  occipital cortex are colored point clouds. Needs the `viz3d` extra
  (`pip install plotly scikit-image`, or `pip install -e ".[viz3d]"`).
