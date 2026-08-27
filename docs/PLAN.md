# NeuroAM Studio — architecture plan (v0.1)

Goal: an **ASCENT-style, config-driven, open pipeline** for multiscale neural-stimulation
modeling built on the lab's Admittance Method (AM) lineage (Cela/Lazzi 3D multiresolution
Impedance Method), with **direct sparse assembly** (no text netlist), reusable unit-current
basis fields, waveform superposition, and NEURON coupling — while staying **file-compatible**
with the legacy toolchain (`.in`, `.model`, `.mrm`, `.net`, `.cur`, `.vof`, coordinates/`.v`).

## Why this shape

- The lab's existing flow is: ModelTool → `.model`+`.in` → `mesher` → `.mrm` → `netgen` →
  `.net` (text SPICE netlist, 6.5 GB for a 54M-voxel model) → Python/C++ solver → `.vof`
  → `interp3` → `.v` → hoc `extracellular`.
- NeuroAM keeps the *formats* (so every existing lab model still runs) but replaces the
  netlist round-trip with direct in-memory assembly, and replaces manual per-study scripts
  with a versioned JSON config + CLI, mirroring ASCENT's sample/model/sim separation.

## Verified physics (from the Cela reference PDF + real netlists)

- `.in` `material id ρx ρx_im ρy ρy_im ρz ρz_im` — values are **resistivities** (Ω·m);
  insulator 1e7, metal electrode 1e-7.
- Each (multiresolution) voxel of size (sx,sy,sz) unit-voxels contributes **12 edge
  resistors** between its 8 corner nodes; per-axis value:
  `R_axis = 4·ρ_axis·s_axis / (s_perp1·s_perp2·Δ)`  (Δ = unitvoxelsize, m).
  Verified against netgen output: QuarterHead model, ρ=5000 Ω·m, Δ=0.25 mm →
  size-1 cube R=8e7 Ω, size-2 cube R=4e7 Ω (exact match).
- Parallel resistors across shared edges sum as conductances; ground node named `0`;
  DC/quasi-static current sources injected at named nodes; solve G·V = I.
- Because the system is linear, fields from unit-current sources superpose:
  `V(t) = Σ_k a_k(t)·V^(k)` — this replaces re-solving per waveform time step.

## Package layout

```
neuroam/
  materials.py   Material library: ρ/σ per axis, refs; JSON I/O; legacy .in parsing
  model.py       VoxelModel (labels, Δ, materials): .model/.in I/O, primitives, electrodes
  mesh.py        .mrm read/write; uniform mesh generation; (external mesher supported)
  assembly.py    Direct CSR assembly: uniform (vectorized) and multires (.mrm records)
  netlist.py     Legacy .net reader (fast) + netgen-equivalent writer (for cross-checks)
  solver.py      CG/BiCGStab/GMRES + preconditioners; direct solve; basis fields; superposition
  fields.py      V on nodes → voxel grids; E, J; .vof/.vavg export; ROI dose metrics
  waveforms.py   pulses, SCB/ACB biphasic, sine, arbitrary; legacy .cur I/O
  coupling.py    trilinear sampling at compartment coords (interp3-equivalent);
                 coordinates/.v file I/O; Vext(t) = basis × waveform
  neuron_link.py optional NEURON driver (e_extracellular playback, recording) — guarded import
  cache.py       content-hash keyed result store (solve once, reuse)
  pipeline.py    JSON config runner (model → solve → fields → couple → outputs)
  cli.py         python -m neuroam run/info/validate
  viz.py         slice maps, waveform and Vm plots (matplotlib, headless-safe)
```

## Validation strategy (tests/)

1. **Analytical** — point source in homogeneous medium: V(r)=ρI/(4πr) in the near field;
   two-layer slab between plate electrodes: exact 1D voltage division; current
   conservation: Σ|KCL residual| ≈ 0, ground current = −I_src.
2. **Legacy equivalence** — assembly from a model == matrix parsed from a
   netgen-equivalent `.net` written by us (identical semantics to the legacy Python
   solver's parser); multires path on an all-size-1 `.mrm` == uniform path.
3. **Multires sanity** — mixed-size synthetic mesh vs uniform solution in homogeneous
   medium (agreement within tolerance).
4. **Coupling** — trilinear sampler vs hand-computed values; `.v` files reproduce the
   Makewaveform2 convention (T × n_compartments).

## Deliberate v0.1 limits (documented, not hidden)

- Resistive/quasi-static only (legacy `Cap` netlist entries are read and ignored with a
  warning). Complex admittance is the next milestone.
- The multiresolution *mesher* is not reimplemented — use the existing lab mesher
  (`mesher.exe`/`mesher_CARC`, or the pure-Python `Mesher/mesher.py` from the unreleased
  repo) to produce `.mrm`; NeuroAM consumes `.mrm` directly.
- Workstation scale: chunked assembly handles ~1e8-voxel models in RAM; PETSc/distributed
  is a later phase.
- GUI: CLI + PNG/section outputs now; PySide6/PyVista viewer is Phase 2 (the API is
  designed so the GUI is a thin layer over `pipeline.py`).

## Roadmap after v0.1

1. Complex admittance + frequency sweeps; voltage-controlled electrodes.
2. Native Python multiresolution mesher integration (port of `Mesher/mesher.py`, Zarr-backed).
3. HDF5/Zarr result store; population-level NEURON runs; threshold search utilities.
4. PySide6 + PyVista interactive front-end.
5. Cross-species study configs (rabbit/rat/human) as `configs/` presets.

## IP note

The AM formulation and file formats derive from the Cela/Lazzi BEML toolchain and the
lab's unreleased PAM lineage. This project is for **internal lab use**; settle licensing
and authorship with the lab before any public release.
