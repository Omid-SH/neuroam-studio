# NeuroAM Studio — Detailed Design

The implementation walkthrough behind [ARCHITECTURE.md](ARCHITECTURE.md):
module internals, the physics/math each one encodes, and — deliberately —
the real engineering decisions and bugs-found-and-fixed that shaped the
current code, not just a description of the end state. File references are
`path:line` against the state of the repository this document was written
against; re-check line numbers if a module has moved on since.

## Contents

1. [Anatomy: `VoxelModel` and materials](#1-anatomy-voxelmodel-and-materials)
2. [Electrode design: geometry, frames, electrodes](#2-electrode-design-geometry-frames-electrodes)
3. [Assembly: the resistor-network physics](#3-assembly-the-resistor-network-physics)
4. [Solving: preconditioners and basis-field superposition](#4-solving-preconditioners-and-basis-field-superposition)
5. [Fields: reconstructing a full-resolution field from a multires solve](#5-fields-reconstructing-a-full-resolution-field-from-a-multires-solve)
6. [Coupling and NEURON](#6-coupling-and-neuron)
7. [Anatomical registration](#7-anatomical-registration)
8. [Caching strategy](#8-caching-strategy)
9. [Visualization: `viz3d.Scene`](#9-visualization-viz3dscene)
10. [Config pipeline and CLI](#10-config-pipeline-and-cli)
11. [Testing strategy](#11-testing-strategy)

---

## 1. Anatomy: `VoxelModel` and materials

`neuroam/model.py` — `VoxelModel` is a plain dataclass: `labels` (int array,
one material id per voxel), `dx` (voxel size, meters), `materials`
(`MaterialLibrary`), `sources`/`ground_nodes`/`terminals`. It reads/writes
the legacy `.in` (text: unit voxel size, per-material resistivity lines,
file references, spice-style source/ground node declarations) and `.model`
(a dense label volume in a specific `.model`-layout — `nz` blocks of `ny×nx`
text rows) format bit-for-bit.

`node_name(x, y, z) -> "xxxxyyyyzzzz"` (`model.py:38`) is the fixed-width
12-digit node identifier every legacy format shares (`.net`, `.vof`,
`nodename` directives) — a coordinate triple packed into one string key,
parsed back with `parse_node_name`. Every module that touches legacy text
I/O agrees on this convention.

`MaterialLibrary` (`materials.py`) stores per-axis resistivity (supporting
anisotropic tissue) plus optional real/imaginary parts (the imaginary part
is parsed from legacy `Cap` netlist lines and intentionally ignored by the
resistive solver — a documented v0.1 limit, not a silent drop).

## 2. Electrode design: geometry, frames, electrodes

This is the v0.2 milestone (see `docs/ELECTRODES.md` for the full writeup;
this section is the short version).

**`geometry.py`** — an SDF/CSG region library in millimetres (not voxels),
composable with `|`/`&`/`-`. Being resolution- and species-independent is
the point: the same `Region` expression rasterizes correctly at a 50 µm rat
model or a 200 µm human one. `rasterize(region, grid, lattice=n)` matters
for one specific real problem: the lab's hand-drawn electrodes were often
authored on a coarser lattice than the model's native voxel size (e.g. a
166 µm lattice inside an 83 µm model, every 2×2×2 block uniformly filled or
empty) — matching that lattice, not just the SDF boundary, is worth ~0.45
Dice when trying to reproduce a hand-built electrode parametrically
(`docs/AM_VERIFICATION.md`).

**`frames.py`** — `Frame.eye(model, retina_labels)` fits a sphere to a
tissue shell (least-squares), giving an anatomical coordinate system:
origin at the globe centre, `e3` along the corneal axis, `spherical_of(P)`
→ `(r, θ, φ)`. This is what makes "central retina" (`θ ≤ 5°` from the
posterior pole) or "the local radial direction at this point"
(`P - origin`, normalized) meaningful, reusable concepts instead of
per-script magic numbers. `normal_at(model, labels, point)` is the more
general (non-spherical-assumption) alternative — a signed-distance-field
gradient at a point, for surfaces that aren't well-approximated by a single
sphere.

**`electrodes.py`** — `ElectrodeSpec` describes an electrode by region +
role (`source`/`ground`/`target`) + terminal kind; `register(model, specs)`
paints labels and builds `Terminal` objects consumed by `assembly.py`.
`fit_spherical_band`/`fit_tube_path`/`refine_by_dice` go the other
direction: given a hand-built electrode's voxels, recover the parametric
spec that reproduces it (centre, radii, angular extent for a band; a
geodesic centre-line for a tube) — this is how `docs/AM_VERIFICATION.md`'s
Dice numbers were actually produced, and it's what turns a one-off
hand-drawn montage into something a future script can regenerate and
perturb.

## 3. Assembly: the resistor-network physics

`assembly.py`. Per the netgen-verified formula (`docs/PLAN.md`): a voxel of
per-axis size `(sx, sy, sz)` and resistivity `ρ_a` contributes 12 edge
resistors between its 8 corners; along axis `a`:

```
R_a = 4 ρ_a s_a / (s_b s_c Δ)        [Ω]        (Δ = unit voxel size, m)
```

Shared edges between adjacent voxels combine as parallel conductances —
this is what direct assembly does implicitly by accumulating into a COO/CSR
matrix, without ever materializing the legacy text netlist. Two independent
paths build the same kind of system:

- **`assemble_uniform`** — every voxel is size 1; nodes are the dense
  `(nx+1)(ny+1)(nz+1)` lattice, built with vectorized numpy (no Python loop
  over voxels).
- **`assemble_mrm`** — voxels come from `.mrm` records (the lab mesher's
  variable-size blocks); only lattice points actually referenced by some
  block's corners become nodes (`assembly.py:299-370`). This is what makes
  a 912×900×504 (~414M voxel) model solvable at 25.1M unknowns instead of
  ~415M — but it means most of the *original* full-resolution lattice is
  not a real degree of freedom at all (§5).

Both paths converge on one `System` dataclass (`G`, `node_coords`, `keep`/
`index_of` for the ground-eliminated reduction, `terminal_rows`/
`terminal_weights` for non-`node` terminals, and — since this session's
work — `records`, so `fields.fill_hanging_nodes` can reconstruct the
non-mesh lattice points later without re-deriving them from the `.mrm`
file). **Invariant, enforced by test, not just claimed**: on an all-size-1
`.mrm`, `assemble_mrm` and `assemble_uniform` must produce byte-identical
matrices (`test_mrm_size1_equals_uniform`).

**Terminal kinds** (`_reduce_and_index`, `assembly.py:110-206`): `node`
(legacy single injection node — kept for bit-exact reproduction of archived
results), `supernode` (electrode's nodes projected into one merged unknown
via `P^T G P`, `P` a 0/1 selector matrix — physically correct for an
equipotential conductor, and empirically ~1.6× fewer CG iterations since
removing the stiff 1e-7 Ω·m internal resistors improves conditioning), and
`distributed` (a fixed weight vector splitting one ampere across many
nodes, for realistic non-equipotential sources).

## 4. Solving: preconditioners and basis-field superposition

`solver.py`. `solve(system, I, method=..., precond=...)` wraps
CG/BiCGStab/GMRES/direct with diag/ILU/AMG preconditioning.
`make_preconditioner(..., kind="amg")` (`solver.py:43-63`) uses
`pyamg.smoothed_aggregation_solver` — on the real 6.44M-unknown rat-eye
model this took CG from ~2,400 iterations (378 s) to ~50 (~150 s); on the
25.1M-unknown full-head SCL-ON montage, 5,910 iterations (diag, 2,724 s) →
43 iterations (AMG, ~300–660 s depending on machine load). This is not a
marginal optimization — it's the difference between a solve that fits in a
coffee break and one that doesn't.

`unit_current_vector`/`solve_basis`/`superpose` (`solver.py:104-168`)
encode the linearity of the resistive system directly: solve once per
source electrode at 1 A, and any time-varying multi-electrode waveform is
then `V(t) = Σ_k a_k(t) V^(k)` — a superposition, never a re-solve. This is
the same idea `coupling.build_v_matrix`/`superpose_v_matrix` apply on the
NEURON side (a waveform scales a precomputed unit field, it doesn't
re-derive one). `solve_basis` is currently a plain sequential loop over
source electrodes — a known, not-yet-taken opportunity to parallelize
across electrodes (each solve is independent), noted in `docs/PLAN.md`'s
roadmap rather than implemented speculatively.

## 5. Fields: reconstructing a full-resolution field from a multires solve

`fields.py`. `node_grid(system, v)` scatters the reduced solution back onto
a dense `(nx+1, ny+1, nz+1)` array — for a multires system, only positions
that are *real* mesh nodes get a value; everything else is 0, which is
**wrong**, not just imprecise, if you take a gradient across it.

**`fill_hanging_nodes`** exists specifically to fix that, and its current
form is the result of a real correctness bug found mid-project:

- **v1 (retired)**: iterative neighbor-averaging diffusion, up to
  `max(nx,ny,nz)` passes over the whole grid. Slow (doesn't scale: at
  129³ it took 1.19 s vs. the current approach's 0.14 s, and the gap widens
  with size), and not even the *right* answer — a `.mrm` block is a
  12-edge-resistor, 8-corner element, so a block's interior is by
  construction the **trilinear** interpolation of its own 8 corners, not a
  diffusion average.
- **v2 (current)**: blocks are grouped by size (a handful of distinct sizes
  in a real mesh) and each group's corner-gather + trilinear matmul runs
  concurrently in its own thread (`ThreadPoolExecutor`) — numpy releases the
  GIL for the heavy array ops, so this scales with core count without
  copying the multi-GB grid between processes. **A subtlety that cost a
  debugging pass**: a coarse block's face-interior point can numerically
  coincide with a node an adjacent *finer* block already solved for exactly
  (a T-junction) — the first version let the coarse block's approximate
  estimate silently overwrite that real value depending on thread
  scheduling (caught by a determinism test: run the fill twice, expect
  identical output, and it wasn't). Fixed by masking out any already-real
  node from every group's contribution before scattering, and averaging
  (not racing) any residual disagreement between two blocks over a still-
  undefined point.

`efield`/`current_density` take the corner-potential gradient and Ohm's law
per axis; `write_vof`/`read_vof` are the legacy node-voltage format
(§ARCHITECTURE.md "Caching"); `write_vavg` is the voxel-center-average,
`.model`-layout variant used for ParaView-style exports.

## 6. Coupling and NEURON

**`coupling.py`** encodes the legacy `interp3`/`get_coords.hoc`/
`Makewaveform2.py` conventions exactly: coordinates in unit-voxel units,
a unit-field file as one value per compartment, a `.v` waveform matrix as
`T × n_compartments` (NEURON's `Stim` hoc plays column *i* into compartment
*i*'s `e_extracellular`). `sample_node_grid` is `scipy.ndimage.
map_coordinates` (trilinear, order=1) — already fully vectorized over every
compartment in one call, nothing to optimize there.

**`neuron_link.py`** is the guarded (import-`neuron`-lazily) NEURON driver.
`get_segment_coordinates`/`get_segment_edges` turn a live NEURON model into
plain arrays (positions via arclength interpolation of `x3d/y3d/z3d`;
connectivity via each section's own chain plus `parentseg()` for
inter-section edges) — this is what lets the rest of the codebase (viz3d,
morphology mapping) never import `neuron` at all. `run_extracellular` plays
a `(T, n_segments)` mV matrix into every segment's `e_extracellular` via
`Vector.play` (the idiomatic, fast NEURON pattern — not a manual
per-timestep loop) and records Vm + spikes via `APCount`.

**`load_hoc_cell(dirpath)`** builds a cell from the lab's own
`makecell.hoc`-style build script (`Import3d_SWC_read` + region biophysics
keyed by section-name pattern matching) and returns only the sections it
just created (`h.allsec()` is global NEURON process state; anything else
already built — another cell, a prior test — has to be diffed out, not
assumed absent). Two hard constraints this module works around, both found
by running it for real rather than assumed up front:

1. **Mechanism collision.** Two sibling cells built from byte-identical
   `.mod` sources (the D1/A2i pair) each ship their own separately
   -compiled `nrnmech.dll`. Loading the second one doesn't just fail to add
   anything — NEURON refuses to redefine an already-registered mechanism
   name, and the resulting hoc-level error leaves the interpreter unable to
   build anything afterward, even after the Python exception is caught. Fix:
   a process-wide fingerprint of already-loaded `.mod` filename sets
   (`_LOADED_MECHANISM_SETS`) skips the redundant, unsafe reload.
2. **Template redefinition.** A hoc `begintemplate`/`obfunc` is a top-level,
   process-global declaration; running the *same* build script twice (even
   for a different cell/config) silently aborts the second build partway
   through with no sections created. Fix: `load_hoc_cell` hashes the hoc
   script's own content and raises a clear, actionable `RuntimeError` if
   asked to run identical content twice in one process, rather than
   returning an empty result. Building two such cells means two processes —
   this is why every registration script in `examples/` that needs both D1
   and A2i runs each as a separate subprocess.

**`NeuronRunResult`** (`neuron_link.py`) is the save/replay boundary: `t_ms`,
`vm_mV` (T×n_seg), `spikes_ms`, registered geometry (`seg_vox`/`edges`,
optionally a denser `swc_vox`/`swc_edges`/`swc_to_seg` nearest-neighbor
mapping), `unit_v`, an optional cropped `field_crop` (small regardless of
the source model's size), and a free-form `meta` dict carrying every
parameter needed to know what produced this run. `save_neuron_run`/
`load_neuron_run` serialize it to one `.npz` (arrays + a JSON-string `meta`
array) — deliberately NEURON-free to load back (`examples/
27_replay_neuron_run.py` proves this: it regenerates all three views from a
saved run in under a second, having never imported `neuron`).

## 7. Anatomical registration

`morphology.py:estimate_depth_axis` + `coupling.py:place_aligned`/
`rotation_between_vectors` — added specifically because a retinal ganglion
cell registered by translation alone (whatever orientation its source SWC
reconstruction happened to have) does not put its dendrites toward the
photoreceptor side and its soma toward the vitreal side, which is what real
anatomy requires regardless of where on the (curved) retina the cell sits.

- **`estimate_depth_axis`**: PCA (the least-variance principal axis)
  restricted to soma + dendrites + the short proximal-axon subregions,
  deliberately *excluding* the bulk of the distal axon — which runs
  tangentially for hundreds of micrometers once in the nerve fiber layer
  and would otherwise dominate the estimate with an in-plane, not
  through-plane, direction. Sign-corrected (soma vs. dendrite-centroid
  projection) so the axis points soma → dendrites, not the reverse.
- **`rotation_between_vectors`**: Rodrigues' formula, the minimal rotation
  mapping one unit vector onto another (the rotation *about* that axis —
  "roll" — is left unconstrained, since aligning two vectors alone can't
  determine it).
- **`place_aligned`**: rotate about a pivot (the soma centroid) so the
  cell's local depth axis aligns with the *local* radial direction at the
  target point (`Frame.eye`'s globe center), then translate the pivot to
  that point.

Verified numerically (not just by construction): after registration, the
rotated depth axis matches the target radial direction to 1e-6 degrees, and
the dendrite region ends up ~16 µm farther from the globe center than the
soma — consistent with real GCL/IPL layering, for a placement anywhere on
the (curved) retina, not just one that happened to face a convenient
direction.

## 8. Caching strategy

Three distinct mechanisms, each solving a different problem — conflating
them was an early mistake this project corrected once the difference
actually mattered in practice:

| Mechanism | What it holds | Who it's for |
|---|---|---|
| `cache.ResultCache` | content-hash-keyed arrays (e.g. a full solved grid + \|J\|) under `.neuroam_cache/` | internal reuse only — a second script asking a different question about the same model+electrode config |
| `.vof` (`fields.write_vof`) | the legacy node-voltage format, one line per real mesh node | the actual deliverable — portable, documented, usable by any tool that reads `.vof`, independent of any neuron |
| `NeuronRunResult` (`neuron_link.save_neuron_run`) | one NEURON run's full time series + geometry + a small field crop | answering a different question about *that run* without NEURON or the AM solver |

`ResultCache.get_or_compute(kind, key, compute)` (`cache.py:58-66`) is
intentionally simple: hash the exact inputs that determine the output
(`hash_inputs` — arrays by content, everything else by canonical JSON),
check for a cached array file, compute and store on miss. The fingerprint
for a full-head solve is cheap on purpose — path + size + mtime of the
`.in`/`.mrm` files, not their multi-hundred-MB content — since hashing
hundreds of megabytes just to check a cache would itself take real time.

## 9. Visualization: `viz3d.Scene`

A `Scene` accumulates typed layers — `add_materials` (isosurface around a
label set, marching cubes + optional Laplacian smoothing to take the stair-
steps off), `add_electrode`, `add_field_on_surface` (color a structure's own
surface by a field — current density "on the retina"), `add_field_isosurface`,
`add_field_points`, `add_slice`, `add_neuron` (a morphology as connected
line segments, NaN-broken into one Plotly trace, optionally colored by a
per-point value) — into one `go.Figure` with a click-to-toggle legend. Every
high-level `view_*` function is a curated composition of these layers for
one specific question; identity is carried by a small fixed chromatic
palette (source=red, ground=blue, target=green, context=grey), validated
CVD-safe, never color-alone (every layer has a legend entry and hover text).

**`view_neuron_placement_editor`** is the one departure from "static,
self-contained figure": Plotly has no built-in drag/rotate gizmo for a
single trace, so it embeds ordinary HTML range inputs plus a small
hand-written JS transform — translate/rotate sliders recompute every
point's position client-side (mirroring `coupling.transform_coordinates`'s
rotation math exactly: intrinsic Rx·Ry·Rz about the point cloud's own
centroid) and push the result into the neuron trace via `Plotly.restyle`.
Nothing talks back to Python; the payoff is a live readout that prints the
exact `place_aligned(...)`/`transform_coordinates(...)` call to make once a
placement looks right — the number you saw previewed is the number that
actually runs. (When publishing such a page somewhere with a stricter
script allowlist than a local file — e.g. this project's own artifact
hosting, which only permits a small CDN allowlist — `plotlyjs_src` accepts
an explicit URL instead of Plotly's own CDN.)

## 10. Config pipeline and CLI

`pipeline.py:run(config)` chains model-loading → electrode registration →
basis-field solve (through `ResultCache`) → field computation → NEURON
coupling → file outputs from one JSON spec, in the spirit of ASCENT's
config-driven workflow. `cli.py` exposes `run`/`info`/`validate`/`view` as
`python -m neuroam ...`. Every example script in `examples/` is a thinner,
more explicit alternative to the config path — useful when a script needs
to do something the JSON schema doesn't cover yet (a new registration
strategy, a new view combination), and the intended place to prototype
before it's worth promoting into the pipeline schema.

## 11. Testing strategy

107 tests (`tests/`), organized by what they guarantee:

- **Analytical** (`test_solver_analytic.py`) — point source in a
  homogeneous medium vs. `V=ρI/4πr`; two-layer slab vs. exact 1D voltage
  division; current conservation; AMG vs. direct agreement. Independent of
  any legacy reference — these can't silently agree with a shared bug.
- **Legacy equivalence** (`test_legacy_equivalence.py`) — direct assembly
  vs. a parsed netgen-equivalent `.net`; uniform vs. all-size-1 multires;
  mixed-size multires vs. uniform in a homogeneous medium; `.vof` round-trip
  (including the large-mesh performance regression test for the vectorized
  `write_vof`).
- **Real lab model** (`test_real_lab_model.py`) — gated on
  `NEUROAM_REFDATA` pointing at the lab's own `Sph_80` reference files;
  assembly-vs-the-actual-lab-mesher's-`.mrm`, not just an internally
  consistent synthetic mesh.
- **I/O round-trips** (`test_io_and_coupling.py`, `test_morphology.py`,
  `test_neuron_results.py`) — every file format, plus the rotation/
  registration math (`rotation_between_vectors`, `place_aligned`,
  `estimate_depth_axis` against both a synthetic known-orientation cell and
  the real D1/A2i SWC files).
- **NEURON-gated** (`test_neuron_link.py`) — skipped whole-file if `neuron`
  isn't importable; hoc-templated cells are tested via subprocess isolation
  (see §6) so testing both D1 and A2i for real doesn't hit the one-cell-
  per-process constraint the code itself has to work around.
- **Electrodes/geometry/viz** (`test_electrodes.py`, `test_viewer.py`,
  `test_viz3d.py`) — rasterization vs. analytic volumes, rotation
  invariance, QC failure modes, and (for viz3d) that the actually-generated
  HTML contains the interactive machinery it's supposed to (trace indices,
  JS function names, the embedded data payload) — not just that a file of
  plausible size got written.

See [VALIDATION.md](VALIDATION.md) and [AM_VERIFICATION.md](AM_VERIFICATION.md)
for the numeric results these tests (and the example scripts) actually
produced against real data.
