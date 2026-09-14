# Merge notes — Claude AM/visualization update (2026-09-02)

**Nothing of yours was overwritten.** The update tarball was extracted with
`tar` refusing to clobber existing files; six files were new and went straight
into the tree, one of which needs a decision. This folder holds the pieces that
collide with your own work so you can merge them deliberately.

## You already have a parallel implementation

Your tree contains modules I did not write and did not touch:

| file | what it does |
|---|---|
| `neuroam/geometry.py` | SDF regions in **millimetres**, composable with `\| & -` — resolution- and species-independent |
| `neuroam/frames.py` | anatomical frames (globe centre from a sphere fit to the retina, corneal axis) so a spec is addressed anatomically, not by voxel index |
| `neuroam/electrodes.py` | `Terminal` / `ElectrodeSpec` / `ElectrodeQC` / `Overlay` / `Montage`, `register()`, `region_from_spec()`, `specs_from_config()` |
| `neuroam/viewer.py` | interactive **PyVista** viewer |

That design is better than mine for the cross-species goal — physical units and
anatomical addressing are exactly what makes a montage transfer between rat,
rabbit and human. Keep it as the trunk.

## What landed in the tree (safe, additive)

* `neuroam/viz3d.py` — plotly **HTML** scene builder. Complements `viewer.py`
  rather than competing: PyVista is the native interactive tool on a machine
  with VTK; viz3d writes a self-contained `.html` that opens anywhere (no VTK,
  shareable with collaborators, works over remote/headless). Needs
  `plotly` + `scikit-image`.
* `examples/06_view_model_and_field.py`, `examples/07_verification_figure.py`
  — use `viz3d` only, no electrode API.
* `tests/test_viz3d.py`, `docs/AM_VERIFICATION.md`.

## What is parked here (needs your decision)

* `electrodes_claude.py` — my electrode module. **Do not drop it over your
  `electrodes.py`.** Two things in it are worth transplanting into your design:

  1. **Fitting existing hand-built electrodes back into specs.**
     `fit_spherical_band()`, `fit_tube_path()` (geodesic-centroid centre-line,
     handles the J hook), `refine_by_dice()`, `compare_masks()`. This is what
     turns your existing ModelTool montages into reusable parametric specs —
     and it is how the verification below was measured.
  2. **Grid-aware rasterization.** `rasterize(..., grid=2)`. The lab's
     electrodes were drawn on a **166 µm lattice** inside the 83 µm model
     (every 2×2×2 block is 100 % filled). Matching that lattice is worth
     ~0.45 Dice — without it a generated electrode cannot reproduce a
     hand-built one. Worth adding as a rasterization option to `geometry.py`.

* `test_electrodes_claude.py` — tests for the above (your own
  `tests/test_electrodes.py` was left untouched).
* `examples/05_electrode_placement_verification.py` was moved here as
  `05_electrode_placement_verification_claudeAPI.py`: it imports my
  `ElectrodeSpec(shape=..., params=..., grid=...)` / `place()` API and will not
  run against your `electrodes.py`. Port it to `register()` +
  `specs_from_config()` and it becomes a permanent regression test for
  placement.

## Verification result it produced (worth keeping either way)

Rat eye, 83 µm crop, CL-stim / J-ground:

* raw model + reference electrode voxels == the hand-built model **bit-for-bit**
* electrodes regenerated from fitted specs: Dice **0.893** (CL ring),
  **0.975** (J lead), **0.952** (insulation); only **736 of 6.34 M voxels**
  differ in material
* retinal dose agrees to **+3.4 %** on mean |J|, r = **0.948**
* the reference CL electrode is a **ring** (spherical band 91.3°–96.5° from the
  corneal axis), not a disc; the J lead is a square-section hooked needle with
  insulation over the shaft only

Full write-up: `docs/AM_VERIFICATION.md`.

## Also worth taking

`neuroam/solver.py` in my copy adds an **algebraic-multigrid preconditioner**
(`solve(..., precond="amg")`, needs `pyamg`). On the 6.44 M-unknown rat eye
model it went from **2 395 CG iterations (378 s) to 47 (~150 s)**. Your
`solver.py` was modified on your side so I did not touch it — the change is
small and self-contained (a new branch in `make_preconditioner`).
