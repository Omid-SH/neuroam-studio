# AM verification — electrode placement & visualization (rat eye, 83 µm)

Answers the question "if NeuroAM places the SCL and J electrodes itself on the
raw rat head model, does it reproduce the lab's hand-built SCL–J model?"

Everything below is produced by
`_claude_am_update/05_electrode_placement_verification_claudeAPI.py` and
`examples/21_verification_figure.py`; raw numbers live in
`samples/ratcc_eye_83um/verification/verification_report.json`.

The verification script currently targets a standalone electrode API
(`_claude_am_update/electrodes_claude.py`) that predates `neuroam.electrodes`.
Its fitting logic now lives in the trunk module as `fit_spherical_band()`,
`fit_tube_path()`, `refine_by_dice()` and `compare_masks()`, and
`neuroam.geometry.rasterize(..., lattice=2)` reproduces the authoring-lattice
match described below — see [ELECTRODES.md](ELECTRODES.md#11-fitting-existing-electrodes-back-into-specs).
Porting the script itself to `register()` + `specs_from_config()`, so this
verification becomes a runnable regression test, is still open
(`_claude_am_update/MERGE_NOTES.md`).

## Setup

* Model: 180×160×220 crop at native 83 µm of the lab's with-retina rat head
  (crop offset (230, 290, 150) in the 912×900×504 full model), containing the
  whole eye, retina (label 77), CL ring stimulator (101), J return (102) and
  its insulation (110).
* Raw arm: the same crop of `RatCC_full_912_900_504_Res_83um_with_retina.model`
  — verified to contain **zero** voxels of labels 101/102/110.
* All arms solved for a unit-current field, CG + algebraic-multigrid
  preconditioning, rtol 1e-8, with the **same** injection and return nodes so
  the comparison is about geometry rather than node choice.

## What the reference electrodes actually are

Recovered by differencing the hand-built model against the raw one (6 968
voxels differ, all of them electrode):

| label | what | geometry |
|---|---|---|
| 101 | contact-lens **ring** stimulator | spherical band, centre (300.9, 356.9, 241.4), r 31.6–33.5 vox, 91.3°–96.5° from the corneal axis — an annulus at the limbus, **not** a disc |
| 102 | **J** return electrode | square-section lead, half-width 1.4 vox, running z 343 → 244 then hooking out in (−x, +y); 13-node centre-line |
| 110 | insulation | square sheath r 2.0–3.5 vox over the **shaft only** (z 270–349), leaving the hook exposed |

All three are **2×2×2-block aligned**: every coarse cell is 100 % filled, i.e.
the electrodes were drawn on a 166 µm lattice inside the 83 µm model. NeuroAM
reproduces this with `rasterize(..., grid=2)`; ignoring it costs ~0.45 Dice.

## Results

**Arm B — exact replay.** Raw model + the reference electrode voxels stamped
back in is **bit-identical** to the hand-built model (`replay_exact: true`).
Since the assembled matrix is a pure function of labels, materials and dx, an
identical field follows; this confirms the raw model is the same base and that
nothing else differs between the two files.

**Arm C — parametric placement.** Electrodes generated from the fitted spec:

| electrode | Dice | generated / reference voxels |
|---|---|---|
| CL ring (101) | **0.893** | 1 256 / 1 144 |
| J lead (102) | **0.975** | 1 920 / 1 888 |
| insulation (110) | **0.952** | 4 128 / 3 936 |

Only **736 of 6 336 000 voxels (0.012 %)** end up with a different material.
Over the voxels that are the same material in both arms:

| region | Δ mean \|J\| | Δ E p95 | rel-L2 | Pearson r | median per-voxel |
|---|---|---|---|---|---|
| retina | **+3.4 %** | +6.5 % | 0.132 | **0.948** | 4.5 % |
| vitreous | −2.1 % | −6.5 % | 0.166 | 0.954 | 3.8 % |
| eye (retina+vitreous) | +2.2 % | −5.6 % | 0.132 | 0.990 | 3.9 % |
| all tissue | +0.2 % | −4.0 % | 0.229 | 0.976 | 10.7 % |

Retinal dose per 1 A: **1.823e4 V/m (hand-built) vs 1.884e4 V/m (generated)**.

> Comparing over *all* voxels of a region instead is misleading: a voxel that
> is vitreous in one arm and electrode metal in the other differs in |J| by
> ~10⁷ (ρ 1e-7 vs 3.05 Ω·m), which alone drives the raw vitreous rel-L2 to 17.
> The tissue-matched numbers above are the meaningful ones; both are recorded.

## Verdict

* The **AM pipeline is verified exactly** — same voxels in, same system, same
  field (arm B).
* **Parametric placement reproduces the hand-built montage to ~3 % on
  integrated retinal dose** and r ≈ 0.95 per voxel, with residual differences
  confined to a thin band next to the electrode where the two rasterizations
  disagree. For reproducing an *existing* montage exactly, use the reference
  voxels (arm B / `place()` with an explicit mask); for **new** placements and
  cross-species transfer, the parametric path is the one to use, and this is
  its measured fidelity.
* Useful by-product: a ~10 % difference in electrode geometry moves mean
  retinal dose by ~3 %, i.e. the answer is not hypersensitive to electrode
  discretization — worth stating in the methods of any montage-comparison
  study.

## Solver note

Algebraic multigrid (`precond="amg"`) cut this 6.44 M-unknown solve from
**2 395 CG iterations (378 s) to 47 iterations (~150 s)**. It is now available
as `solve(..., precond="amg")` and is the recommended setting for models of
this size.

## Visualization

`examples/20_view_model_and_field.py` writes three self-contained HTML views
into `samples/ratcc_eye_83um/verification/views/`:

1. `01_model_and_electrodes.html` — anatomy + electrodes before simulation
2. `02_current_density.html` — |J| painted on each structure, plus isosurfaces
   and cut-planes
3. `03_retina_only.html` — retina alone, coloured by |J|, hot-spot cloud

Every legend entry is a toggle: click "head tissue" off to get the eye, then
turn off vitreous/cornea to get the retina alone. `verification_figure.png`
is the static summary of the comparison above.
