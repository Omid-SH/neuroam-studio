# Electrode design and registration

*NeuroAM Studio v0.2 — `neuroam.geometry`, `neuroam.frames`, `neuroam.electrodes`*

## The problem this replaces

Until now a new electrode was created by editing the voxel array by hand:

```matlab
% New Project/Modify_model.m
model_new = model3D;
for i = 211:212
    for j = 196:201
        for k = 120:121
            model_new(j,k,i) = 102;
        end
    end
end
dlmwrite('RatCC_full_..._Ring_Res_166um.model', reshape(model_new, ny*nz, nx), ' ')
```

and then wiring it up by patching the `.in` file (`IN_editor_2.py`), which
picks the terminal node as *the middle element of `np.argwhere(model == label)`*.

Four things go wrong with that, and all four are things we have actually hit:

| Failure | Consequence |
|---|---|
| Index ranges are tied to one resolution and one anatomy | Nothing transfers from 166 µm to 83 µm to 41.5 µm, or from rat to rabbit to human |
| Axis-aligned loops cannot rotate | A ring on the cornea, a needle on an oblique trajectory, or a patch on a curved orbit cannot be expressed |
| The terminal node is chosen by scan order | In `Models/with retina/sample.in` the declared ground node `(418, 388, 484)` is **not in the 102 electrode at all** — the current returned through whatever tissue was at that node |
| Every montage is a full copy of the anatomy | 862 MB per montage at 83 µm, 6.9 GB at 41.5 µm; seven montages of the same rat head |

v0.2 addresses each: electrodes are millimetre geometry, placed in an
anatomical frame, connected through an explicit terminal model, stored as a
sparse overlay, and checked before they are ever solved.

---

## 1. Geometry — `neuroam.geometry`

Electrodes are **signed-distance regions in millimetres**, independent of the
grid they land on. They compose with `|` (union), `&` (intersection) and `-`
(difference), and they rotate.

```python
from neuroam.geometry import Torus, Cylinder, Needle, Grid, rasterize

ring   = Torus(ring_radius_mm=2.695, tube_radius_mm=0.12)
recess = ring.offset(0.05) - ring            # 50 µm insulating skin
probe  = Needle(radius_mm=0.075, length_mm=4.0, tip_length_mm=0.3)

idx, _, _ = rasterize(ring.place(origin_mm, axis), Grid.from_model(model))
```

Available primitives: `Sphere`, `Box`, `Cylinder`/`Disc`, `Capsule`, `Cone`,
`Torus`, `Annulus`, `SphericalShell`, `SphericalCap`, `SphericalBand`,
`Needle`, `Cuff`, `Polyline` (swept wire), plus `.offset()` and `.shell()`
on any of them, and `MaskRegion` to bring anatomy labels into an expression.

Rasterization is **bounding-box limited and supersampled** (3×3×3 sub-samples
per voxel by default): painting a 240 µm wire ring into a 912×900×504 model
evaluates a few hundred thousand sample points, not 414 million. Rasterized
volumes match analytic volumes to within a few percent for features several
voxels thick (`tests/test_electrodes.py`).

## 2. Registration — `neuroam.frames`

A `Frame` gives an electrode an **anatomical address** instead of an index.
For the eye it is fitted from the anatomy itself:

```python
from neuroam.frames import Frame
frame = Frame.eye(model, retina_labels=(77,))
# origin = least-squares centre of the retina shell
# +e3    = corneal axis (away from the retina centroid)
```

On the RatCC 83 µm crop this recovers centre `(89.00, 85.00, 91.00)` vox and
globe radius **3.155 mm with a 48 µm sphere-fit RMS**, and the corneal axis it
derives agrees with the axis of the montage's existing contact-lens ring to
**0.34°** — the frame is well determined by the anatomy alone.

Convenience shapes on the globe:

```python
frame.ring(r_mm=3.360, theta_deg=53.3, tube_radius_mm=0.12)   # wire ring on the surface
frame.cap(r_mm=3.360, half_angle_deg=30, thickness_mm=0.15)   # contact-lens dome
frame.band(3.360, 45, 60, 0.15)                               # wide curved ring
frame.place(sdf, r_mm=..., theta_deg=..., phi_deg=...)        # anything, anywhere
```

`theta_deg` slides an electrode from the corneal apex (0°) toward the equator
(90°) — which is the parameter a montage study actually wants to sweep, and
the one index loops cannot express.

For non-spherical surfaces there are surface operators:

```python
from neuroam.frames import surface_band, conform, snap_to_surface, surface_point, normal_at

conform(footprint, model, labels=[9], offset_mm=0.0, thickness_mm=0.2)
snap_to_surface(sdf, model, labels=[9], direction=frame.e3, standoff_mm=0.1)
```

`conform` drapes a footprint onto a thin shell hugging a tissue, so a patch
electrode follows curvature instead of cutting through it; `snap_to_surface`
bisects the translation until the electrode body clears the tissue by exactly
the requested standoff.

## 3. Terminal models — how the electrode connects

| `terminal` | Meaning | Use it when |
|---|---|---|
| `supernode` *(default)* | Every lattice node of the electrode body is merged into **one unknown** — an ideal equipotential metal contact. The solve returns the electrode potential, hence its access impedance. | Designing new electrodes; anything where the answer should not depend on which node the wire is soldered to |
| `node` | Legacy: current enters at a single node and the 10⁻⁷ Ω·m metal body does the equipotentializing | Reproducing archived runs bit-for-bit |
| `distributed` | The ampere is split over the electrode's nodes by weight, with no equipotential constraint | Independently driven contact arrays |

The supernode is implemented as a projection of the assembled system,
`G ← Pᵀ G P`, with `P` the node→electrode aggregation; ground electrodes
eliminate all of their nodes instead of one. Validation:

- **Slab between two full-face supernodes** reproduces `R = ρL/A` to a
  relative error of **1e-13**.
- **Disc on a half-space** converges to the spreading resistance `ρ/(4a)` from
  below as the grounded boundary recedes (0.91 of `ρ/4a` at a 32-voxel domain).
- **Supernode vs painted metal** agree to <2 % in the field away from the
  electrode, confirming the two descriptions are the same physics.

This also fixes a real modelling issue in the existing montages: the J ground
electrode is a single conductor, but the legacy `.in` ties **one node** of it
to 0 V. Under a supernode the whole electrode is the return path.

> **Connectivity note.** Electrode pieces are counted with 26-connectivity,
> because in the admittance formulation two voxels that share only an edge or
> a corner still share lattice nodes and are therefore one conductor. Counted
> with 6-connectivity the shipped contact-lens ring looks like 13 floating
> fragments; it is one.

## 4. Overlays and montages — `neuroam.electrodes`

`register()` paints electrodes into a **sparse overlay** (`(N,3)` indices +
labels, compressed `.npz`), so anatomy and montage are separate artefacts:

```python
from neuroam.electrodes import ElectrodeSpec, Montage

mont = Montage("CLring_JGND", [
    ElectrodeSpec("CL", region=frame.ring(3.360, 53.3, 0.12),
                  role="source", label=101, terminal="supernode",
                  waveform="Cur1", min_voxels=200),
    ElectrodeSpec("J", from_label=102, role="ground", label=102,
                  terminal="supernode"),
]).build(model)

print(mont.report())
mont.save("out/")                      # CLring_JGND.overlay.npz + .montage.json
mont.write_legacy(model, "out/")       # bake .model/.in for the lab mesher
```

One 83 µm anatomy plus *n* overlays of a few tens of kB replaces *n* × 862 MB
`.model` files. `from_label=` adopts an electrode that is already painted in
the anatomy, so existing montages enter the same machinery.

`ElectrodeSpec.overwrite` controls what the electrode is allowed to replace
(`"any"`, `"background"`, or an explicit label list), and `insulation=` paints
a second region as an insulating backing (label 110 by default).

## 5. Quality control

`register()` returns an `ElectrodeQC` per electrode, and the pipeline writes it
into `run_manifest.json`. It reports painted volume, exposed surface area,
connected pieces, **which tissues the electrode actually touches and over how
much area**, overlap with other electrodes, and whether the terminal node is
genuinely inside the body it claims to drive. `qc.charge_density(I, pw)` gives
µC/cm²/phase for a safety check.

Warnings that fire automatically:

- terminal node is not a node of this electrode *(the `sample.in` bug)*
- electrodes share voxels — they are shorted
- several electrically separate pieces, only one of which a `node` terminal drives
- fewer voxels painted than `min_voxels` — usually a units or placement error
- voxels blocked by the overwrite policy

Example, from the shipped RatCC montage re-registered through v0.2:

```
CL [source/supernode] label 101: 1,314 vox (0.7513 mm^3), area 18.55 mm^2, 1 component(s)
    bbox [48, 44, 58]..[96, 92, 124]  centroid [71.8, 67.8, 91.0]
    touches [66:1765, 13:397, 55:232, 9:186, 33:62, 12:41]  terminal (61, 80, 66) (2974 node(s), inside=True)
```

## 6. Config and CLI

```json
"model":  { "legacy_in": "...", "relabel": {"101": 66} },
"frames": { "eye": {"type": "eye", "retina_labels": [77]} },
"electrodes": [
  {"name": "CL", "role": "source", "label": 101, "terminal": "supernode",
   "waveform": "Cur1",
   "geometry": {"type": "torus", "ring_radius_mm": 2.695, "tube_radius_mm": 0.12,
                "frame": "eye", "at": {"axial_mm": 2.006}},
   "min_voxels": 200},
  {"name": "J", "role": "ground", "label": 102, "terminal": "supernode",
   "from_label": 102}
]
```

- `frames`: `eye` (fit the globe), `shell` (fit any shell-like label set),
  `axes` (explicit origin + axis).
- `geometry.at`: `{"axial_mm", "lateral_mm", "phi_deg", "roll_deg"}` or
  `{"r_mm", "theta_deg", "phi_deg"}` in the named frame; `{"origin_mm"/"origin_vox", "axis"}`
  without one.
- `geometry.snap`: `{"labels": [...], "standoff_mm": 0.1}` — slide onto the tissue.
- `geometry.conform`: `{"labels": [...], "offset_mm": 0, "thickness_mm": 0.2}` — drape.
- Composites: `{"type": "union"|"intersect"|"difference", "a": {...}, "b": {...}}`.
- `model.relabel` strips an existing electrode before a redesign.

```bash
python -m neuroam electrodes configs/ratcc_cl_ring.json          # register + QC, no solve
python -m neuroam electrodes configs/ratcc_cl_ring.json -o out/  # + overlay/montage files
python -m neuroam electrodes configs/ratcc_cl_ring.json --legacy out/   # + bake .model/.in
python -m neuroam run configs/ratcc_cl_ring.json                 # full solve
```

The `electrodes` command exits non-zero when any QC warning fires, so it can
gate a batch of montages in CI.

v0.1 configs still load unchanged: entries with `shape` keep using the old
voxel-index primitives.

## 7. Validation against the existing contact-lens ring

The montage's contact-lens electrode was reverse-engineered from
`RatCC_full_CLStim_JGND_912_900_504_Ring_Res_83um_with_retina.model`: it is a
**torus**, 360° with no gap, ring radius **2.700 ± 0.007 mm**, lying in a plane
**2.001 ± 0.008 mm** anterior of the globe centre — i.e. on the r = 3.36 mm eye
surface at **53.3°** from the corneal apex — with a wire radius of ≈ 0.12 mm.

`frame.ring(3.360, 53.3, 0.12)` reproduces it with:

| metric | value |
|---|---|
| voxels | 1,289 designed vs 1,144 existing |
| IoU | 0.51 |
| max surface separation | **1.41 voxels (117 µm)** |
| truth-only voxels within 1 voxel of the design | 96.3 % |

The modest IoU is a surface-to-volume artefact, not a placement error: the
ring is only ~3 voxels thick, so a ±½-voxel disagreement along its whole
surface costs most of the overlap. The two rings never separate by more than
1.4 voxels anywhere, and >94 % of the disagreement is a one-voxel skin.

## 8. Does it matter? Field results on the real montage

Solved on the RatCC 83 µm eye crop (6.44 M unknowns), retina = label 77:

| case | geometry | terminal | retina E_mean (V/m/A) | Z (Ω) | CG iters |
|---|---|---|---|---|---|
| A | shipped label 101 | legacy node | 1.8226e4 | 552.3 | 2395 |
| B | shipped label 101 | supernode | 1.8227e4 | 552.3 | 1632 |
| C | `frame.ring(3.360, 53.3°, 0.12)` | supernode | 1.8167e4 | 555.4 | 1628 |
| D | `frame.ring(3.360, 40.0°, 0.12)` | supernode | 1.7092e4 | 666.4 | 1624 |
| E | `frame.cap(3.360, 30°, 0.15)` | supernode | 1.6567e4 | 778.9 | 1630 |

Three things to take from this:

1. **The parametric ring is a faithful drop-in.** C differs from the
   hand-painted electrode by 0.3 % in retina E_mean and 0.6 % in impedance.
2. **The supernode is free accuracy and a 1.6× speedup.** B changes the answer
   by 1.5e-4 relative — the metal body was already equipotential — but removes
   the stiff 1e-7 Ω·m resistors from the system and cuts the solve from 588 s
   to 375 s.
3. **Electrode placement is worth sweeping.** 13° of ring position is 6 % of
   retina dose and 21 % of access impedance; the intuitive "bigger electrode"
   (E, 65 % more exposed area) is the *worst* of the three at delivering field
   to the retina.

Details in `docs/VALIDATION.md`.

## 9. Limits

- Rasterization is binary at a 0.5 occupancy threshold; there is no
  partial-volume material blending yet (`rasterize(..., fraction=True)`
  exposes the fractions if you want to experiment).
- `conform` and `snap_to_surface` build a full-volume distance transform, so
  on a 400 M-voxel model they want the cropped region, not the whole head.
- Voltage-controlled (Dirichlet) electrodes are not implemented; only current
  injection. The supernode machinery is what a voltage electrode will be built
  on.
- CAD/STL import is not included; `Polyline` plus CSG covers wire and shank
  geometries, and an STL voxelizer is a contrib-level addition.

---

## 10. Addendum (v0.2.1): arcs, surface addresses, portable montages

Added while porting the TES-GPS and VIRON clinical electrodes onto RatCC —
see [`CLINICAL_ELECTRODES.md`](CLINICAL_ELECTRODES.md).

**`Frame.arc`** — the open-arc counterpart of `Frame.ring`. A thread electrode
(DTL / OkuEl class) contacts an *arc* of the limbus, not a closed loop, and a
clinical spec quotes contact length in mm:

```python
frame.arc(r_mm=3.46, theta_deg=74.0, tube_radius_mm=0.10,
          phi0_deg=225.0, span_deg=100.0)          # 5.80 mm of thread
frame.arc_length_mm(3.46, 74.0, 100.0)             # -> 5.80
```

In a config: `{"type": "arc", "frame": "eye", "r_mm": …, "theta_deg": …,
"tube_radius_mm": …, "phi0_deg": …, "span_deg": …}`.

**`at.on_surface`** — an anatomical address for a *surface* contact. Instead of
a fixed `r_mm`, march out from the frame origin along (θ, φ), land on the named
tissue, and orient the body along that tissue's outward normal:

```json
{"type": "cylinder", "radius_mm": 1.3, "height_mm": 1.0, "frame": "eye",
 "at": {"theta_deg": 85, "phi_deg": 45,
        "on_surface": {"labels": [13], "standoff_mm": 0.0, "max_mm": 14.0}},
 "conform": {"labels": [13], "offset_mm": -0.10, "thickness_mm": 0.30}}
```

A fixed radius does not survive a change of model — skin sits at a different
distance from the globe centre in rat, rabbit and human, and at a different
distance at each azimuth in *one* rat. `on_surface` is what makes a periorbital
montage portable.

**`skip_if_missing: true`** — an electrode with no landing site in *this* model
(a crop that stops short, a species whose orbit is shaped differently) is
dropped with a `RuntimeWarning` instead of failing the run:

```
RuntimeWarning: electrode 'E4' skipped: no [13] tissue along theta=85.0
phi=225.0 in frame 'eye' — the model may be cropped there
```

One montage file then survives being carried across models, and the run tells
you exactly which contacts the model could not host.

**Measured limbus.** The RatCC cornea/sclera crossover on the outer shell is at
**θ = 74–75°** — a 6.5 mm cornea on a 6.3 mm globe. The shipped contact-lens
ring at θ = 53.3° is therefore ~20° *inside* the limbus, on the mid-cornea, not
at it.

**Eye-frame azimuth convention** (RatCC, derived from tissue profiles; confirm
on the full head model): φ = 45° superior, 135° nasal/rostral, 225° inferior,
315° temporal/caudal. See `CLINICAL_ELECTRODES.md` §4.

## 11. Fitting existing electrodes back into specs

A lab montage usually exists first as **hand-painted voxels**, not as a spec.
Turning it into one is what makes it reusable across resolutions and species,
and is how the numbers in [`AM_VERIFICATION.md`](AM_VERIFICATION.md) were
produced.

```python
from neuroam.electrodes import fit_spherical_band, fit_tube_path, refine_by_dice, dice, compare_masks
from neuroam.geometry import Grid, rasterize

grid = Grid.from_model(model)
cl_idx = np.argwhere(model.labels == 101)
region, info = fit_spherical_band(cl_idx, grid, axis_hint=frame.e3)
# info: {center_mm, r_in_mm, r_out_mm, axis, theta_min_deg, theta_max_deg, ...}

j_idx = np.argwhere(model.labels == 102)
lead, info = fit_tube_path(j_idx, grid)             # handles a curved/hooked lead
```

`fit_spherical_band` least-squares fits a sphere to the voxel centres and
measures the polar-angle band about `axis_hint` (fix this to an anatomical
axis for a ring; without it, only a cap's axis is well defined).
`fit_tube_path` builds a 26-connectivity graph over the voxels, walks the two
geodesic extremes, and takes the centroid of each geodesic-distance shell as a
centre-line node — this tracks a curved or hooked lead that no single
cylinder or straight-line fit would.

Both return a placed `Region` ready to drop straight into an `ElectrodeSpec`,
plus a plain-dict `info` for inspection or JSON export. `refine_by_dice`
coordinate-descends a parameter dict to tighten a fit against the original
voxels:

```python
def build(p):
    return SphericalBand(0.5 * (p["r_in"] + p["r_out"]), p["r_out"] - p["r_in"],
                         p["theta_max"], p["theta_min"]).place(p_center, axis)

best, best_dice = refine_by_dice(reference_mask, build, info, sweep={
    "r_in": [-0.01, 0, 0.01], "r_out": [-0.01, 0, 0.01]}, grid=grid)
```

`dice(a, b)` and `compare_masks(a, b, name)` (Dice, Jaccard, missed/extra
voxel counts) are the same metrics used to score a fit or to check a
regenerated electrode against the reference it was fitted from.

**Authoring lattice.** Lab electrodes are frequently hand-drawn on a coarser
lattice than the model — e.g. every voxel decision made at 166 µm inside an
83 µm model, so each 2×2×2 block is uniformly filled. Reproducing a montage
drawn this way needs `rasterize(..., lattice=2)` (equivalently
`ElectrodeSpec(..., lattice=2)`): the region is evaluated on a lattice that
many times coarser and the result expanded back, rather than supersampled at
full resolution — matching this authoring grid is worth a large jump in Dice
against the original voxels (see `AM_VERIFICATION.md`). Because regions are
mm-native, no parameter rescaling is needed; only the evaluation grid changes.
