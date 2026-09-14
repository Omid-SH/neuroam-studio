# Clinical electrodes: TES-GPS and VIRON on the rat head model

*NeuroAM Studio — extracted 2026-09-01. Machine-readable parameter record:
`configs/clinical_electrode_parameters.json`. Rat instantiations:
`configs/ratcc_tes_gps.json`, `configs/ratcc_viron_periorbital.json`,
`configs/ratcc_viron_orbital6.json`. Script: `examples/07_clinical_analogues.py`.*

Two randomised sham-controlled trials are running right now in glaucomatous
optic neuropathy, both German, both delivering a few hundred microamps, and
they disagree about the most basic question in the field: **where do you put
the electrode.** TES-GPS lays a silver thread on the eye. VIRON never touches
the eye at all. This note extracts what each protocol actually specifies,
says what each one leaves out, and records how both were scaled onto the
RatCC anatomy so the two can be compared in the same solver.

---

## 1. What the protocols say

### TES-GPS — transcorneal, contact with the globe

Lorenz K, Schuster A, Michel HM, Ruckes C, Kronfeld K, Schippert R, Stett A,
Beck A. *BMJ Open* 2026;**16**:e112879 (doi 10.1136/bmjopen-2025-112879),
NCT06682962. Single centre (Mainz), n = 50, randomised 1:1, double-masked,
18 months, sponsor Okuvision.

| | |
|---|---|
| System | **OkuStim 2**, Okuvision GmbH (Reutlingen), class IIa, battery-powered, worn like glasses |
| Active electrode | **OkuEl M** — an arched clip carrying fine threads; the conductive element is a **silver thread**, single-use |
| Active placement | **On the ocular surface at the inferior limbus**, resting just above the lower eyelid, *free of pressure* at the conjunctiva |
| Contact length | **≥ 10 mm** of thread against the eye (from the instructions for use) |
| Counter electrode | **OkuEl Counter Electrodes** — adhesive skin pads on the **forehead, between eyebrow and hairline**. Two pads, one electrical terminal |
| Waveform | Symmetric **biphasic rectangular** current pulses, **5 ms positive / 5 ms negative**, **20 Hz** |
| Amplitude | **up to 1 mA**, set per patient *just below the individual tolerance threshold*; adjustable up during the four training visits, down thereafter |
| Dose | 30 min, **once weekly**, 18 months; at home after in-clinic training |
| Sham | Identical set-up; a brief subthreshold current for the first 30 s, then **0 µA** |

Not stated anywhere: **the thread diameter**. Neither the protocol nor the
instructions for use quote one. DTL-class fibres run 0.05–0.25 mm; the
instructions do warn that current density at the active electrode "can exceed
2 mA/cm²", which at 1 mA implies an exposed area below ~0.5 cm² — consistent
with a thread but not a constraint that pins the diameter.

Note a discrepancy worth knowing about: the older OkuStim instructions for
use place the counter electrode **at the temple**; the TES-GPS protocol
specifies the **forehead**. The protocol is the authority for this trial, but
any comparison with the published retinitis-pigmentosa OkuStim literature is
comparing two different return paths.

### VIRON — transorbital, nothing touches the eye

Schittkowski M, Pohlner J, Mercieca K, Grohmann C, Kröger L, Prokosch V,
Lorenz K, Beck A, Haueisen J, Hunold A, et al. *BMJ Open* 2025;**15**:e091705
(doi 10.1136/bmjopen-2024-091705), DRKS00029129. Multicentre (Göttingen,
Bonn, Hamburg, Cologne, Mainz), n = 300, three arms, double-blind.

| | |
|---|---|
| Device | **DC-Stimulator PLUS**, neuroConn GmbH (Ilmenau), model 0021, class IIa |
| Electrodes | **Two**, on **periorbital skin**. Group 1 ("classical"): fixed **at the temples**, always the same positions. Group 2: positions **individually optimised** from an MRI-based FEM |
| Eyes | **Open**, straight-ahead fixation (group 1 and sham); optimised gaze direction in group 2 |
| Waveform | **Sinusoidal** alternating current, **10 Hz** |
| Amplitude | **600 µA peak-to-peak** (= 300 µA amplitude) |
| Dose | 25 min/day, 10 sessions: 2 × 5 consecutive days with a weekend break |
| Sham | 200 µA p-p, 40 Hz, 60 s |

**VIRON states no electrode size and no electrode material.** That is the
single biggest gap for anyone trying to model it. Two companion papers by the
same authors supply the geometry the trial is built on:

- **Gall C, Schmidt S, Schittkowski MP, et al.** *PLoS ONE* 2016;**11**:e0156134
  — the trial VIRON cites for its "classical" montage. **Four 10 mm Grass gold
  cup electrodes (SAFELEAD), two per eye, on the skin near the orbital cavity**,
  used **one at a time**; return a **32 × 30 mm stainless steel plate on the
  right arm**; biphasic square pulses in bursts, 8–25 Hz, at 125 % of the 5 Hz
  phosphene threshold. Their own FEM modelled a single circular electrode
  **10 mm diameter × 3 mm high, σ = 1.5 S/m, above the right eyebrow**, return
  at the neck, ±0.5 mA.
- **Hunold A, Ortega D, Freitag S, Link D, Antal A, Klee S, Haueisen J.**
  *Life* 2025;**15**:820 — **six 10 mm Ag/AgCl cups (E1–E6) around the right
  orbit** with Ten20 paste, counter at the **temple (C1)** or **vertex (C2)**;
  an alternative montage uses a **30/75 mm ID/OD ring** around the eye against
  a **10 × 10 cm occipital** electrode. Phosphene thresholds 122 ± 60 µA
  (light-adapted) vs 329 ± 110 µA (dark-adapted).

That last paper carries the result that most directly justifies what the
electrode module is for: **moving the electrode shifted the phosphene hot spot
by 56° (temple return) / 39° (vertex return) of polar distance, while changing
gaze direction shifted it by only 25° / 18°.** Electrode placement steers the
retinal target more than the eye does.

Note also that VIRON group 1 says *two* electrodes while the montage it cites
uses *four, one at a time* against a distant arm return. These are different
circuits. The protocol text is what is modelled here.

---

## 2. Side by side

| | **TES-GPS (OkuEl M)** | **VIRON (rtACS)** |
|---|---|---|
| Contacts the eye | **yes** — thread on the ocular surface | **no** — skin only |
| Active geometry | thread / line contact, ≥ 10 mm long | disc / patch, 10 mm diameter (inherited) |
| Active material | silver | gold cup (Gall) / Ag/AgCl + paste (Hunold) |
| Active site | inferior limbus | periorbital skin; temples (group 1) |
| Return | 2 adhesive pads, forehead | second periorbital electrode; arm plate in the cited work |
| Electrode separation | ~cm, across the orbit | ~cm to tens of cm |
| Waveform | biphasic rectangular pulses | sinusoid |
| Timing | 5 ms/phase, 20 Hz | 10 Hz continuous |
| Amplitude | ≤ 1 mA, sub-tolerance | 600 µA p-p, fixed |
| Dose | 30 min weekly × 18 months | 25 min daily × 10 days |
| Individualisation | amplitude only (tolerance threshold) | **electrode position** (MRI + SimNIBS FEM) |
| Endpoint window | 18 months | immediately post + 24 weeks |

The physical contrast is the point. The OkuEl thread injects current **inside**
the high-resistance skin barrier, directly into the tear film and cornea, so
almost all of it crosses the globe. The VIRON electrodes inject **outside** it,
so most of the current shunts through scalp and soft tissue and only a fraction
reaches the retina — which is exactly why VIRON needs per-patient current-flow
modelling and TES-GPS does not. VIRON's own proof of principle quantifies the
size of the prize: an optimised montage reached **0.18 ± 0.02 A/m²** in the
target region versus **0.09 ± 0.01 A/m²** for the standard montage — a factor
of two, purely from where the electrodes sit.

VIRON's tissue conductivities (S/m), for reference when comparing against our
`.in` material tables: skin 0.456, blood 0.6, spongy bone 0.025, compact bone
0.008, CSF 1.654, grey matter 0.275, white matter incl. optic nerve head
0.126, eye muscles 0.16, cornea 0.5, sclera 0.56, retina 0.7, vitreous 1.55,
aqueous 1.8, lens 0.32.

---

## 3. Scaling onto the rat

Human electrodes cannot be placed at their literal size on a 6 mm eye. Every
dimension below is scaled **geometrically by globe radius**:

> k = R_rat / R_human = **3.155 mm / 12.0 mm = 0.263**

3.155 mm is not a literature value — it is the least-squares sphere fit to the
RatCC retina shell (label 77), RMS 48 µm, the same fit `Frame.eye` performs.
Lengths scale by k, areas by k² = 0.069. **Currents are not scaled**: the
solver returns fields per ampere and the waveform amplitude is a multiplier,
so the clinical amplitudes are kept in the configs and any dose convention
(match total current, match charge density, match retinal field) can be applied
afterwards.

| quantity | clinical | × k | modelled | why the difference |
|---|---|---|---|---|
| OkuEl thread diameter | ~0.15 mm (DTL class) | 0.04 mm | **0.20 mm** | 0.04 mm is half a voxel at 83 µm. 0.20 mm is the thinnest wire this grid resolves as a connected conductor; a true scaled fibre needs the 41.5 µm model |
| OkuEl contact length | ≥ 10 mm ≈ 100° of the limbal circle | — | **5.8 mm = 100° arc** | scaled as an *angle*, which is the invariant; 5.8 mm on the rat limbus |
| OkuEl counter pad | ~2 × 2 cm² adhesive | 0.28 cm² | **r = 2.8 mm disc (0.25 cm²)** | the headset's two pads are one terminal, so one pad of the same total area |
| rtACS cup | 10 mm diameter | 2.6 mm | **r = 1.3 mm disc** | direct |
| rtACS return plate | 32 × 30 mm | 8.4 × 7.9 mm | **r = 2.6 mm disc** | crop-limited, see §5 |
| periorbital radius | orbital rim ~20–25 mm from globe centre | 5.3–6.6 mm | **skin lands at 4.3–7.2 mm** | measured, not assumed — the rat orbit falls in the scaled band |

### The limbus is measured, not assumed

The rat cornea is enormous relative to the globe, so a human limbus angle
would be badly wrong. Binning the outer shell (r ∈ [3.15, 3.50] mm) of cornea
(66) against sclera (33) by polar angle gives the crossover at

> **limbus θ = 74–75°** from the corneal apex → corneal chord radius
> 3.36 · sin 74° = **3.23 mm**, i.e. a **6.5 mm cornea** on a 6.3 mm globe.

That is a real result for the anatomy and it relocates the electrode: **the
lab's existing contact-lens ring sits at θ = 53.3°, roughly 20° inside the
limbus, on the mid-cornea.** The OkuEl analogue at θ = 74° is a materially
different placement, not a variant of the same one — and §8 of
`docs/ELECTRODES.md` already showed that 13° of ring position is worth 6 % of
retinal dose and 21 % of access impedance.

---

## 4. The azimuth convention (new)

The eye frame fixes the polar axis (the corneal axis) but not the azimuth, and
"inferior limbus" and "supraorbital" are azimuth statements. On RatCC the
fitted frame comes out

```
e1 = ( 0.5,   -0.5,    0.7071)
e2 = (-0.5,    0.5,    0.7071)
e3 = (-0.7071, -0.7071, 0.0   )     <- corneal axis
```

so **φ = 45° is exactly +z** and **φ = 225° is −z**. Radial tissue profiles
from the globe centre outward settle the anatomy:

| φ | what the ray passes through | reading |
|---|---|---|
| 45° | muscle to 5.9 mm, ~1 mm of skin (13), then air | **superior / dorsal** |
| 225° | sclera, then muscle and fat past 7.6 mm, **no skin** | **inferior / ventral** — into the masseter |
| 135° | muscle, skin at 5.1–6.6 mm, then air | **nasal / rostral** — toward the snout |
| 315° | muscle, fat, bone; no skin exit inside the crop | **temporal / caudal** — toward the midline and braincase |

Corroboration: the optic nerve (label 24) centroid sits at θ = 171°, φ = 323° —
essentially at the posterior pole, leaning the way a rat optic nerve leans,
caudo-medially toward the chiasm. The palpebral/masseter asymmetry and the
nerve tilt agree.

**This is derived from the crop and should be confirmed on the full
912 × 900 × 504 head model before any of it goes in a paper.** It is recorded
in each config's `_orientation` field and as constants in
`examples/07_clinical_analogues.py`, so changing it is a one-line edit and
every montage moves with it.

---

## 5. What was built, and what registered

Three montages, all on `samples/ratcc_eye_83um/RatCC_eye_180_160_220_83um_with_retina.in`
(180 × 160 × 220 at 83 µm, 6.44 M nodes). Each config first strips the lab's
own hardware (`relabel`: 101 → cornea, 102 → fat, 110 → muscle) so the clinical
electrode is placed on clean anatomy, and every contact is a `supernode`
terminal.

**`ratcc_tes_gps.json` — TESGPS_OkuEl**

| electrode | geometry | voxels | area | components | contacts |
|---|---|---|---|---|---|
| `OkuEl` (source) | arc, r 3.46 mm, θ 74°, φ₀ 225°, span 100°, wire r 0.10 mm | 309 (0.177 mm³) | 5.47 mm² | 1 | sclera 421, **cornea 305**, muscle 28 |
| `CTR` (ground) | skin pad r 2.8 mm, θ 75°, φ 45° | 6 534 (3.74 mm³) | 47.3 mm² | 1 | **skin 2 914**, air 2 570, muscle 1 386 |

**`ratcc_viron_periorbital.json` — VIRON_periorbital**

| electrode | geometry | voxels | area | components | contacts |
|---|---|---|---|---|---|
| `E_sup` (source) | skin pad r 1.3 mm, θ 85°, φ 45° | 2 137 (1.22 mm³) | 13.3 mm² | 1 | skin 695, air 1 118 |
| `RET` (ground) | skin pad r 2.6 mm, θ 85°, φ 165° | 9 750 (5.58 mm³) | 75.2 mm² | 1 | **skin 5 408**, air 3 786 |

**`ratcc_viron_orbital6.json` — VIRON_orbital6** — the six-cup Hunold array,
each cup an **independent source**, so one run produces six basis fields and
any montage (including an optimised one) is a linear combination of them. This
is the montage space VIRON group 2 searches with SimNIBS.

| electrode | φ | voxels | area | components | skin contact |
|---|---|---|---|---|---|
| `E1` superior | 45° | 2 137 | 13.3 mm² | 1 | 695 |
| `E2` supero-nasal | 105° | 3 255 | 21.3 mm² | 1 | 1 432 |
| `E3` naso-inferior | 165° | 2 960 | 22.9 mm² | 1 | 1 710 |
| `E4` inferior | 225° | — | — | — | **skipped: no skin in the crop** |
| `E5` infero-temporal | 285° | — | — | — | **skipped: no skin in the crop** |
| `E6` supero-temporal | 345° | 2 280 | 18.0 mm² | 1 | 1 068 |
| `C1` counter | 15° | 5 745 | 42.2 mm² | 1 | 2 482 |

No QC warnings fire on any of the three: every contact is a single 26-connected
conductor, every terminal node is inside its own electrode, and no two
electrodes share a voxel. Overlays are in
`samples/ratcc_eye_83um/electrode_designs/` as
`{TESGPS_OkuEl, VIRON_periorbital, VIRON_orbital6}.overlay.npz` + `.montage.json`,
a few tens of kB each — the anatomy is not duplicated.

### Honest limits of this registration

1. **The crop cannot host the returns.** The eye crop is 15 × 13 × 18 mm. It
   carries periorbital skin only for φ ≈ 330°→225° through 0° and 180°; the
   inferior/temporal sector runs out of it, which is why E4 and E5 are skipped.
   It contains **no forehead, no temple, no trunk at all.** Every contact
   labelled `CROP STAND-IN` is placed where the crop allows, not where the
   protocol puts it. `C1` in particular sits only 30° from `E1`, which makes
   E1's access impedance unrepresentative. **Access impedances and absolute
   retinal dose from this crop are not publishable numbers** — the *relative*
   comparison between electrode geometries at the eye is.
2. **The thread is voxel-limited.** 0.20 mm diameter is 2.4 voxels at 83 µm.
   The clinically scaled 0.04 mm is out of reach until the 41.5 µm model.
3. **The azimuth convention is inferred** (§4), not read off a labelled atlas.
4. Surface pads occupy the outer ~0.1 mm of skin as well as the air above it
   (`overwrite: [0, 13]`), which is the modelling stand-in for conductive
   paste bridging the stratum corneum. It is a choice, and it lowers electrode
   impedance relative to a pad that only rests on top.

---

## 6. Module changes this required

Three small additions to v0.2, all tested (`tests/test_electrodes.py`):

- **`Frame.arc(r, theta, tube, phi0, span)`** and `Frame.arc_length_mm(...)` —
  the open-arc counterpart of `Frame.ring`. A thread electrode contacts an
  *arc* of the limbus, not a closed loop, and contact length in mm is what a
  clinical spec quotes. Exposed to configs as `{"type": "arc", ...}`.
- **`at.on_surface: {"labels": [...], "standoff_mm": ...}`** — an anatomical
  address for a *surface* contact: march out from the frame origin along
  (θ, φ), land on the named tissue, orient along its outward normal. Unlike a
  fixed `r_mm` this carries across resolutions and species, where the skin sits
  at a different distance from the globe centre. This is what makes a
  periorbital montage portable rat → rabbit → human.
- **`skip_if_missing: true`** on an electrode entry — a contact with no landing
  site in *this* model (a crop that stops short, a species whose orbit is
  shaped differently) is dropped with a `RuntimeWarning` instead of failing the
  run, so one montage file survives being carried across models.

## 7. Next

1. **Re-register on the full 912 × 900 × 504 head model.** This is the blocking
   item: it fixes the returns, restores E4/E5, and confirms §4. Registration
   is cheap; a solve wants a workstation or CARC (~50–60 GB).
2. Solve all three and compare on one axis — retinal E_mean per ampere and per
   *delivered* milliamp — against cases A–E in `docs/VALIDATION.md`. The
   quantitative form of "does touching the eye matter" is one table.
3. Use the six basis fields from `VIRON_orbital6` to reproduce VIRON's own
   optimisation on the rat: maximise current density over a target retinal
   patch and see whether the 2× the trial reports for a human head survives at
   rat scale, where the electrodes are proportionally much closer together.
4. Carry the same three configs to the rabbit and human models for Paper 3.
   The clinical numbers live in `configs/clinical_electrode_parameters.json`
   unscaled, so only `k` changes.
