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

## 7. Done: full-model registration, solve, and comparison (v0.5)

Items 1–2 below are complete. Script: `examples/38_solve_clinical_configs.py`
(registration + solve, one process per config) and
`examples/39_clinical_config_comparison.py` (the comparison against the
original 6). A fourth literature design was added along the way, from a
paper specifically about the eye-closed case (Zhou et al. 2026, *iScience*):

**TpES/TcES — transpalpebral (eyelid) vs. transcorneal, the eye-closed
question directly.** TpES: four 5 mm human discs on eyelid skin (only the
superior channel, CH1, is modelled here — see §7.4); scaled to a 1.0 mm
rat-scale radius (up from the pure-geometric 0.66 mm — see the script's own
comment for why: the smaller radius doesn't reliably land a supernode
terminal on the reused mesh). TcES: their own transcorneal control, a
**flat annular disc** (9.5 mm OD / 8 mm ID, not a wire loop) on the corneal
surface, `Frame.band(3.36, 18.2, 21.8, 0.10)` at rat scale. Both return at a
skin site standing in for their neck electrode (this head-only model has no
literal neck — same honest substitution §5 already makes for other
returns). Biphasic rectangular pulses, 20 Hz cathodic-first, 10 ms/phase;
the paper's own reference retinal field (~2–2.3 V/m at 1 mA, human head
model) is the external sanity check these numbers should land near the same
order of magnitude as, not an exact target (different species, different
anatomy).

### 7.1 Eye open vs. closed — modelled literally, and a real bug this
   surfaced in the first attempt

`neuroam.electrodes.close_eyelid()` paints a thin skin shell
(`thickness_mm=0.30`, matching this project's existing skin-conform
convention — not a measured rat-eyelid thickness, no such measurement was
found in the literature search for this) over the exposed cornea, run as a
second registration of the same montage. Only the three skin-only
montages (VIRON_periorbital, VIRON6, TpES) have a physically sensible closed
variant — TES-GPS and TcES require direct corneal/limbal contact and cannot
be run with the eye shut.

**The full-head "open"/"closed" pairs in `examples/38` do not actually test
this at all -- caught only by a direct challenge to re-verify the finding,
not by anything in the first pass.** `assemble_mrm` (the multires path every
full-head config uses) takes each mesh element's material id from the
reused `.mrm` file's own embedded `records[:, 6]`, **never from
`model.labels`** — confirmed by reading the assembly code directly. That
mesh was built once, for SCL-ON's original open-eye anatomy, and is frozen;
`close_eyelid()` paints `model.labels`, which this path never reads for
bulk tissue. The electrode footprints are also byte-identical between the
open and closed registrations of every pair (same voxel count, same
centroid, confirmed from their own registration reports), so there is no
side channel either. **The full-head open/closed pairs are, numerically,
the same experiment solved twice.** The small field differences originally
reported (up to ~5,000 A/m², ~79M of 413M voxels) are the signature of
ordinary CG/AMG run-to-run non-determinism — the two runs converged in a
different number of iterations (140 vs. 142 for VIRON_periorbital) with
different residuals — not an anatomy effect that never reached the solve.
Read every `_open`/`_closed` pair in §7.2's figure and data as one
data point duplicated, not two.

**The valid test:** `examples/41_verify_eyelid_closure_local.py`, on the
small eye-only crop (`RatCC_eye_180_160_220_83um_with_retina`, 6.44M
voxels) using `assemble_uniform`, which builds G directly from
`model.labels` every time — no static mesh in between, so the eyelid
closure genuinely reaches the solve here. Same VIRON_periorbital electrode
geometry as `examples/07`. Result:

| | open | closed |
|---|---|---|
| retina median \|J\| (A/m², per amp) | 134,861.05 | 134,861.13 |
| CG iterations | 25 | 25 |
| residual | 6.34e-08 | 6.36e-08 |

Retina median ratio (closed/open): **1.0000006** — genuinely, validly
unchanged this time (matching iteration counts, not the mismatched 140/142
above, is itself a sign this comparison is well-posed). The raw field does
differ by a real, non-trivial amount close to the eyelid (max 287 A/m²,
~6.15M of 6.42M voxels differ at all, consistent with a small but genuine
perturbation spreading from the 568-voxel patch) — but 287 A/m² is ~0.2% of
the retina's own typical magnitude (~134,861), and it doesn't move the
retina dose. The original qualitative conclusion — *eye state matters near
the eye, not for the bulk retinal dose a periorbital electrode delivers* —
turns out to be correct, but only this local-crop check actually establishes
it; the full-head pairs never tested it.

### 7.2 Comparison against the original 6 configs

`examples/39_clinical_config_comparison.py` — median `|J|` (A/m²) at each
config's **own paper-cited clinical amplitude** (not the original 6's 200 µA
convention; different real devices, different real doses — matching them
would misrepresent both):

| config | central retina | peripheral retina | occipital | dose |
|---|---|---|---|---|
| SCL-ON | 98.4 | 33.2 | 0.0024 | 200 µA |
| SCL-IntraCranial | 4.73 | 19.2 | 0.151 | 200 µA |
| SCL-TransCranial | 5.27 | 18.9 | 0.091 | 200 µA |
| ON-IntraCranial | 12.6 | 10.9 | 0.143 | 200 µA |
| ON-TransCranial | 13.3 | 11.5 | 0.098 | 200 µA |
| IntraCranial-TransCranial | 0.72 | 0.70 | 0.218 | 200 µA |
| **TES-GPS** | 86.7 | 139 | 0.032 | 1 mA |
| **VIRON periorbital** (open/closed) | 22.6 | 24.4 | 0.043 | 300 µA |
| **VIRON6** (open/closed) | 22.9 | 24.1 | 0.341 | 300 µA |
| **TpES** (open/closed) | 400 | 426 | 1.07 | 4.8 mA |
| **TcES** | 39.9 | 86.4 | 0.208 | 1 mA |

Figure: `samples/ratcc_eye_83um/verification/views/clinical_config_comparison.png`.
Full per-config numbers (including the per-ampere, dose-independent values):
`samples/ratcc_eye_83um/verification/clinical/config_comparison.json`.

At their own clinical doses, every literature design reaches central-retina
current density comparable to or higher than most of the original 6 at 200 µA
— unsurprising since their currents are 1.5–24× higher and periorbital
electrodes are much smaller (higher local current density) than the ring/plate
returns the original 6 use. TpES's headline retinal number is driven by its
4.8 mA upper test bound, the highest tested in its own source paper, not a
typical operating point.

### 7.3 Charge-density safety (Shannon criterion)

`neuroam/safety.py` implements Shannon's `k = log10(Q^2/A)` screening
criterion (k≤1.5 conservative, ~1.75 the looser DBS-convention line) and
scores every source electrode at its own registration
(`examples/38`'s own console output, and each config's
`*_registration_report.json` under `samples/ratcc_eye_83um/verification/clinical/`):

| config | electrode area | amp × phase width | k | vs. k≤1.5 |
|---|---|---|---|---|
| TES-GPS (OkuEl) | 5.47 mm² | 1 mA × 5 ms | 2.660 | **−11.6 dB over** |
| VIRON periorbital (E_sup) | 10.1 mm² | 300 µA × 50 ms | 3.347 | **−18.5 dB over** |
| TpES (CH1) | 9.14 mm² | 4.8 mA × 10 ms | 4.402 | **−29.0 dB over** |
| TcES (ring) | 5.48 mm² | 1 mA × 10 ms | 3.261 | **−17.6 dB over** |

**Every one of these is over the conservative Shannon line, by 12–29 dB.**
This is not a modelling artefact to explain away — it is the direct,
literal consequence of the project's own established scaling convention
(`docs/CLINICAL_ELECTRODES.md` §3: *"Currents are not scaled"*): a human
electrode's clinical current, carried unchanged onto a geometrically
scaled-down (area × k² ≈ 0.069) rat electrode, multiplies the charge density
by roughly the same factor the area shrank by. None of the original human
trials are unsafe at their own scale — the mismatch is purely an artefact of
preserving absolute current across a species-scale change. A rat experiment
built from these numbers would need the current scaled down (by area, or to
a matched charge density) to stay under the same line the human protocol
respects; this repo's own dose convention leaves that choice to whoever
runs the experiment, which is exactly why the safety check exists as a
separate, explicit step rather than an assumption baked into "same current,
smaller electrode."

### 7.4 Electrode contact verification (surface vs. inserted)

`ElectrodeQC.check_contact()` (built on a new `replaced_by_label` QC signal,
alongside the existing `contact_by_label`) classifies every electrode as
**surface** (rests on the target tissue), **inserted** (its own footprint
replaces the target tissue — true here of most pads, by construction: the
`conform` step's `offset_mm=-0.10` deliberately drapes 0.1 mm into the skin,
modelling conductive gel/paste bridging the stratum corneum, per §5.4's own
existing note), or **floating** (no contact at all — a real placement bug,
not a modelling choice). Every one of the 8 registrations was checked before
solving; none floated. See `neuroam/electrodes.py`'s `close_eyelid`,
`ElectrodeQC.check_contact`/`contact_mode`, and `ContactCheck`.

### 7.5 Honest limits of this pass

- **TpES models one channel (CH1, superior), not all four independently, and
  VIRON6 models one cup (E1) against the full 6-cup array's own return, not
  all six simultaneously or the true 6-basis-field optimisation.** Both are
  documented, deliberate scope cuts — the underlying superposition machinery
  (`neuroam.solver.superpose`) supports building the full basis-field set
  later; it is 4–6× the solve time this pass budgeted for, and (for VIRON6)
  the real clinical protocol it's modelled on also drives one cup at a time,
  not all six.
- **The reused SCL-ON `.mrm` mesh is not re-optimised for these new,
  spatially separate electrodes** — same caveat §5 already raises for the
  small-crop designs, now on the full model. One electrode (TpES's CH1) was
  large enough to need bumping past its pure-geometric size specifically to
  land a valid terminal node on this mesh; watch for this on any future
  config placed somewhere this mesh is coarse.
- **The reused static mesh has a sharper consequence than "not re-optimised"
  for anything that changes bulk tissue rather than electrode placement**
  (§7.1): it is *completely blind* to any `model.labels` change outside a
  registered electrode's own footprint. Every `_open`/`_closed` pair in this
  pass's full-head results is the identical experiment solved twice, not two
  anatomies — only `examples/41`'s small-crop `assemble_uniform` check is a
  valid test of an anatomy change. Any future full-head anatomy edit (not
  just eyelid closure) needs the same local-crop-with-uniform-assembly
  treatment, or a re-meshed `.mrm`, to actually reach the solve.
- **"Neck" doesn't exist in this head-only crop** for TpES/TcES's return —
  substituted with a distant skin site, same honest limitation §5 already
  documents for other configs' returns.
- The Shannon criterion is a screening heuristic fit to cortical/nerve
  damage thresholds, not a validated line for cornea/sclera/eyelid tissue —
  treat the −12 to −29 dB figures as "clearly outside the well-characterised
  safe region," not as a precise, tissue-specific risk quantification.

## 8. Next

1. Full `VIRON_orbital6` basis-field decomposition (6 independent solves) to
   reproduce VIRON's own current-density optimisation at rat scale, and all
   four TpES channels independently, per §7.5.
2. Carry the validated configs to the rabbit and human models for Paper 3.
   The clinical numbers live in `configs/clinical_electrode_parameters.json`
   unscaled, so only `k` changes.
3. A literal rat-eyelid-thickness measurement (histology or a literature
   value found by a more targeted search) would replace §7.1's borrowed
   0.30 mm skin-conform thickness with a validated one.
