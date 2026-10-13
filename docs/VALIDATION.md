# Verification and validation

NeuroAM Studio is evaluated at several independent boundaries: analytic
physics, equivalence with established AM files and systems, cross-validation
against a separate solver implementation, numerical behaviour of the
multiresolution and terminal models, electrode-registration fidelity, and
field-to-neuron coupling.

No single comparison establishes correctness. The layered approach is
designed so that each level detects a different class of error.

## Current automated test snapshot

The suite was run from a clean invocation on 2026-10-13:

```text
106 passed, 4 skipped in 40.67 s
```

Environment: Windows, Python 3.10.8, NumPy 2.2.6, SciPy 1.15.3, and
Matplotlib 3.10.9. The four skipped tests require the separately distributed
`Sph_80` reference dataset selected by `NEUROAM_REFDATA`. Optional-package
tests run when their dependencies are installed.

Reproduce the local suite with:

```bash
python -m pytest -q
```

## Summary of principal results

| Level | Check | Result |
|---|---|---|
| Analytic | Point source versus $\rho I/(4\pi r)$ | 1.91% median; 8.49% maximum error |
| Analytic | Isotropy at equal radius | $2.4\times10^{-14}$ peak-to-peak/mean |
| Analytic | Two-layer slab | less than $10^{-6}$ relative error |
| Conservation | Direct-solve current residual | less than $10^{-9}$ relative |
| Legacy equivalence | Direct assembly versus parsed netlist | identical sparsity; values agree to $10^{-9}$ |
| Legacy equivalence | Size-one multiresolution versus uniform | matrices agree to $10^{-12}$ |
| Model replay | Reference electrode voxels restored to raw anatomy | bit-identical label volume |
| Independent solver | Six montages and 27,455 retinal voxels | mean ratio 0.9900–1.0047 |
| Terminals | Supernode versus legacy terminal | 2,395 to 1,632 CG iterations with negligible dose change |
| Preconditioning | Algebraic multigrid | 2,395 to 47 CG iterations on the eye crop |
| Mesh | Multiresolution versus uniform | 3.70–4.20 times fewer unknowns |
| Electrode fitting | Parametric versus hand-built contacts | Dice 0.893, 0.975, and 0.952 |
| Coupling | Trilinear sampler on a linear field | $10^{-12}$ error |

## Analytic and numerical verification

A 1 mA point source in a $60^3$ homogeneous cube ($\rho=1\ \Omega\,m$,
100 µm voxels, 226,973 unknowns, grounded corners) was compared with
$V=\rho I/(4\pi r)$ in difference form. The median error was 1.91% and the
maximum was 8.49%, with the largest error nearest the singular source.
Potentials sampled at equal radius along the six cardinal axes had a
peak-to-peak variation of $2.4\times10^{-14}$ of the mean.

A two-layer slab between full-face plate electrodes matched the exact
series-resistance solution to better than $10^{-6}$ relative error.
Current-conservation residuals were below $10^{-9}$ for the direct solve, and
iterative and direct solutions agreed to $10^{-6}$.

In a refinement study, reducing voxel size from 166.7 to 80 µm reduced median
point-source error from 10.13% to 0.585%. At smaller voxel sizes, the reported
difference metric rises slightly because its reference node remains fixed in
voxel coordinates and therefore moves physically closer to the source
singularity. This convention must be considered when interpreting that sweep.

## Equivalence with established AM workflows

Four checks evaluate compatibility at the system and file levels:

1. Edge resistances reproduce archived netlist-generator values exactly on a
   reference problem.
2. Direct uniform assembly and parsing an equivalent generated netlist produce
   matrices with identical sparsity and values agreeing to $10^{-9}$.
3. Multiresolution assembly with all cell sizes set to one matches uniform
   assembly to $10^{-12}$. For the `Sph_80` model (272,623 meshed voxels),
   assembly from the reference `.mrm` file produces the same matrix and covers
   exactly the voxels with defined material.
4. Replaying the reference electrode voxels into the raw rat anatomy
   reproduces the hand-built montage file bit for bit.

These checks compare the inputs and linear systems directly, rather than only
comparing derived summary statistics.

## Cross-validation against an independent implementation

A separately implemented MATLAB/C++ AM workflow and NeuroAM solved the same
six `.in`, `.model`, and `.mrm` inputs: a $912\times900\times504$ rat head at
83 µm with 25.1 million unknowns in the multiresolution mesh. Both workflows
used the same retinal region containing 27,455 voxels.

After scaling NeuroAM's unit-current fields to the 200 µA amplitude of the
archived results, the voxelwise ratio of reference to NeuroAM current density
was 0.9900–1.0047 on average across the six montages. Within-montage standard
deviations were 0.5–3.3%. The maximum for the highest-dose montage was
383.85 A/m² in NeuroAM and 383.93 A/m² in the reference output. Both
implementations produced the same ordering of montage-level retinal dose.

| Montage | NeuroAM mean J (A/m²) | Reference mean J (A/m²) | Ratio, mean ± SD |
|---|---:|---:|---:|
| SCL-ON | 37.23 | 37.30 | 1.0025 ± 0.0051 |
| SCL-IntraCranial | 21.74 | 21.63 | 0.9911 ± 0.0148 |
| SCL-TransCranial | 21.60 | 21.49 | 0.9900 ± 0.0165 |
| ON-IntraCranial | 11.30 | 11.26 | 1.0047 ± 0.0326 |
| ON-TransCranial | 11.91 | 11.87 | 1.0042 ± 0.0329 |
| IntraCranial-TransCranial | 0.688 | 0.681 | 0.9919 ± 0.0300 |

Residual differences are consistent with solver tolerances and
boundary-gradient conventions. The comparison supports numerical equivalence
for these inputs; it is not evidence of biological or clinical validity.

The comparison script is `examples/33_verify_against_oecrgc.py`. It requires
the independently distributed reference project selected by
`NEUROAM_OECRGC_DIR`.

## Terminals, preconditioning, and multiresolution cost

On a 6.44-million-unknown eye crop, replacing a legacy single-node terminal
with an equipotential supernode changed retinal dose by $1.5\times10^{-4}$
relative and access impedance by $2\times10^{-5}$, while reducing CG
iterations from 2,395 to 1,632 and solve time from 588 to 375 seconds.
Algebraic multigrid on the same model reduced the iteration count to 47.

Uniform and multiresolution assembly were compared on controlled grids from
$16^3$ to $160^3$, with seven repeated assemble-and-solve runs per condition.
The multiresolution models used 3.70–4.20 times fewer unknowns and were
1.46–11.83 times faster end to end. Median local differences from the
corresponding uniform solutions were 0.05–1.98%; maximum differences were
3.03–9.52% near the source or a coarsening boundary.

These results support multiresolution use when the region of interest is
separated from the coarsened far field. They do not support using a coarsened
region when a few percent of local error near a contact is consequential.

COMSOL measurements in the research figures use a different refinement
convention from the NeuroAM scaling sweep. They provide computational context
only and are not a matched-mesh or matched-accuracy performance benchmark.

## Electrode registration

Parametric contacts fitted to a hand-built montage were regenerated from their
geometric specifications. Dice overlap was 0.893 for the contact-lens ring,
0.975 for the return lead, and 0.952 for its insulation. Only 736 of 6,336,000
crop voxels (0.012%) received a different material assignment.

Across retinal voxels with matching material, current-density magnitude had a
Pearson correlation of 0.948 and integrated retinal dose differed by 3.4%.
Exact replay of the reference masks remains available when bit-identical
reproduction is required; the parametric representation is intended for
controlled placement changes and transfer between compatible anatomies.

## Coupling and end-to-end tests

- Trilinear sampling is exact to $10^{-12}$ on an analytic linear field.
- Label volumes, stimulus files, coordinates, unit fields, and coupling
  matrices pass round-trip tests.
- Symmetric and 1:4 asymmetric biphasic waveforms balance charge to better
  than $10^{-9}$.
- The configuration pipeline builds a model, solves and caches its basis
  field, computes field metrics, writes a manifest, and reuses the cache on a
  second identical run.
- A synthetic tissue–electrode–axon example has an activation threshold of
  approximately 83 µA for a 200 µs cathodic-first biphasic pulse. This is an
  end-to-end regression test, not validation of a particular preparation.

## Interpretation boundary

The results above support software correctness, numerical consistency, and
compatibility for the tested models. They do not establish clinical efficacy,
human perceptual thresholds, or stimulation safety. See
[Scope and limitations](LIMITATIONS.md) before interpreting a model output.
