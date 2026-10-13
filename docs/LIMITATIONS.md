# Scope and limitations

NeuroAM Studio is a research software framework for numerical modelling. Its
outputs are model-dependent quantities, not clinical recommendations or
evidence of therapeutic efficacy.

## Field formulation

- The solver currently uses a resistive, quasi-static formulation. Capacitive
  entries in legacy material files are parsed but are not included in the
  solve.
- Electrodes are current-controlled. Voltage-controlled (Dirichlet) terminals
  are not implemented.
- Material properties are assigned per voxel. Electrode rasterization is
  binary and does not perform partial-volume conductivity blending.
- Tissue conductivity, anatomy, and boundary conditions are inputs whose
  uncertainty must be evaluated for each scientific application.

## Meshing and scale

- NeuroAM consumes existing multiresolution `.mrm` meshes but does not create
  them. Changes to tissue labels outside a local electrode footprint require a
  compatible remesh or a uniform solve.
- Multiresolution results agree closely with uniform results away from
  coarsening boundaries, but local differences can reach several percent near
  a source or mesh transition. Use a uniform or locally refined model when
  those regions determine the quantity of interest.
- Assembly and solution are in-memory and workstation-scale. Full-head models
  can require tens of gigabytes of memory; distributed solving is not
  implemented.

## Electrode and neuronal models

- Electrode placement quality depends on anatomical labels and on the fitted
  coordinate frame. Registration reports and visual inspection remain part of
  model quality control.
- Eye or tissue crops cannot represent distant return electrodes faithfully;
  absolute impedance and dose from a crop should not be interpreted as a
  full-head result.
- The included neuronal demonstrations use a small number of reconstructed
  cells. They do not establish population recruitment, perceptual thresholds,
  or safety limits, and they do not model axons of passage from other cells.
- High exploratory stimulus amplitudes in examples are numerical test inputs,
  not dosing recommendations.

## Comparisons with other software

Available COMSOL results provide computational context for a homogeneous
point-source problem, but the published sweeps use different refinement
conventions. They are not a matched-physics, matched-mesh performance
benchmark, and the documentation makes no quantitative speed claim against
finite-element software.

See [Validation](VALIDATION.md) for the tests that have been completed and the
scope of each reported result.
