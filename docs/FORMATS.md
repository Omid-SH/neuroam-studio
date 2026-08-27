# Legacy file formats (digested)

Primary reference: *Multi-resolution 3D Impedance Method — Implementation
reference and notes*, Carlos J. Cela (BEML / Lazzi group), V0.99.9 — the PDF in
`Python-AM_NEURON-Unreleased/Admittance Method/Documents/`.  Everything below was
additionally verified against real files in the lab folders.

## `.in` — simulation parameter file
Plain ASCII, one declaration per line; `#`/`%` start comments.

| declaration | meaning |
|---|---|
| `world nx ny nz` | model size in unit voxels |
| `unitvoxelsize s` | voxel edge length in **meters** |
| `material id rx ix ry iy rz iz` | per-axis **resistivity** (ohm·m), real + imaginary parts. Insulator ≈ 1e7, metal electrode ≈ 1e-7 |
| `matrix f` / `meshfile f` / `networkfile f` / `nodevoltagefile f` | file names for `.model`(`.mat`), `.mrm`, `.net`, `.vof` |
| `maximumsize n` | max multires cluster size (power of 2; 1 = uniform) |
| `volume cs sx sy sz ox oy oz cr` | high-resolution sub-volume constraint |
| `clusteringratio n` | max neighbor size ratio (must be 2 for netgen/solver) |
| `spice I <cur> [node x y z] [node x y z]` | current source macro; first node = injection, `<cur>.cur` holds the waveform |
| `nodename x y z name` | rename a node in the netlist; `name = 0` ⇒ ground |
| `defaultmaterial id` | fallback for labels missing from the material list |

## `.model` / `.mat` — voxel labels
ASCII integers. The lab's `.model` files are headerless: `nz` blocks of `ny`
lines × `nx` values (legacy loader: `reshape(nz, ny, nx).transpose(2,1,0)` →
`[x][y][z]`). (The documented `.mat` variant adds an `X Y Z` header line.)
Material 0 / undefined material = "outside world": the mesher does not mesh it.

## `.mrm` — multiresolution mesh
One voxel record per line: `id x y z sx sy sz material` — anchor = lower corner
(unit-lattice coords), sizes in unit voxels. Produced by `mesher`; consumed by
`netgen` — and directly by `neuroam.assembly.assemble_mrm` (no netlist needed).

## `.net` — SPICE-subset netlist (what netgen emits)
Header: `* param nodename <name> 0` (ground), ` I <cur> <node+> 0` (source).
Then per voxel: `* voxel x y z sx sy sz mat` followed by **12 `Res` lines**
(edge resistors between the voxel's 8 corners). Node names are
`xxxxyyyyzzzz` (4 digits per coordinate, lower-left-back corner).

**Resistor value** (verified numerically against QuarterHead netlist:
ρ=5000 Ω·m, Δ=0.25 mm → size-1: 8e7 Ω, size-2 cube: 4e7 Ω):

    R_axis = 4 · ρ_axis · s_axis / (s_perp1 · s_perp2 · Δ)

Shared edges parallel-combine, i.e. conductances add — which is what direct
assembly does without ever writing the text file.
`Cap` lines encode the imaginary part (ignored by the resistive solver).

## `.cur` — stimulus waveform
First line `% <dt_seconds> <x>`, then one current amplitude (amperes) per
line, one per time step.

## `.vof` — node voltages
One line per netlist node: `<nodename> <amplitude> [phase]` (volts; ground
row uses name `0`).

## `.vavg` / `_V.raw`, `_J.raw`
Voxel-center averages (mean of 8 corner node voltages) as text in the
`.model` layout, and float32 raw volumes for ParaView.

## coordinates / `.v` — NEURON coupling
- coordinates file (`get_coords.hoc` output): `x y z` per compartment in
  unit-voxel units (floats).
- unit-field file (`interp3` output): one line, one value per compartment.
- `.v` waveform matrix (`Makewaveform2.py` output): T rows × n_compartments
  columns; the Stim hoc plays column *i* into compartment *i*'s
  `e_extracellular` (mV).
