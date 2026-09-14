# Sample: RatCC with-retina eye region (83 um, CL-stim / J-gnd)

Native-resolution crop of the lab model
`RatCC_full_CLStim_JGND_912_900_504_Ring_Res_83um_with_retina.model`
(D:\AM models\Models\with retina), containing the whole eye with the retina
layer (label 77), the contact-lens stimulating electrode (101) and the J
return electrode (102), plus surrounding head tissue.

- crop (0-based voxels of the full model): x [230,410), y [290,450), z [150,370)
  -> world 180 x 160 x 220 at 83 um (unchanged resolution; nothing downsampled)
- to map a crop coordinate back to the full model: add (230, 290, 150)
- electrodes (crop frame, IN_editor_2 node convention):
  stim node (96, 46, 92) inside label 101; ground node (129, 104, 134) inside 102
- materials: the lab `sample.in` table + `material 0077 0.1` (retina) from
  `calculate_retina_opticnerve_EF.m`
- note: in the *with-retina* full model the legacy sample.in ground node
  (418,388,484) lands on material 0; the label-102 electrode actually spans
  x[346,361] y[390,409] z[234,343] (0-based) — the full-model .in provided
  here uses its middle voxel, node (359, 394, 284).

## Run

    python -m neuroam run configs/ratcc_eye_83um.json

Inspect the anatomy and painted electrodes interactively without solving:

    python -m neuroam view samples/ratcc_eye_83um/RatCC_eye_180_160_220_83um_with_retina.in --labels 24 33 66 77

This adds 3D surfaces for the optic nerve (24), sclera (33), cornea (66), and
retina (77); the metal source (101) and ground (102) are detected automatically.
The movable colored plane shows every other tissue label in cross-section.
Install the optional viewer first with `python -m pip install -e ".[viewer]"`.

6.44M unknowns; ~2400 CG iterations (diagonal preconditioner). Outputs land in
`samples/ratcc_eye_83um/results/`: V and |E| slices, unit-current `.vof`,
`.npz` fields, and retina/electrode dose metrics in `run_manifest.json`
(per 1 A — multiply by your amplitude; Cur1 = 200 uA in the bundled waveform).

Reference result (this crop, unit current): retina E_mean = 1.82e4 V/m/A,
E_p95 = 3.03e4 V/m/A -> at 200 uA: 3.6 / 6.1 V/m. KCL residual 2.2e-7.

## Full model

`RatCC_full_CLStim_JGND_912_900_504_Ring_Res_83um_with_retina.in` (in this
folder) pairs with the full 912x900x504 model in `D:\AM models\Models\with
retina` — copy the .in next to the .model, or use `configs/` with absolute
paths. Fair warning: ~420M nodes; the current uniform assembler needs roughly
50-60 GB peak RAM — run it on a large workstation or CARC, or mesh it with the
lab mesher and pass the `.mrm` (multiresolution path) to shrink the system.
