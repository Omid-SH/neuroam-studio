# D1 retinal ganglion cell — sample morphology

Copied from `D:\OEC-RGC\neuron_exstim_pkg\Single-RGC`, a real reconstructed
morphology (2,873-point `.swc` trace; 329 hoc sections, 1,645 segments once
NEURON discretizes it) already validated by that package's own test runs —
not a toy ball-and-stick.

## What's here (and what isn't)

* `makecell.hoc` — the original build script, with **one line changed**:
  `execute("forall delete_section()", cell)` became 7 explicit
  `cell.<name> { delete_section() }` calls (one per the template's own
  placeholder sections). The original deletes *every* section that exists
  anywhere in the process, not just this cell's own placeholders — harmless
  the first time anything is built, destructive to any other cell already
  built in the same process. Everything else (reads `CellTypes.txt`,
  imports `morphology/8.swc` via `Import3d_SWC_read`, assigns
  Fohlmeister-family biophysics — `spike`/`Ih`/`IT`/`capump` — per region by
  section-name pattern) is unmodified.
* `CellTypes.txt`, `morphology/8.swc` — its two data dependencies.
* `*.mod` — the 6 mechanism sources (`capump`, `Ih`, `IT`, `I_Leak`,
  `spike`, `INaP`; only the first five are actually inserted by
  `makecell.hoc` for this cell type, `INaP` is carried over unused in case a
  future cell copied from the same lineage needs it).
* `nrnmech.dll` — **recompiled fresh in this project's own NEURON
  environment** via `nrnivmodl`, not copied from the source folder. A
  compiled mechanism DLL is an ABI artifact of a specific NEURON build; the
  original prebuilt one is not something to trust across machines. Rebuild
  it yourself any time with `nrnivmodl .` (or `mknrndll .` on Windows) from
  inside this directory if your NEURON version differs.
* **Not copied**: the prebuilt `nrnmech.dll`, `Synapses.txt` (unused by a
  single cell — vestigial from this model's network-simulation lineage),
  `7.swc`/`9.swc`/the stray "hehe 8 - D1 RGC.swc" duplicate, `img.png`, and
  the ~300 MB `axon_exstim_{vm,vext}.csv` example outputs from the
  source folder's own prior run.

## Loading it

```python
from neuroam.neuron_link import load_hoc_cell, get_segment_coordinates, get_segment_edges

sections = load_hoc_cell("samples/neuron_models/rgc_d1")
coords_um = get_segment_coordinates(sections, dx_um=1.0)   # local frame
edges = get_segment_edges(sections)                        # tree connectivity
```

See `examples/26_register_neuron.py` for the full pipeline: registering it
into an AM field, driving it with a real waveform, and producing all three
neuron views (context / local field / activity).

## One cell per process

`../rgc_a2i` ships a byte-identical `makecell.hoc` (same lineage, different
morphology/`CellTypes.txt`). hoc's `begintemplate`/`obfunc` declarations are
process-global and cannot be redefined — building both cells in the *same*
NEURON process doesn't raise a catchable error, it silently aborts the
second build. `load_hoc_cell()` detects this specific case (identical
script content already run) and raises a clear `RuntimeError` rather than
returning an empty, confusing result. If you need both cells at once, build
them in separate processes.

## Species — an open question, not settled

The D1/A2(i) cell-type naming convention comes from a **rabbit** retina
study (Rockhill, Daly, MacNeil, Brown & Masland, 2002). But a survey of a
related project (`Paknahad2020_D1_A2_Rat`, which uses this exact same
`8.swc`) found its own investigation concluded the morphological parameters
in this lineage actually trace to Chen & Chiao (2014), a **mouse** retina
paper — not rabbit, not rat. Treat this as real reconstructed RGC
morphology of unconfirmed, likely non-rabbit species — good for building
and testing a pipeline, not yet something to cite as validated rabbit (or
rat) data. See `../rgc_a2i/README.md` for its sibling cell and the same
caveat.
