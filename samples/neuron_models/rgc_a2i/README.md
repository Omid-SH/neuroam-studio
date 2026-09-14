# A2i retinal ganglion cell — sample morphology

The sibling of `../rgc_d1`: same lineage, same `makecell.hoc` build script
(literally the same file, copied again — it already had an `"A2i"` biophysics
branch sitting unused, since the source folder's own `CellTypes.txt` only
ever registered the D1 cell), different reconstructed morphology (`9.swc`
instead of `8.swc`) and different channel densities (e.g. `gnabar_spike`
0.35 at the soma vs. D1's 0.2 — confirmed distinct at load time).

## Provenance

`morphology/9.swc` is **not** from `neuron_exstim_pkg` — that folder only
ships `8.swc` (D1) with `9.swc` present but never referenced by any
`CellTypes.txt`. This copy comes from
`D:\RabbitRGC_Model\Paknahad2020_D1_A2_Rat\morphology\9.swc`, which a survey
of that project confirmed is **byte-identical** to
`neuron_exstim_pkg\Single-RGC\morphology\9.swc` — same underlying
reconstruction, just sitting in a project that had actually wired up the A2i
cell type (`CellTypes.txt` here is new, written for this copy: `9 GC_A2i`).

`*.mod`, `nrnmech.dll` — same as `rgc_d1`: copied from `Single-RGC`'s
mechanism sources and **recompiled fresh** here via `nrnivmodl`/`mknrndll`,
not the source folder's prebuilt binary. `makecell.hoc` carries the same
one-line fix as `rgc_d1`'s copy (scoped `delete_section()` instead of a
process-wide `forall`) — see that folder's README for why.

## One cell per process

This `makecell.hoc` is byte-identical to `rgc_d1`'s (same lineage script).
hoc cannot redefine a `begintemplate`/`obfunc` a second time in one
process — build D1 and A2i in **separate processes** if you need both at
once. `load_hoc_cell()` raises a clear error if you try to build both in
the same one; see `rgc_d1/README.md`'s "One cell per process" section.

## Species — an open question, not settled

The D1/A2(i) naming convention itself comes from a **rabbit** retina study
(Rockhill, Daly, MacNeil, Brown & Masland, 2002). But per the
`Paknahad2020_D1_A2_Rat` project's own investigation, the actual
morphological parameters in this lineage trace to Chen & Chiao (2014), a
**mouse** retina paper — not rabbit, and not rat despite that folder's name.
Treat this cell (and its D1 sibling) as **real reconstructed RGC
morphology of unconfirmed, likely non-rabbit species** — useful for
building and testing a registration/stimulation pipeline, but don't cite it
as a validated rabbit (or rat) model without resolving that provenance
question first.

## Loading it

```python
from neuroam.neuron_link import load_hoc_cell

sections = load_hoc_cell("samples/neuron_models/rgc_a2i")
```

Same API as `rgc_d1` — everything in `examples/26_register_neuron.py` works
by just pointing `RGC_DIR` here instead.
