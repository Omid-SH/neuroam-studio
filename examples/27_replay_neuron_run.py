"""Regenerate a neuron run's views from a saved bundle -- no NEURON, no re-solve.

Loads ``examples/out/neuron_registration/run.npz`` (written by
``examples/26_register_neuron.py``) and rebuilds all three views from it
alone: the registered geometry, every segment's Vm at every recorded time
step, and a field snapshot cropped tight around the cell are all that's in
that file, and that's all this script touches. It never imports
``neuroam.neuron_link``'s NEURON-facing pieces (``run_extracellular``,
``load_hoc_cell``) and never calls the AM solver -- this is the answer to
"if I ask a different question about this run later, do I have to rerun the
simulation?": no, as long as the question is about a run that already
happened (a different crop, a different color scale/time window, a static
snapshot at some other instant). A different stimulus, placement, or model
is a new run, not something a saved bundle can answer.

Run:  python examples/26_register_neuron.py    (once, to produce run.npz)
      python examples/27_replay_neuron_run.py   (any time after, no NEURON needed)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from neuroam.materials import Material, MaterialLibrary
from neuroam.model import VoxelModel
from neuroam.neuron_link import load_neuron_run
from neuroam import viz3d

RUN_PATH = Path(__file__).parent / "out" / "neuron_registration" / "run.npz"
OUT = Path(__file__).parent / "out" / "neuron_registration_replay"
OUT.mkdir(parents=True, exist_ok=True)

if not RUN_PATH.exists():
    sys.exit(f"{RUN_PATH} not found -- run examples/26_register_neuron.py first")

result = load_neuron_run(RUN_PATH)
meta = result.meta
print(f"loaded {RUN_PATH}: {result.vm_mV.shape[0]} steps x "
     f"{result.vm_mV.shape[1]} segments, {len(result.spikes_ms)} spike(s), "
     f"{meta['n_spikes']} recorded at save time")
print(f"scenario: {meta['cell_dir']} in a {meta['world']} voxel volume, "
     f"{meta['amp_A']*1e6:.0f} uA @ {meta['dt_ms']} ms/step "
     f"-- from meta, not re-derived")

# ---- rebuild ONLY the (free) geometry the original tissue model had -- no
# solve, this is just voxel labels for the context/field surfaces to render
# against. A real full-head model would cost its (cheap) .model load here,
# never the AM solve.
n = meta["world"][0]
lib = MaterialLibrary([Material.from_sigma(1, 0.3, name="tissue")])
m = VoxelModel.empty(tuple(meta["world"]), dx=meta["dx_m"], materials=lib,
                     background=1, name="rgc_demo_replay")
m.add_electrode(shape="box", role="source", material=101, waveform="stim",
                corner=(24, 24, 0), size=(6, 6, 2), node=(27, 27, 0))
m.add_electrode(shape="box", role="ground", material=100,
                corner=(24, 24, n - 2), size=(6, 6, 2), node=(27, 27, n))

# ---- drop the saved (cropped) field back into a full-size array so the
# existing view_neuron_field can re-crop it exactly as the original run did
field = np.zeros((n, n, n))
lo = np.asarray(meta["field_crop_origin_vox"])
hi = lo + np.asarray(result.field_crop.shape)
field[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = result.field_crop

viz3d.view_neuron_context(
    m, result.swc_vox, result.swc_edges, OUT / "context.html",
    neuron_name="D1 retinal ganglion cell",
    source_ids=tuple(meta["source_ids"]), ground_ids=tuple(meta["ground_ids"]),
    title="RGC demo (replayed) — neuron location in the tissue volume")
print(f"wrote {OUT / 'context.html'}")

vm_on_swc = result.vm_mV[:, result.swc_to_seg]
peak_t = np.argmax(np.abs(vm_on_swc - vm_on_swc[0]).max(axis=1))
viz3d.view_neuron_field(
    m, field, result.swc_vox, result.swc_edges, OUT / "field.html",
    neuron_name="D1 retinal ganglion cell", pad_vox=15,
    values=vm_on_swc[peak_t], values_colorbar_title="Vm (mV) at peak response",
    unit_label=meta.get("field_unit_label", "|J| (A/m^2)"),
    title="RGC demo (replayed) — |J| around the cell, colored by its peak response")
print(f"wrote {OUT / 'field.html'}")

viz3d.animate_neuron_activity(
    result.swc_vox, result.swc_edges, result.t_ms, vm_on_swc,
    OUT / "activity.html", dx=meta["dx_m"], units="um",
    colorbar_title="Vm (mV)",
    title="RGC demo (replayed) — membrane potential over time",
    subtitle=f"{meta['amp_A']*1e6:.0f} uA biphasic pulse, "
            f"{len(result.spikes_ms)} spike(s) at soma")
print(f"wrote {OUT / 'activity.html'}")

print(f"\nAll outputs in {OUT} -- regenerated with no NEURON import, no AM solve")
