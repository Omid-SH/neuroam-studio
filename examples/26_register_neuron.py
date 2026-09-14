"""Register a real multi-compartment neuron into an AM field and visualize it.

End to end: solve a small tissue volume with a stimulating/return electrode
pair, load the lab's D1 retinal ganglion cell (``samples/neuron_models/
rgc_d1`` -- copied from ``neuron_exstim_pkg/Single-RGC``, a real reconstructed
morphology, not a toy cell), place it near the stimulating electrode, sample
the unit field along every compartment, drive it with a real biphasic pulse,
and record every segment's Vm. Produces three views:

  1. ``context.html``  -- where the cell sits inside the tissue volume
  2. ``field.html``    -- zoomed in: the field immediately around the cell
  3. ``activity.html`` -- the cell's own Vm, playing back over time, painted
                          onto its real (SWC) morphology
  4. ``run.npz``       -- the raw run (see neuroam.neuron_link.NeuronRunResult):
                          every segment's Vm at every recorded time step, the
                          registered geometry, and a cropped field snapshot.
                          A different question about *this* run -- a
                          different crop, a different color scale, per-branch
                          spike timing, a static snapshot at some other
                          instant -- is answered by loading this file back
                          (examples/27_replay_neuron_run.py), not by
                          re-solving the field or re-running NEURON.

Run:  python examples/26_register_neuron.py     (requires `pip install neuron`)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from neuroam.materials import Material, MaterialLibrary
from neuroam.model import VoxelModel
from neuroam.assembly import assemble_uniform
from neuroam.solver import solve, unit_current_vector
from neuroam.fields import node_grid, efield, current_density
from neuroam.coupling import sample_node_grid, build_v_matrix
from neuroam.waveforms import biphasic_pulse_train
from neuroam.neuron_link import (neuron_available, load_hoc_cell,
                                 get_segment_coordinates, get_segment_edges,
                                 run_extracellular, NeuronRunResult,
                                 save_neuron_run)
from neuroam.morphology import read_swc
from neuroam import viz3d

REPO = Path(__file__).resolve().parents[1]
RGC_DIR = REPO / "samples/neuron_models/rgc_d1"
OUT = Path(__file__).parent / "out" / "neuron_registration"
OUT.mkdir(parents=True, exist_ok=True)

if not neuron_available():
    sys.exit("NEURON is not installed: pip install neuron")

# ---- 1. tissue + electrode field (unit current) ---------------------------
n = 60
dx = 1e-5                     # 10 um voxels -> 600 um cube
lib = MaterialLibrary([Material.from_sigma(1, 0.3, name="tissue")])
m = VoxelModel.empty((n, n, n), dx=dx, materials=lib, background=1,
                     name="rgc_demo")
m.add_electrode(shape="box", role="source", material=101, waveform="stim",
                corner=(24, 24, 0), size=(6, 6, 2), node=(27, 27, 0))
m.add_electrode(shape="box", role="ground", material=100,
                corner=(24, 24, n - 2), size=(6, 6, 2), node=(27, 27, n))

system = assemble_uniform(m)
res = solve(system, unit_current_vector(system, "stim"), method="cg", rtol=1e-9)
grid = node_grid(system, res.v)
Ex, Ey, Ez, Emag = efield(grid, m.dx)
_, _, _, Jmag = current_density(m, Ex, Ey, Ez)
print(f"field solved: {system.n:,} unknowns, residual {res.residual:.1e}")

# ---- 2. load the real D1 RGC and register it near the electrode -----------
sections = load_hoc_cell(RGC_DIR)
seg_um = get_segment_coordinates(sections, dx_um=1.0)   # local frame, microns
edges = get_segment_edges(sections)
n_seg = len(seg_um)
print(f"D1 RGC loaded: {len(sections)} sections, {n_seg} segments")

morph = read_swc(RGC_DIR / "morphology" / "8.swc")
swc_um = morph.xyz_um
swc_edges = morph.edges()

# place the cell 150 um above the stimulating electrode, centered over it,
# by translating the (soma-anchored) local frame -- one rigid offset applied
# identically to the simulation compartments and the dense SWC trace so both
# stay in registration with each other
target_vox = np.array([27.0, 27.0, 22.0])          # ~150 um above the source
target_um = target_vox * dx * 1e6
translate_um = target_um - seg_um[0]

seg_vox = (seg_um + translate_um) / (dx * 1e6)
swc_vox = (swc_um + translate_um) / (dx * 1e6)

inside = np.all((seg_vox >= 0) & (seg_vox <= n), axis=1)
print(f"segments inside the tissue volume: {inside.sum()}/{n_seg}")

# ---- 3. sample the field, build a real stimulus, run NEURON ---------------
unit_v = sample_node_grid(grid, seg_vox)        # V per A, one per segment
print(f"unit field along the cell: {unit_v.min():.3g} .. {unit_v.max():.3g} V/A")

dt_ms = 0.025
amp_A = 2e-4        # 200 uA, the lab's usual Cur1 amplitude (see docs/VALIDATION.md)
wave = biphasic_pulse_train(amp_A=amp_A, pulse_width_s=0.4e-3, period_s=20e-3,
                           n_pulses=1, dt=dt_ms * 1e-3, ratio=1.0, delay_s=2e-3)
vmat_mV = build_v_matrix(unit_v, wave, unit_scale=1e3)

run = run_extracellular(sections, vmat_mV, dt_ms=dt_ms, tstop_ms=15.0,
                        record_segments=list(range(n_seg)),
                        spike_section_index=0)
print(f"ran {run.vm_mV.shape[0]} steps x {run.vm_mV.shape[1]} segments, "
     f"{len(run.spikes_ms)} spike(s) at soma")

# ---- 4. map simulated Vm onto the dense SWC trace --------------------------
from scipy.spatial import cKDTree
nn_idx = cKDTree(seg_vox).query(swc_vox)[1]
vm_on_swc = run.vm_mV[:, nn_idx]                # (T, n_swc_points)

# ---- 4b. save everything needed to answer a different question about this
# run later, or regenerate any of the three views, without re-solving the
# field or re-running NEURON -- see neuroam.neuron_link.NeuronRunResult.
pad_vox = 15
lo = np.clip(np.floor(swc_vox.min(axis=0) - pad_vox).astype(int), 0, n)
hi = np.clip(np.ceil(swc_vox.max(axis=0) + pad_vox).astype(int), 0, n)
field_crop = Jmag[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]

result = NeuronRunResult(
    t_ms=run.t_ms, vm_mV=run.vm_mV, spikes_ms=np.asarray(run.spikes_ms),
    seg_vox=seg_vox, edges=edges, unit_v=unit_v,
    swc_vox=swc_vox, swc_edges=swc_edges, swc_to_seg=nn_idx,
    field_crop=field_crop,
    meta=dict(model="rgc_demo", world=[n, n, n], dx_m=dx,
             cell_dir=str(RGC_DIR.relative_to(REPO)),
             source_ids=[101], ground_ids=[100],
             target_vox=target_vox.tolist(), dt_ms=dt_ms, amp_A=amp_A,
             pulse_width_s=0.4e-3, period_s=20e-3, delay_s=2e-3,
             tstop_ms=15.0, n_spikes=len(run.spikes_ms),
             field_crop_origin_vox=lo.tolist(), field_unit_label="|J| (A/m^2)"))
run_path = save_neuron_run(OUT / "run.npz", result)
print(f"wrote {run_path} ({run_path.stat().st_size / 1e6:.1f} MB) "
     f"-- reload with neuroam.neuron_link.load_neuron_run, no NEURON needed")

# ---- 5. the three views -----------------------------------------------------
viz3d.view_neuron_context(
    m, swc_vox, swc_edges, OUT / "context.html",
    neuron_name="D1 retinal ganglion cell",
    source_ids=(101,), ground_ids=(100,),
    title="RGC demo — neuron location in the tissue volume")
print(f"wrote {OUT / 'context.html'}")

viz3d.view_neuron_field(
    m, Jmag, swc_vox, swc_edges, OUT / "field.html",
    neuron_name="D1 retinal ganglion cell", pad_vox=15,
    values=vm_on_swc[np.argmax(np.abs(vm_on_swc - vm_on_swc[0]).max(axis=1))],
    values_colorbar_title="Vm (mV) at peak response",
    title="RGC demo — |J| around the cell, colored by its peak response")
print(f"wrote {OUT / 'field.html'}")

viz3d.animate_neuron_activity(
    swc_vox, swc_edges, run.t_ms, vm_on_swc, OUT / "activity.html",
    dx=m.dx, units="um", colorbar_title="Vm (mV)",
    title="RGC demo — membrane potential over time",
    subtitle=f"{amp_A*1e6:.0f} uA biphasic pulse, {len(run.spikes_ms)} spike(s) at soma")
print(f"wrote {OUT / 'activity.html'}")

print(f"\nAll outputs in {OUT}")
