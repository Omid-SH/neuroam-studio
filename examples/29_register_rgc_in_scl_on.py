"""Register a real RGC into a real RatCC montage, using the cached solve.

Loads the field cache written by examples/28_solve_scl_on_for_registration.py
for the named config (so this never re-solves the 25.1M-unknown field),
places the named cell's soma at the same central-retina anchor used to
build that cache -- anatomically oriented (see neuroam.morphology.
estimate_depth_axis / neuroam.coupling.place_aligned: the cell's own
soma-to-dendrite axis is rotated to the local retinal radial direction,
not just translated in with whatever orientation the source reconstruction
happened to have) -- drives it with the montage's real 200 uA Cur1 biphasic
waveform, records every segment, and produces the three neuron views, a
live placement editor, and a reloadable run.npz.

D1 and A2i share a byte-identical hoc build script and cannot be built in
the same NEURON process (see neuroam.neuron_link.load_hoc_cell's docstring)
-- run this once per cell, as separate processes:

    python examples/29_register_rgc_in_scl_on.py d1 SCL-ON
    python examples/29_register_rgc_in_scl_on.py a2i SCL-ON

config-key one of SCL-ON, SCL-IntraCranial, SCL-TransCranial,
ON-IntraCranial, ON-TransCranial, IntraCranial-TransCranial (default SCL-ON).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.spatial import cKDTree

from neuroam.materials import Material, MaterialLibrary
from neuroam.model import VoxelModel
from neuroam.coupling import sample_node_grid, build_v_matrix, place_aligned
from neuroam.frames import Frame
from neuroam.waveforms import biphasic_pulse_train
from neuroam.morphology import read_swc, estimate_depth_axis
from neuroam.neuron_link import (neuron_available, load_hoc_cell,
                                 get_segment_coordinates, get_segment_edges,
                                 run_extracellular, NeuronRunResult,
                                 save_neuron_run)
from neuroam import viz3d

RETINA = 77

REPO = Path(__file__).resolve().parents[1]
SAMP = REPO / "samples/ratcc_eye_83um"

CONFIGS = {
    "SCL-ON": "RatCC_full_CLStim_JGND_912_900_504_Ring_Res_83um_with_retina",
    "SCL-IntraCranial": "RatCC_full_CLStim_NeedleGND_912_900_504_Ring_Res_83um_with_retina",
    "SCL-TransCranial": "RatCC_full_CLStim_PatchGND_912_900_504_Ring_Res_83um_with_retina",
    "ON-IntraCranial": "RatCC_full_JStim_NeedleGND_912_900_504_Ring_Res_83um_with_retina",
    "ON-TransCranial": "RatCC_full_JStim_PatchGND_912_900_504_Ring_Res_83um_with_retina",
    "IntraCranial-TransCranial": "RatCC_full_NeedleStim_PlateGND_912_900_504_Res_83um_with_retina",
}
CELLS = {
    "d1": (REPO / "samples/neuron_models/rgc_d1", "8.swc", "D1 retinal ganglion cell"),
    "a2i": (REPO / "samples/neuron_models/rgc_a2i", "9.swc", "A2i retinal ganglion cell"),
}

if not neuron_available():
    sys.exit("NEURON is not installed: pip install neuron")
if len(sys.argv) < 2 or sys.argv[1] not in CELLS:
    sys.exit(f"usage: python {Path(__file__).name} {{{'|'.join(CELLS)}}} [config-key]")
cell_key = sys.argv[1]
cfg = sys.argv[2] if len(sys.argv) > 2 else "SCL-ON"
if cfg not in CONFIGS:
    sys.exit(f"unknown config {cfg!r}; choose from {list(CONFIGS)}")
cell_dir, swc_name, cell_label = CELLS[cell_key]
fname = CONFIGS[cfg]

CACHE_PATH = SAMP / "verification" / f"{cfg.replace(' ', '_')}_registration_cache.npz"
FULL_IN = SAMP / f"{fname}.in"

OUT = Path(__file__).parent / "out" / "scl_on_registration" / cfg / cell_key
OUT.mkdir(parents=True, exist_ok=True)

if not CACHE_PATH.exists():
    sys.exit(f"{CACHE_PATH} not found -- run "
            f"examples/28_solve_scl_on_for_registration.py {cfg!r} first")

# ---- 1. load the cached field crop (no re-solve) ---------------------------
cache = np.load(CACHE_PATH, allow_pickle=False)
grid_crop = cache["grid_crop"].astype(np.float64)
jmag_crop = cache["jmag_crop"].astype(np.float64)
labels_crop = cache["labels_crop"]
origin_vox = cache["origin_vox"]
anchor_vox_global = cache["anchor_vox"]
dx = float(cache["dx_m"])
world = cache["world"]
anchor_vox_local = anchor_vox_global - origin_vox
print(f"[{cfg}/{cell_key}] loaded cache: grid crop {grid_crop.shape}, "
     f"anchor (local) {anchor_vox_local}, dx {dx*1e6:.0f} um")

# ---- 2. load the full model once -- needed both for the retina's local
# radial direction (below) and the context view (step 5); loading it here
# means it isn't loaded twice.
full_model = VoxelModel.from_legacy(FULL_IN)
fr = Frame.eye(full_model, (RETINA,))
anchor_mm = fr.grid.centers_mm(anchor_vox_global.reshape(1, 3))[0]
target_axis = anchor_mm - fr.origin_mm
target_axis = target_axis / np.linalg.norm(target_axis)
print(f"[{cfg}/{cell_key}] local radial direction at the anchor: {target_axis}")

# ---- 3. load the cell and register it, anatomically oriented, at the anchor
sections = load_hoc_cell(cell_dir)
seg_um = get_segment_coordinates(sections, dx_um=1.0)
edges = get_segment_edges(sections)
n_seg = len(seg_um)
print(f"[{cfg}/{cell_key}] {cell_label} loaded: {len(sections)} sections, {n_seg} segments")

morph = read_swc(cell_dir / "morphology" / swc_name)
swc_um = morph.xyz_um
swc_edges = morph.edges()

# a retinal ganglion cell is thin along exactly the axis its own soma and
# dendrites lie on (soma in the ganglion cell layer, dendrites reaching into
# the inner plexiform layer) -- anatomically, THAT axis should end up
# perpendicular to the retina here, i.e. aligned with the local radial
# direction, not just carried over from whatever orientation the source
# reconstruction happened to have (a pure translate, as this was before).
local_axis = estimate_depth_axis(morph)
soma_um = morph.xyz_um[morph.types == 1].mean(axis=0)
target_um_local = anchor_vox_local * dx * 1e6

seg_vox_local = place_aligned(seg_um, local_axis, target_axis,
                              target_um_local, pivot=soma_um) / (dx * 1e6)
swc_vox_local = place_aligned(swc_um, local_axis, target_axis,
                              target_um_local, pivot=soma_um) / (dx * 1e6)
seg_vox_global = seg_vox_local + origin_vox
swc_vox_global = swc_vox_local + origin_vox

inside = np.all((seg_vox_local >= 0) & (seg_vox_local <= np.array(grid_crop.shape) - 1), axis=1)
print(f"[{cfg}/{cell_key}] segments inside the cached crop: {inside.sum()}/{n_seg}")

# ---- 4. sample the real field, drive with the montage's real waveform -----
unit_v = sample_node_grid(grid_crop, seg_vox_local)
print(f"[{cfg}/{cell_key}] unit field along the cell: {unit_v.min():.3g} .. {unit_v.max():.3g} V/A")

dt_ms = 0.025
amp_A = 2e-4        # 200 uA Cur1 -- the montage's own validated amplitude
wave = biphasic_pulse_train(amp_A=amp_A, pulse_width_s=0.4e-3, period_s=20e-3,
                           n_pulses=1, dt=dt_ms * 1e-3, ratio=1.0, delay_s=2e-3)
vmat_mV = build_v_matrix(unit_v, wave, unit_scale=1e3)

run = run_extracellular(sections, vmat_mV, dt_ms=dt_ms, tstop_ms=15.0,
                        record_segments=list(range(n_seg)), spike_section_index=0)
print(f"[{cfg}/{cell_key}] ran {run.vm_mV.shape[0]} steps x {run.vm_mV.shape[1]} segments, "
     f"{len(run.spikes_ms)} spike(s) at soma")

# ---- 5. map onto the dense SWC trace, save the reloadable bundle ----------
nn_idx = cKDTree(seg_vox_local).query(swc_vox_local)[1]
vm_on_swc = run.vm_mV[:, nn_idx]

field_pad = 15
lo = np.clip(np.floor(swc_vox_local.min(axis=0) - field_pad).astype(int), 0, None)
hi = np.clip(np.ceil(swc_vox_local.max(axis=0) + field_pad).astype(int) + 1,
            None, np.array(jmag_crop.shape))
field_crop = jmag_crop[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]

result = NeuronRunResult(
    t_ms=run.t_ms, vm_mV=run.vm_mV, spikes_ms=np.asarray(run.spikes_ms),
    seg_vox=seg_vox_local, edges=edges, unit_v=unit_v,
    swc_vox=swc_vox_local, swc_edges=swc_edges, swc_to_seg=nn_idx,
    field_crop=field_crop,
    meta=dict(model=fname, config=cfg, cell=cell_key,
             cell_dir=str(cell_dir.relative_to(REPO)),
             world=world.tolist(), dx_m=dx,
             anchor_vox_global=anchor_vox_global.tolist(),
             origin_vox=origin_vox.tolist(), dt_ms=dt_ms, amp_A=amp_A,
             pulse_width_s=0.4e-3, period_s=20e-3, delay_s=2e-3, tstop_ms=15.0,
             n_spikes=len(run.spikes_ms), field_crop_origin_local_vox=lo.tolist(),
             field_unit_label="|J| (A/m^2)"))
run_path = save_neuron_run(OUT / "run.npz", result)
print(f"[{cfg}/{cell_key}] wrote {run_path} ({run_path.stat().st_size/1e6:.1f} MB)")

# ---- 6. the three views + placement editor ---------------------------------
lib = MaterialLibrary([Material.from_sigma(1, 1.0)])   # unused by viz3d, placeholder
local_model = VoxelModel(labels=labels_crop, dx=dx, materials=lib,
                         name=f"{cfg}_crop_{cell_key}")

viz3d.view_neuron_context(
    full_model, swc_vox_global, swc_edges, OUT / "context.html",
    neuron_name=cell_label, focus={"retina": [77]},
    source_ids=(101,), ground_ids=(100, 102),
    context_step=6, secondary_step=3,
    title=f"{cfg} — {cell_label} registered on the retina")
print(f"[{cfg}/{cell_key}] wrote {OUT / 'context.html'}")

viz3d.view_neuron_placement_editor(
    full_model, swc_vox_global, swc_edges, OUT / "placement_editor.html",
    neuron_name=cell_label, focus={"retina": [77]},
    source_ids=(101,), ground_ids=(100, 102),
    context_step=6, secondary_step=3,
    title=f"{cfg} — {cell_label} placement editor")
print(f"[{cfg}/{cell_key}] wrote {OUT / 'placement_editor.html'}")

peak_t = int(np.argmax(np.abs(vm_on_swc - vm_on_swc[0]).max(axis=1)))
viz3d.view_neuron_field(
    local_model, jmag_crop, swc_vox_local, swc_edges, OUT / "field.html",
    neuron_name=cell_label, pad_vox=field_pad,
    values=vm_on_swc[peak_t], values_colorbar_title="Vm (mV) at peak response",
    title=f"{cfg} — |J| around the {cell_label}, colored by peak response")
print(f"[{cfg}/{cell_key}] wrote {OUT / 'field.html'}")

viz3d.animate_neuron_activity(
    swc_vox_local, swc_edges, run.t_ms, vm_on_swc, OUT / "activity.html",
    dx=dx, units="um", colorbar_title="Vm (mV)",
    title=f"{cfg} — {cell_label} membrane potential over time",
    subtitle=f"{amp_A*1e6:.0f} uA biphasic Cur1 pulse, {len(run.spikes_ms)} spike(s) at soma")
print(f"[{cfg}/{cell_key}] wrote {OUT / 'activity.html'}")

print(f"\n[{cfg}/{cell_key}] All outputs in {OUT}")
