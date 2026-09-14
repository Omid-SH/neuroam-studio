"""Vm-trace figures for the pulse-train validation (examples/32): what the
2-spikes-vs-1-spike finding in docs/STIMULATION_PARAMETER_STUDY.md section 8
actually looks like, not just the summary CSV numbers.

Re-runs exactly three NEURON trials for one cell, at the one location/
amplitude where both cells reach threshold (SCL-ON, ecc 0 deg, correct
orientation, cathodic-first, 10 mA):

1. The single pulse from examples/31 (20 ms settle, one 400 us pulse,
   35 ms trial).
2. The 20 Hz pulse train from examples/32 (100 ms settle, 2 pulses over
   100 ms, 250 ms trial).
3. That same cell's zero-drive control (250 ms, no stimulus) -- the trace
   examples/32 subtracts from every trial's raw trace (see its module
   docstring / docs/STIMULATION_PARAMETER_STUDY.md section 8.2).

Produces one two-panel figure per cell: the single-pulse trial's raw Vm on
top, the pulse-train trial overlaid with its own control on the bottom (same
250 ms window, so the second-pulse effect is directly visible against the
trace it's compared to).

Run once per cell (separate processes -- see neuroam.neuron_link.load_hoc_cell):
    python examples/34_pulse_train_traces.py d1
    python examples/34_pulse_train_traces.py a2i

Writes: examples/out/pulsetrain_figures/<cell>_vm_traces.png
       examples/out/pulsetrain_figures/<cell>_train_waveform.png
       examples/out/pulsetrain_figures/<cell>_traces.npz (raw arrays, for reuse)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from neuroam import viz
from neuroam.coupling import sample_node_grid, build_v_matrix, place_aligned
from neuroam.waveforms import biphasic_pulse_train
from neuroam.morphology import read_swc, estimate_depth_axis
from neuroam.neuron_link import (neuron_available, load_hoc_cell,
                                 get_segment_coordinates, run_extracellular)

from _sweep_common import CachedMontage

CELLS = {
    "d1": (Path(__file__).resolve().parents[1] / "samples/neuron_models/rgc_d1", "8.swc"),
    "a2i": (Path(__file__).resolve().parents[1] / "samples/neuron_models/rgc_a2i", "9.swc"),
}

if not neuron_available():
    sys.exit("NEURON is not installed: pip install neuron")
if len(sys.argv) != 2 or sys.argv[1] not in CELLS:
    sys.exit(f"usage: python {Path(__file__).name} {{{'|'.join(CELLS)}}}")
cell_key = sys.argv[1]
cell_dir, swc_name = CELLS[cell_key]

OUT = Path(__file__).parent / "out" / "pulsetrain_figures"
OUT.mkdir(parents=True, exist_ok=True)

dt_ms = 0.025
AMP_UA = 10000
ECC = 0

# single-pulse trial (examples/31's design)
SP_SETTLE_MS, SP_TSTOP_MS, SP_PERIOD_S = 20.0, 35.0, 20e-3
# pulse-train trial (examples/32's design)
TR_SETTLE_MS, TR_HZ, TR_DUR_MS, TR_TAIL_MS = 100.0, 20.0, 100.0, 50.0
TR_PERIOD_S = 1.0 / TR_HZ
TR_N_PULSES = max(1, round(TR_DUR_MS / (TR_PERIOD_S * 1e3)))
TR_TSTOP_MS = TR_SETTLE_MS + TR_DUR_MS + TR_TAIL_MS

print(f"[{cell_key}] single pulse: settle={SP_SETTLE_MS}ms tstop={SP_TSTOP_MS}ms; "
     f"train: {TR_N_PULSES} pulses @ {TR_HZ}Hz, settle={TR_SETTLE_MS}ms "
     f"tstop={TR_TSTOP_MS}ms", flush=True)

# ---- cell + placement, identical recipe to examples/31 and 32 -------------
sections = load_hoc_cell(cell_dir)
seg_um = get_segment_coordinates(sections, dx_um=1.0)
morph = read_swc(cell_dir / "morphology" / swc_name)
local_axis = estimate_depth_axis(morph)
soma_um = morph.xyz_um[morph.types == 1].mean(axis=0)
print(f"[{cell_key}] cell loaded: {len(sections)} sections, {len(seg_um)} segments", flush=True)

m = CachedMontage("SCL-ON")
anchor_vox = m.voxel_near(ECC, 0)
radial = m.radial_direction(anchor_vox)
grid_crop, jmag_crop, origin = m.crop(anchor_vox, pad_vox=40.0)
target_um = (anchor_vox - origin) * m.dx * 1e6
seg_vox = place_aligned(seg_um, local_axis, radial, target_um, pivot=soma_um) / (m.dx * 1e6)
seg_vox = np.clip(seg_vox, 0, np.array(grid_crop.shape) - 1.001)
unit_v = sample_node_grid(grid_crop, seg_vox)


def run_trial(tstop_ms: float, wave):
    vmat_mV = build_v_matrix(unit_v, wave, unit_scale=1e3)
    n_needed = int(round(tstop_ms / dt_ms))
    if vmat_mV.shape[0] < n_needed:
        pad = np.zeros((n_needed - vmat_mV.shape[0], vmat_mV.shape[1]))
        vmat_mV = np.vstack([vmat_mV, pad])
    return run_extracellular(sections, vmat_mV, dt_ms=dt_ms, tstop_ms=tstop_ms,
                             record_segments=[0], spike_section_index=0)


# 1. single pulse (examples/31)
sp_wave = biphasic_pulse_train(amp_A=AMP_UA * 1e-6, pulse_width_s=400e-6,
                               period_s=SP_PERIOD_S, n_pulses=1, dt=dt_ms * 1e-3,
                               ratio=1.0, delay_s=SP_SETTLE_MS * 1e-3, cathodic_first=True)
sp = run_trial(SP_TSTOP_MS, sp_wave)
print(f"[{cell_key}] single pulse: {len(sp.spikes_ms)} spike(s) "
     f"{list(np.round(sp.spikes_ms, 2))}", flush=True)

# 2. pulse train (examples/32)
tr_wave = biphasic_pulse_train(amp_A=AMP_UA * 1e-6, pulse_width_s=400e-6,
                               period_s=TR_PERIOD_S, n_pulses=TR_N_PULSES,
                               dt=dt_ms * 1e-3, ratio=1.0,
                               delay_s=TR_SETTLE_MS * 1e-3, cathodic_first=True)
tr = run_trial(TR_TSTOP_MS, tr_wave)
print(f"[{cell_key}] pulse train: {len(tr.spikes_ms)} spike(s) "
     f"{list(np.round(tr.spikes_ms, 2))}", flush=True)

# 3. zero-drive control, same length as the train trial
zero_mat = np.zeros((int(round(TR_TSTOP_MS / dt_ms)), len(seg_um)))
ctrl = run_extracellular(sections, zero_mat, dt_ms=dt_ms, tstop_ms=TR_TSTOP_MS,
                         record_segments=[0], spike_section_index=0)
print(f"[{cell_key}] zero-drive control: {len(ctrl.spikes_ms)} spike(s)", flush=True)

np.savez_compressed(
    OUT / f"{cell_key}_traces.npz",
    sp_t=sp.t_ms, sp_v=sp.vm_mV[:, 0], sp_spikes=np.asarray(sp.spikes_ms),
    tr_t=tr.t_ms, tr_v=tr.vm_mV[:, 0], tr_spikes=np.asarray(tr.spikes_ms),
    ctrl_t=ctrl.t_ms, ctrl_v=ctrl.vm_mV[:, 0])

# ---- figure: single-pulse trial on top, train-vs-control below ------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), dpi=130)
viz.save_vm_comparison(
    None, {f"single pulse ({len(sp.spikes_ms)} spike)": (sp.t_ms, sp.vm_mV[:, 0])},
    title=f"{cell_key.upper()} -- single 400 us pulse, ecc 0 deg, 10 mA (examples/31 design)",
    ax=ax1)
ax1.axvline(SP_SETTLE_MS, color="gray", ls="--", lw=0.8, alpha=0.6)
viz.save_vm_comparison(
    None, {f"20 Hz train ({len(tr.spikes_ms)} spikes)": (tr.t_ms, tr.vm_mV[:, 0]),
          "zero-drive control": (ctrl.t_ms, ctrl.vm_mV[:, 0])},
    title=f"{cell_key.upper()} -- 20 Hz/100 ms pulse train, ecc 0 deg, 10 mA (examples/32 design)",
    ax=ax2)
for pulse_i in range(TR_N_PULSES):
    ax2.axvline(TR_SETTLE_MS + pulse_i / TR_HZ * 1e3, color="gray", ls="--", lw=0.8, alpha=0.6)
fig.tight_layout()
fig_path = OUT / f"{cell_key}_vm_traces.png"
fig.savefig(fig_path)
plt.close(fig)
print(f"[{cell_key}] wrote {fig_path}", flush=True)

wf_path = OUT / f"{cell_key}_train_waveform.png"
viz.save_waveform(wf_path, tr_wave, title=f"{TR_N_PULSES}x 400 us biphasic pulses @ {TR_HZ} Hz, "
                                          f"{AMP_UA/1000:.0f} mA, cathodic-first")
print(f"[{cell_key}] wrote {wf_path}", flush=True)
