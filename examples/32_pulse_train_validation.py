"""Pulse-train validation: continuous stimulation, not a single pulse.

Two corrections to examples/31's design, both raised as real concerns
worth re-testing rather than assuming they don't matter:

1. **Continuous stimulation, not a single pulse.** Retinal prosthetic
   systems drive RGCs with a sustained pulse train, not one isolated pulse
   -- 20 Hz is the literature-cited typical rate for epiretinal systems
   (search: "typical stimulus frequency used in epiretinal prosthetic
   systems is 20 Hz", RGCs shown to fire reliably at matching rates up to
   ~50 Hz). This settles 100 ms, then drives a 20 Hz train for 100 ms
   (2 pulses), then records 50 ms of tail.
2. **The settling transient doesn't actually finish by 20 ms** (examples/31's
   choice) -- a zero-drive control run here shows the cell still drifting
   ~1.5 mV between 100 ms and 250 ms, driven by the slow calcium-channel
   dynamics (IT/capump) in this mechanism, not the fast spiking kinetics.
   There is no clean "fully settled" instant to measure a deviation from
   within any practical simulation window. Fixed properly, not by waiting
   longer: a **zero-drive control trace is computed once per cell** (same
   tstop, same everything except no stimulus) and subtracted from every
   trial's trace point-by-point -- this isolates the stimulus-locked
   response regardless of whether the intrinsic drift has finished.

Run once per cell (separate processes):
    python examples/32_pulse_train_validation.py d1
    python examples/32_pulse_train_validation.py a2i

Writes: examples/out/sweep/<cell>_pulsetrain.csv
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

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

OUT = Path(__file__).parent / "out" / "sweep"
OUT.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT / f"{cell_key}_pulsetrain.csv"

dt_ms = 0.025
settle_ms = 100.0          # per the user's correction
train_hz = 20.0             # literature-typical epiretinal rate
train_duration_ms = 100.0   # "stimulate for like 100 ms"
period_s = 1.0 / train_hz
n_pulses = max(1, round(train_duration_ms / (period_s * 1e3)))
tail_ms = 50.0
tstop_ms = settle_ms + train_duration_ms + tail_ms

AMPS_UA = [50, 100, 200, 500, 1000, 2000, 5000, 10000]
LOCATIONS = [0, 80]

print(f"[{cell_key}] settle={settle_ms}ms, {n_pulses} pulses @ {train_hz}Hz "
     f"over {train_duration_ms}ms, tstop={tstop_ms}ms", flush=True)

# ---- load the cell once, and its zero-drive control trace once ------------
sections = load_hoc_cell(cell_dir)
seg_um = get_segment_coordinates(sections, dx_um=1.0)
n_seg = len(seg_um)
morph = read_swc(cell_dir / "morphology" / swc_name)
local_axis = estimate_depth_axis(morph)
soma_um = morph.xyz_um[morph.types == 1].mean(axis=0)
print(f"[{cell_key}] cell loaded: {len(sections)} sections, {n_seg} segments", flush=True)

t0 = time.time()
zero_mat = np.zeros((int(round(tstop_ms / dt_ms)), n_seg))
control_run = run_extracellular(sections, zero_mat, dt_ms=dt_ms, tstop_ms=tstop_ms,
                                record_segments=[0], spike_section_index=0)
control_v = control_run.vm_mV[:, 0]
control_t = control_run.t_ms
print(f"[{cell_key}] zero-drive control computed ({time.time()-t0:.0f}s), "
     f"drift {settle_ms:.0f}-{tstop_ms:.0f}ms = "
     f"{control_v[-1] - control_v[np.argmin(np.abs(control_t-settle_ms))]:.3f} mV", flush=True)

m = CachedMontage("SCL-ON")

fieldnames = ["cell", "ecc_deg", "amp_uA", "unit_v_mean_VperA",
             "unit_v_relspread_pct", "n_spikes", "first_spike_ms",
             "peak_dev_vs_control_mV", "seconds"]
write_header = not OUT_CSV.exists()
f = open(OUT_CSV, "a", newline="")
writer = csv.DictWriter(f, fieldnames=fieldnames)
if write_header:
    writer.writeheader()

trials = [(ecc, amp) for ecc in LOCATIONS for amp in AMPS_UA]
for i, (ecc, amp) in enumerate(trials):
    t0 = time.time()
    anchor_vox = m.voxel_near(ecc, 0)
    radial = m.radial_direction(anchor_vox)
    grid_crop, jmag_crop, origin = m.crop(anchor_vox, pad_vox=40.0)
    target_um = (anchor_vox - origin) * m.dx * 1e6
    seg_vox = place_aligned(seg_um, local_axis, radial, target_um, pivot=soma_um) / (m.dx * 1e6)
    seg_vox = np.clip(seg_vox, 0, np.array(grid_crop.shape) - 1.001)
    unit_v = sample_node_grid(grid_crop, seg_vox)

    wave = biphasic_pulse_train(
        amp_A=amp * 1e-6, pulse_width_s=400e-6, period_s=period_s,
        n_pulses=n_pulses, dt=dt_ms * 1e-3, ratio=1.0,
        delay_s=settle_ms * 1e-3, cathodic_first=True)
    vmat_mV = build_v_matrix(unit_v, wave, unit_scale=1e3)
    # match lengths: build_v_matrix's length follows the waveform, which may
    # be shorter than the control's tstop -- pad with the final (steady,
    # zero) sample so run_extracellular gets a full-length drive.
    n_needed = int(round(tstop_ms / dt_ms))
    if vmat_mV.shape[0] < n_needed:
        pad = np.zeros((n_needed - vmat_mV.shape[0], vmat_mV.shape[1]))
        vmat_mV = np.vstack([vmat_mV, pad])

    run = run_extracellular(sections, vmat_mV, dt_ms=dt_ms, tstop_ms=tstop_ms,
                            record_segments=[0], spike_section_index=0)
    v = run.vm_mV[:, 0]
    n = min(len(v), len(control_v))
    peak_dev = float(np.abs(v[:n] - control_v[:n]).max())

    spread_pct = float((unit_v.max() - unit_v.min()) / abs(unit_v.mean()) * 100) \
        if unit_v.mean() != 0 else float("nan")
    row = dict(cell=cell_key, ecc_deg=ecc, amp_uA=amp,
              unit_v_mean_VperA=float(unit_v.mean()),
              unit_v_relspread_pct=spread_pct, n_spikes=len(run.spikes_ms),
              first_spike_ms=float(run.spikes_ms[0]) if len(run.spikes_ms) else "",
              peak_dev_vs_control_mV=peak_dev, seconds=round(time.time() - t0, 2))
    writer.writerow(row)
    f.flush()
    print(f"[{cell_key}] {i+1}/{len(trials)}: ecc={ecc} amp={amp}uA "
         f"spikes={len(run.spikes_ms)} dev={peak_dev:.3f}mV "
         f"({row['seconds']:.0f}s)", flush=True)

f.close()
print(f"[{cell_key}] wrote {OUT_CSV}")
