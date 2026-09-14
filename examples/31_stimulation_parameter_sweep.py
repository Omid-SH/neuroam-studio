"""A broad parameter sweep: location, orientation, amplitude, and waveform.

Reuses the 6 already-solved, already-cached fields (examples/28) -- nothing
here re-solves the AM problem, only NEURON. One cell is built ONCE per
process (the hoc-template-per-process constraint -- see
neuroam.neuron_link.load_hoc_cell's docstring) and reused across every
trial; only the registration (location/orientation) and the drive
(amplitude/waveform) change per trial. Each trial records only the soma's
Vm (not the full 1645-segment trace) and a spike count -- this is a
screening sweep, not a repeat of the full per-scenario visualization done
for the base 12 registrations.

Run once per cell (separate processes):
    python examples/31_stimulation_parameter_sweep.py d1
    python examples/31_stimulation_parameter_sweep.py a2i

Writes: examples/out/sweep/<cell>_sweep.csv (one row per trial, appended
live so a partial run is still readable)
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

from _sweep_common import CachedMontage, rotate_about_tangent

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
OUT_CSV = OUT / f"{cell_key}_sweep.csv"

dt_ms = 0.025
# the model's intrinsic channel relaxation from the artificial v_init=-65 mV
# to its own true resting potential (~-56 mV for this Fohlmeister-family
# mechanism) takes ~15-20 ms on its own, with no stimulus at all -- confirmed
# by inspecting a raw trace before trusting any amplitude/waveform sweep
# result. The pulse has to fire *after* that settles, or "response" mostly
# measures the settling transient, not the stimulus.
settle_ms = 20.0
delay_s = settle_ms * 1e-3
tstop_ms = settle_ms + 15.0
period_s = 20e-3

# ---- build the full trial list (see docs/STIMULATION_SWEEP.md for design rationale)
ECC_LOCATIONS = [0, 5, 15, 30, 45, 60, 75, 80]
SCREEN_MONTAGES = ["SCL-ON", "SCL-IntraCranial", "IntraCranial-TransCranial"]
AMPS_UA = [50, 100, 200, 500, 1000, 2000, 5000, 10000]
ORIENT_DEVS = [0, 30, 60, 90, 120, 150, 180]
PULSE_WIDTHS_US = [50, 100, 200, 400, 700, 1000, 2000]
PHIS = [0, 90, 180, 270]

trials = []
# A: location screen
for cfg in SCREEN_MONTAGES:
    for ecc in ECC_LOCATIONS:
        trials.append(dict(stage="A_location", cfg=cfg, ecc=ecc, phi=0,
                           orient_dev=0, amp_uA=2000, pw_us=400, cathodic=True))
# B: amplitude sweep, near-electrode (80) vs far (0), across montages
for cfg in SCREEN_MONTAGES:
    for ecc in (0, 80):
        for amp in AMPS_UA:
            trials.append(dict(stage="B_amplitude", cfg=cfg, ecc=ecc, phi=0,
                               orient_dev=0, amp_uA=amp, pw_us=400, cathodic=True))
# C: orientation sweep, near-electrode vs far, SCL-ON
for ecc in (0, 80):
    for dev in ORIENT_DEVS:
        trials.append(dict(stage="C_orientation", cfg="SCL-ON", ecc=ecc, phi=0,
                           orient_dev=dev, amp_uA=2000, pw_us=400, cathodic=True))
# D: waveform (pulse width x polarity), near-electrode, SCL-ON
for pw in PULSE_WIDTHS_US:
    for cathodic in (True, False):
        trials.append(dict(stage="D_waveform", cfg="SCL-ON", ecc=80, phi=0,
                           orient_dev=0, amp_uA=2000, pw_us=pw, cathodic=cathodic))
# E: azimuth, ring (SCL-ON) vs needle (IntraCranial-TransCranial)
for cfg in ("SCL-ON", "IntraCranial-TransCranial"):
    for phi in PHIS:
        trials.append(dict(stage="E_azimuth", cfg=cfg, ecc=80, phi=phi,
                           orient_dev=0, amp_uA=2000, pw_us=400, cathodic=True))

print(f"[{cell_key}] {len(trials)} trials queued", flush=True)

# ---- load the cell once ----------------------------------------------------
sections = load_hoc_cell(cell_dir)
seg_um = get_segment_coordinates(sections, dx_um=1.0)
n_seg = len(seg_um)
morph = read_swc(cell_dir / "morphology" / swc_name)
local_axis = estimate_depth_axis(morph)
soma_um = morph.xyz_um[morph.types == 1].mean(axis=0)
print(f"[{cell_key}] cell loaded: {len(sections)} sections, {n_seg} segments", flush=True)

# ---- montages are loaded lazily and cached across trials that share one ---
montages: dict = {}


def get_montage(cfg: str) -> CachedMontage:
    if cfg not in montages:
        t0 = time.time()
        montages[cfg] = CachedMontage(cfg)
        print(f"[{cell_key}] loaded montage {cfg} ({time.time()-t0:.0f}s)", flush=True)
    return montages[cfg]


fieldnames = ["stage", "cfg", "cell", "ecc_deg", "phi_deg", "orient_dev_deg",
             "amp_uA", "pw_us", "cathodic_first", "unit_v_mean_VperA",
             "unit_v_relspread_pct", "n_spikes", "first_spike_ms",
             "peak_vm_dev_mV", "seconds"]
write_header = not OUT_CSV.exists()
f = open(OUT_CSV, "a", newline="")
writer = csv.DictWriter(f, fieldnames=fieldnames)
if write_header:
    writer.writeheader()

for i, t in enumerate(trials):
    t0 = time.time()
    m = get_montage(t["cfg"])
    anchor_vox = m.voxel_near(t["ecc"], t["phi"])
    radial = m.radial_direction(anchor_vox)
    target_axis = rotate_about_tangent(radial, t["orient_dev"])
    grid_crop, jmag_crop, origin = m.crop(anchor_vox, pad_vox=40.0)
    anchor_local = anchor_vox - origin
    target_um_local = anchor_local * m.dx * 1e6

    seg_vox_local = place_aligned(seg_um, local_axis, target_axis,
                                  target_um_local, pivot=soma_um) / (m.dx * 1e6)
    inside = np.all((seg_vox_local >= 0) &
                    (seg_vox_local <= np.array(grid_crop.shape) - 1), axis=1)
    if inside.sum() < n_seg:
        seg_vox_local = np.clip(seg_vox_local, 0, np.array(grid_crop.shape) - 1.001)

    unit_v = sample_node_grid(grid_crop, seg_vox_local)
    wave = biphasic_pulse_train(
        amp_A=t["amp_uA"] * 1e-6, pulse_width_s=t["pw_us"] * 1e-6,
        period_s=period_s, n_pulses=1, dt=dt_ms * 1e-3, ratio=1.0,
        delay_s=delay_s, cathodic_first=t["cathodic"])
    vmat_mV = build_v_matrix(unit_v, wave, unit_scale=1e3)

    run = run_extracellular(sections, vmat_mV, dt_ms=dt_ms, tstop_ms=tstop_ms,
                            record_segments=[0], spike_section_index=0)

    # baseline = v right before the pulse (after the intrinsic settling
    # transient has run its course), not v at t=0 -- see the module
    # docstring/settle_ms comment for why that distinction matters here.
    v = run.vm_mV[:, 0]
    base_idx = int(np.argmin(np.abs(run.t_ms - settle_ms)))
    baseline = v[base_idx]
    post = v[base_idx:]
    peak_dev = float(np.abs(post - baseline).max())

    spread_pct = float((unit_v.max() - unit_v.min()) / abs(unit_v.mean()) * 100) \
        if unit_v.mean() != 0 else float("nan")
    row = dict(
        stage=t["stage"], cfg=t["cfg"], cell=cell_key, ecc_deg=t["ecc"],
        phi_deg=t["phi"], orient_dev_deg=t["orient_dev"], amp_uA=t["amp_uA"],
        pw_us=t["pw_us"], cathodic_first=t["cathodic"],
        unit_v_mean_VperA=float(unit_v.mean()), unit_v_relspread_pct=spread_pct,
        n_spikes=len(run.spikes_ms),
        first_spike_ms=float(run.spikes_ms[0]) if len(run.spikes_ms) else "",
        peak_vm_dev_mV=peak_dev,
        seconds=round(time.time() - t0, 2))
    writer.writerow(row)
    f.flush()
    if (i + 1) % 10 == 0 or i == len(trials) - 1:
        print(f"[{cell_key}] {i+1}/{len(trials)} trials done", flush=True)

f.close()
print(f"[{cell_key}] wrote {OUT_CSV}")
