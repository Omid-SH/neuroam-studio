"""Dose-response figure: single pulse (examples/31) vs. 20 Hz pulse train
(examples/32), for both cells at both tested locations -- the direct visual
counterpart to docs/STIMULATION_PARAMETER_STUDY.md section 8.4's table.

No NEURON needed -- reads the already-written sweep CSVs and plots with
``neuroam.viz.save_dose_response``. Four panels (D1 @ ecc 0, D1 @ ecc 80,
A2i @ ecc 0, A2i @ ecc 80), each showing both waveforms' peak deviation vs.
amplitude on a log-x axis, with a star marking the lowest tested amplitude
that produced a spike (10 mA, where it happened at all).

Run:  python examples/35_dose_response_figure.py
Requires: examples/out/sweep/{d1,a2i}_sweep.csv (examples/31) and
         examples/out/sweep/{d1,a2i}_pulsetrain.csv (examples/32).
Writes: examples/out/pulsetrain_figures/dose_response_comparison.png
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from neuroam import viz

HERE = Path(__file__).resolve().parent
SWEEP = HERE / "out" / "sweep"
OUT = HERE / "out" / "pulsetrain_figures"
OUT.mkdir(parents=True, exist_ok=True)


def read_single_pulse(cell: str, ecc: int):
    """Stage B (amplitude sweep), SCL-ON, correct orientation, cathodic-first,
    400 us -- examples/31."""
    amps, devs, spike_amp = [], [], None
    with open(SWEEP / f"{cell}_sweep.csv", newline="") as f:
        for row in csv.DictReader(f):
            if (row["stage"] == "B_amplitude" and row["cfg"] == "SCL-ON"
                    and int(row["ecc_deg"]) == ecc):
                amps.append(float(row["amp_uA"]))
                devs.append(float(row["peak_vm_dev_mV"]))
                if int(row["n_spikes"]) > 0 and spike_amp is None:
                    spike_amp = float(row["amp_uA"])
    order = sorted(range(len(amps)), key=lambda i: amps[i])
    return [amps[i] for i in order], [devs[i] for i in order], spike_amp


def read_pulse_train(cell: str, ecc: int):
    """examples/32: 20 Hz/100 ms train, control-subtracted."""
    amps, devs, spike_amp = [], [], None
    with open(SWEEP / f"{cell}_pulsetrain.csv", newline="") as f:
        for row in csv.DictReader(f):
            if int(row["ecc_deg"]) == ecc:
                amps.append(float(row["amp_uA"]))
                devs.append(float(row["peak_dev_vs_control_mV"]))
                if int(row["n_spikes"]) > 0 and spike_amp is None:
                    spike_amp = float(row["amp_uA"])
    order = sorted(range(len(amps)), key=lambda i: amps[i])
    return [amps[i] for i in order], [devs[i] for i in order], spike_amp


fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=130)
panels = [("d1", 0, axes[0, 0]), ("d1", 80, axes[0, 1]),
         ("a2i", 0, axes[1, 0]), ("a2i", 80, axes[1, 1])]

for cell, ecc, ax in panels:
    sp_amp, sp_dev, sp_spike = read_single_pulse(cell, ecc)
    tr_amp, tr_dev, tr_spike = read_pulse_train(cell, ecc)
    spike_markers = {}
    if sp_spike is not None:
        spike_markers["single pulse (examples/31)"] = sp_spike
    if tr_spike is not None:
        spike_markers["20 Hz train (examples/32)"] = tr_spike
    viz.save_dose_response(
        None,
        {"single pulse (examples/31)": (sp_amp, sp_dev),
         "20 Hz train (examples/32)": (tr_amp, tr_dev)},
        title=f"{cell.upper()} @ ecc {ecc} deg",
        xlabel="amplitude (uA)", ylabel="peak deviation (mV)",
        ylog=True, spike_markers=spike_markers, ax=ax)

fig.suptitle("Single pulse (examples/31) vs. 20 Hz/100 ms pulse train (examples/32): "
             "dose-response, SCL-ON, correct orientation, cathodic-first\n"
             "log-log -- stars mark the lowest spiking amplitude tested",
             fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.93))
out_path = OUT / "dose_response_comparison.png"
fig.savefig(out_path)
plt.close(fig)
print(f"wrote {out_path}")
