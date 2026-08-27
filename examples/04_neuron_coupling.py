"""Full multiscale demo: tissue field -> extracellular drive -> NEURON spikes.

Builds a small tissue volume with a surface electrode, solves the unit-current
field, places a ball-and-stick neuron in the volume, samples the field along
its compartments, scales it with a biphasic waveform, drives NEURON's
``extracellular`` mechanism, and finds the activation threshold by bisection.

Run:  python examples/04_neuron_coupling.py     (requires `pip install neuron`)
Writes PNGs into examples/out/coupling/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from neuroam.materials import Material, MaterialLibrary
from neuroam.model import VoxelModel
from neuroam.assembly import assemble_uniform
from neuroam.solver import solve, unit_current_vector
from neuroam.fields import node_grid
from neuroam.coupling import sample_node_grid, build_v_matrix
from neuroam.waveforms import biphasic_pulse_train
from neuroam.neuron_link import (neuron_available, make_ball_and_stick,
                                 get_segment_coordinates, run_extracellular)
from neuroam import viz

OUT = Path(__file__).parent / "out" / "coupling"
OUT.mkdir(parents=True, exist_ok=True)

if not neuron_available():
    sys.exit("NEURON is not installed: pip install neuron")

# ---- 1. tissue + electrode field (unit current) ---------------------------
n = 40
dx = 5e-5                      # 50 um voxels -> 2 mm cube
lib = MaterialLibrary([Material.from_sigma(1, 0.3, name="tissue")])
m = VoxelModel.empty((n, n, n), dx=dx, materials=lib, background=1)
m.add_electrode(shape="box", role="source", material=101, waveform="stim",
                corner=(18, 18, 0), size=(4, 4, 1), node=(20, 20, 0))
m.add_electrode(shape="box", role="ground", material=100,
                corner=(18, 18, n - 1), size=(4, 4, 1), node=(20, 20, n))

system = assemble_uniform(m)
res = solve(system, unit_current_vector(system, "stim"), method="cg", rtol=1e-9)
grid = node_grid(system, res.v)
print(f"field solved: {system.n:,} unknowns, residual {res.residual:.1e}")

# ---- 2. place a neuron and sample the unit field --------------------------
sections = make_ball_and_stick(axon_len_um=1500.0, nseg=61)
coords_um = get_segment_coordinates(sections, dx_um=1.0)   # um units first
# place: axon runs along +x at 150 um above the electrode, centered
coords_vox = coords_um / (dx * 1e6)
coords_vox[:, 0] += 20 - coords_vox[:, 0].mean()
coords_vox[:, 1] += 20 - coords_vox[:, 1].mean()
coords_vox[:, 2] = coords_vox[:, 2] - coords_vox[:, 2].min() + 150.0 / (dx * 1e6)

unit_v = sample_node_grid(grid, coords_vox)     # volts per amp
print(f"unit field along neuron: {unit_v.min():.3g} .. {unit_v.max():.3g} V/A")

# ---- 3. waveform + threshold search ---------------------------------------
dt_ms = 0.025
wave = biphasic_pulse_train(amp_A=1.0, pulse_width_s=0.2e-3, period_s=5e-3,
                            n_pulses=1, dt=dt_ms * 1e-3, ratio=1.0,
                            delay_s=1e-3)


def spikes_at(amp_A: float) -> int:
    wf = biphasic_pulse_train(amp_A=amp_A, pulse_width_s=0.2e-3,
                              period_s=5e-3, n_pulses=1, dt=dt_ms * 1e-3,
                              ratio=1.0, delay_s=1e-3)
    vmat_mV = build_v_matrix(unit_v, wf, unit_scale=1e3)
    secs = make_ball_and_stick(axon_len_um=1500.0, nseg=61)
    run = run_extracellular(secs, vmat_mV, dt_ms=dt_ms, tstop_ms=8.0,
                            record_segments=[0, len(unit_v) - 1],
                            spike_section_index=len(unit_v) - 1)
    return len(run.spikes_ms), run


lo, hi = 1e-6, 5e-3
n_lo, _ = spikes_at(lo)
n_hi, _ = spikes_at(hi)
assert n_lo == 0 and n_hi >= 1, (n_lo, n_hi)
for _ in range(12):
    mid = np.sqrt(lo * hi)
    nsp, _ = spikes_at(mid)
    if nsp >= 1:
        hi = mid
    else:
        lo = mid
threshold = hi
print(f"activation threshold ~= {threshold*1e6:.1f} uA "
      f"(200 us cathodic-first biphasic, axon 150 um above electrode)")

# ---- 4. plots -------------------------------------------------------------
_, run_thr = spikes_at(threshold * 1.2)
viz.save_vm_traces(OUT / "vm_at_1p2x_threshold.png", run_thr.t_ms,
                   run_thr.vm_mV, labels=["soma", "distal axon"],
                   title=f"Vm at 1.2x threshold ({1.2*threshold*1e6:.0f} uA)")
wf = biphasic_pulse_train(amp_A=threshold * 1.2, pulse_width_s=0.2e-3,
                          period_s=5e-3, n_pulses=1, dt=dt_ms * 1e-3,
                          ratio=1.0, delay_s=1e-3)
viz.save_waveform(OUT / "stimulus.png", wf, "stimulus at 1.2x threshold")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7, 3.2), dpi=130)
ax.plot(np.arange(len(unit_v)), unit_v * threshold * 1e3, "o-", ms=3)
ax.set_xlabel("compartment index (soma -> distal axon)")
ax.set_ylabel("Ve at threshold (mV)")
ax.set_title("extracellular potential along the neuron")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "ve_profile.png")
print(f"wrote plots to {OUT}")
