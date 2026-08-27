"""Analytical validation demo: point source in saline vs V = rho*I/(4*pi*r).

Run:  python examples/01_point_source_validation.py
Writes: examples/out/point_source_validation.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from neuroam.materials import Material, MaterialLibrary
from neuroam.model import VoxelModel
from neuroam.assembly import assemble_uniform
from neuroam.solver import solve, unit_current_vector
from neuroam.fields import node_grid

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

n = 60
dx = 1e-4          # 100 um voxels
rho = 1.0          # 1 ohm*m (sigma = 1 S/m)
I_amp = 1e-3       # 1 mA

lib = MaterialLibrary([Material.isotropic_rho(1, rho, name="saline")])
m = VoxelModel.empty((n, n, n), dx=dx, materials=lib, background=1)
c = n // 2
m.add_electrode(shape="none", role="source", node=(c, c, c), material=101,
                waveform="stim")
m.materials.add(Material.isotropic_rho(101, 1e-7))
for gx in (0, n):
    for gy in (0, n):
        for gz in (0, n):
            m.ground_nodes.append((gx, gy, gz))

print("assembling ...")
system = assemble_uniform(m)
print(f"  {system.n:,} unknowns")
res = solve(system, unit_current_vector(system, "stim") * I_amp, method="cg",
            rtol=1e-10)
print(f"  solved: {res.iterations} iters, residual {res.residual:.2e}, "
      f"{res.seconds:.1f}s")
grid = node_grid(system, res.v)

ks = np.arange(2, 13)
r_ref = 18
v_ref_num = grid[c + r_ref, c, c]
v_num = np.array([grid[c + k, c, c] for k in ks]) - v_ref_num
v_ana = rho * I_amp / (4 * np.pi) * (1 / (ks * dx) - 1 / (r_ref * dx))
err = np.abs(v_num - v_ana) / v_ana

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=130)
ax1.plot(ks * dx * 1e3, v_ana * 1e3, "k-", label="analytic ρI/4πr")
ax1.plot(ks * dx * 1e3, v_num * 1e3, "o", ms=4, label="NeuroAM")
ax1.set_xlabel("distance (mm)"); ax1.set_ylabel("V - V(ref) (mV)")
ax1.legend(); ax1.grid(alpha=0.3); ax1.set_title("point source in saline")
ax2.semilogy(ks * dx * 1e3, err * 100, "o-", ms=4)
ax2.set_xlabel("distance (mm)"); ax2.set_ylabel("relative error (%)")
ax2.grid(alpha=0.3); ax2.set_title(f"median error {np.median(err)*100:.2f}%")
fig.tight_layout()
fig.savefig(OUT / "point_source_validation.png")
print(f"median error {np.median(err)*100:.2f}%, wrote "
      f"{OUT/'point_source_validation.png'}")
