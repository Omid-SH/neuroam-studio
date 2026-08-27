"""Run an existing lab model (.in/.model) through NeuroAM.

Usage:
    python examples/03_legacy_model.py path/to/Model.in [path/to/mesh.mrm]

Loads the legacy pair, assembles directly (no netlist), solves a unit-current
field per source, and writes fields + a .vof next to the outputs directory.
If a .mrm from the lab mesher is given, it is used for multiresolution
assembly instead of the uniform grid.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from neuroam.model import VoxelModel
from neuroam.assembly import assemble
from neuroam.solver import solve, unit_current_vector, kcl_report
from neuroam.fields import (node_grid, fill_hanging_nodes, voxel_average,
                            efield, write_vof, write_vavg)
from neuroam import viz


def main(in_file, mrm=None):
    m = VoxelModel.from_legacy(in_file)
    print(m.summary())
    out = Path(in_file).parent / f"neuroam_out_{m.name}"
    out.mkdir(exist_ok=True)

    system = assemble(m, mrm_path=mrm)
    print(f"assembled: {system.n:,} unknowns, {system.G.nnz:,} nonzeros")

    for src in system.source_rows:
        I = unit_current_vector(system, src)
        r = solve(system, I, method="cg", rtol=1e-8)
        print(f"source '{src}': {r.iterations} iters, residual {r.residual:.2e},"
              f" {r.seconds:.1f}s")
        print("  conservation:", kcl_report(system, r.v, I))

        grid = fill_hanging_nodes(node_grid(system, r.v), system)
        vavg = voxel_average(grid)
        _, _, _, Emag = efield(grid, m.dx)
        write_vof(out / f"{m.name}_{src}_unit.vof", system, r.v)
        write_vavg(out / f"{m.name}_{src}_unit.vavg", vavg)
        viz.save_slice(out / f"{m.name}_{src}_V.png", vavg, "z", None,
                       title=f"{src} unit-current potential", units="V")
        viz.save_slice(out / f"{m.name}_{src}_Emag.png", Emag, "z", None,
                       title=f"{src} |E|", cmap="magma", log=True, units="V/m")
    print(f"wrote outputs to {out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
