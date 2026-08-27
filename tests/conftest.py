import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from neuroam.materials import Material, MaterialLibrary
from neuroam.model import VoxelModel


@pytest.fixture(scope="session")
def saline_cube():
    """Homogeneous conductive cube (sigma = 1 S/m), point source at the
    exact center node (even world size), ground at the 8 domain corners."""
    lib = MaterialLibrary([Material.from_sigma(1, 1.0, name="saline")])
    n = 32
    m = VoxelModel.empty((n, n, n), dx=1e-4, materials=lib, background=1,
                         name="saline_cube")
    c = n // 2
    m.add_electrode(shape="none", role="source", material=101,
                    waveform="stim", node=(c, c, c))
    # ground: all 8 corners of the domain (far reference)
    for gx in (0, n):
        for gy in (0, n):
            for gz in (0, n):
                m.ground_nodes.append((gx, gy, gz))
    return m


@pytest.fixture(scope="session")
def saline_solution(saline_cube):
    """(system, unit-current solution) for the saline cube, solved once."""
    from neuroam.assembly import assemble_uniform
    from neuroam.solver import solve, unit_current_vector

    system = assemble_uniform(saline_cube)
    I = unit_current_vector(system, "stim")
    res = solve(system, I, method="direct")
    return system, I, res


@pytest.fixture
def layered_slab():
    """Two-layer slab between full-face plate electrodes along x.

    Layer A: sigma=1 S/m, thickness 10; layer B: sigma=0.25 S/m, thickness 10.
    1D current flow -> exact series-resistance solution.
    """
    lib = MaterialLibrary([Material.from_sigma(1, 1.0, name="A"),
                           Material.from_sigma(2, 0.25, name="B")])
    nx, ny, nz = 20, 6, 6
    m = VoxelModel.empty((nx, ny, nz), dx=1e-3, materials=lib, background=1,
                         name="slab")
    m.add_slab(2, axis="x", start=10, thickness=10)
    return m
