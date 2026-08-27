"""Analytical validation: point source, layered slab, current conservation."""

import numpy as np
import pytest

from neuroam.assembly import assemble_uniform
from neuroam.solver import solve, unit_current_vector, kcl_report
from neuroam.fields import node_grid, efield, voxel_average
from neuroam.model import VoxelModel
from neuroam.materials import Material, MaterialLibrary, METAL_RHO


def test_point_source_near_field(saline_cube, saline_solution):
    """Potential differences follow rho*I/(4*pi) * (1/r1 - 1/r2).

    Differences are used because the finite corner-ground reference adds a
    constant offset to the whole solution.
    """
    system, I, res = saline_solution
    grid = node_grid(system, res.v)
    m = saline_cube
    c = m.world[0] // 2
    dx = m.dx
    rho = 1.0

    r_ref = 8
    v_ref = grid[c + r_ref, c, c]
    errs = []
    for k in range(3, 8):
        dv_num = grid[c + k, c, c] - v_ref
        dv_ana = rho / (4 * np.pi) * (1.0 / (k * dx) - 1.0 / (r_ref * dx))
        errs.append(abs(dv_num - dv_ana) / dv_ana)
    assert np.median(errs) < 0.05, f"median near-field error {np.median(errs):.3f}"
    assert max(errs) < 0.10, f"max near-field error {max(errs):.3f}"


def test_point_source_isotropy(saline_cube, saline_solution):
    """Same-distance nodes along +/-x, +/-y, +/-z see the same potential."""
    system, I, res = saline_solution
    grid = node_grid(system, res.v)
    c = saline_cube.world[0] // 2
    k = 5
    vals = [grid[c + k, c, c], grid[c - k, c, c], grid[c, c + k, c],
            grid[c, c - k, c], grid[c, c, c + k], grid[c, c, c - k]]
    assert np.ptp(vals) / np.mean(vals) < 1e-9


def test_current_conservation(saline_solution):
    system, I, res = saline_solution
    rep = kcl_report(system, res.v, I)
    assert rep["relative_residual"] < 1e-9
    # dissipated power I^T V must be positive
    assert float(I @ res.v) > 0


def test_layered_slab_exact(layered_slab):
    """1D two-layer conduction matches series-resistance voltage division."""
    m = layered_slab
    nx, ny, nz = m.world
    m.add_slab(101, axis="x", start=0, thickness=1)
    m.add_slab(100, axis="x", start=nx - 1, thickness=1)
    m.materials.add(Material.isotropic_rho(101, METAL_RHO, name="src plate"))
    m.materials.add(Material.isotropic_rho(100, METAL_RHO, name="gnd plate"))
    m.sources = []
    m.ground_nodes = []
    m.add_electrode(shape="none", role="source", node=(0, ny // 2, nz // 2),
                    material=101, waveform="stim")
    m.add_electrode(shape="none", role="ground", node=(nx, ny // 2, nz // 2),
                    material=100)

    system = assemble_uniform(m)
    I_amp = 1e-3
    I = unit_current_vector(system, "stim") * I_amp
    r = solve(system, I, method="direct")
    grid = node_grid(system, r.v)

    dx = m.dx
    A = (ny * dx) * (nz * dx)
    n_a = 10 - 1                # material A voxels between plate and interface
    n_b = (nx - 1) - 10         # material B voxels
    R_a = (1.0 / 1.0) * n_a * dx / A
    R_b = (1.0 / 0.25) * n_b * dx / A

    v0 = grid[1, ny // 2, nz // 2]      # after source plate
    vmid = grid[10, ny // 2, nz // 2]   # at layer interface
    vend = grid[nx - 1, ny // 2, nz // 2]

    v_total = I_amp * (R_a + R_b)
    assert abs((v0 - vend) - v_total) / v_total < 1e-6
    assert abs((vmid - vend) - I_amp * R_b) / (I_amp * R_b) < 1e-6
    # plate cross-sections equipotential to within metal's tiny resistivity
    assert np.ptp(grid[1, :, :]) < 1e-6 * abs(v0)


def test_anisotropic_material():
    """Anisotropy: conduction along y uses rho_y."""
    lib = MaterialLibrary([Material(id=1, rho=(1.0, 4.0, 1.0), name="aniso")])
    nx, ny, nz = 4, 12, 4
    m = VoxelModel.empty((nx, ny, nz), dx=1e-3, materials=lib, background=1)
    m.add_electrode(shape="none", role="source", node=(2, 0, 2), material=101,
                    waveform="s")
    m.materials.add(Material.isotropic_rho(101, 1e-7))
    m.ground_nodes = [(2, ny, 2)]
    system = assemble_uniform(m)
    r = solve(system, unit_current_vector(system, "s") * 1e-3, method="direct")
    grid = node_grid(system, r.v)
    drop_aniso = grid[2, 0, 2] - grid[2, ny, 2]

    lib2 = MaterialLibrary([Material(id=1, rho=(1.0, 1.0, 1.0))])
    m2 = VoxelModel.empty((nx, ny, nz), dx=1e-3, materials=lib2, background=1)
    m2.sources = list(m.sources)
    m2.ground_nodes = list(m.ground_nodes)
    system2 = assemble_uniform(m2)
    r2 = solve(system2, unit_current_vector(system2, "s") * 1e-3, method="direct")
    g2 = node_grid(system2, r2.v)
    drop_iso = g2[2, 0, 2] - g2[2, ny, 2]

    assert drop_aniso > 2.0 * drop_iso


def test_iterative_matches_direct(saline_solution):
    system, I, rd = saline_solution
    rc = solve(system, I, method="cg", rtol=1e-10)
    denom = np.linalg.norm(rd.v)
    assert np.linalg.norm(rc.v - rd.v) / denom < 1e-6
