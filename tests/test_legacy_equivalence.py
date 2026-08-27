"""Equivalence with the legacy netlist toolchain.

1. Direct uniform assembly == matrix parsed from our netgen-equivalent .net.
2. .mrm path on an all-size-1 mesh == uniform path (identical matrices).
3. Resistor values match the real netgen output convention:
   R = 4*rho*s_par / (s_perp1*s_perp2*dx) — checked with the exact numbers
   observed in the lab's QuarterHead netlist (rho=5000, dx=0.25mm:
   size-1 -> 8e7 ohm, size-2 cube -> 4e7 ohm).
4. Multiresolution solution ~= uniform solution in a homogeneous medium.
"""

import numpy as np
import pytest

from neuroam.assembly import assemble_uniform, assemble_mrm
from neuroam.mesh import uniform_records, write_mrm
from neuroam.netlist import write_net, read_net
from neuroam.model import VoxelModel
from neuroam.materials import Material, MaterialLibrary
from neuroam.solver import solve, unit_current_vector
from neuroam.fields import node_grid


def small_model():
    lib = MaterialLibrary([Material.from_sigma(1, 0.5, name="tissue"),
                           Material.from_sigma(2, 2.0, name="csf")])
    m = VoxelModel.empty((8, 8, 8), dx=5e-4, materials=lib, background=1,
                         name="tiny")
    m.add_sphere(2, center=(4, 4, 4), radius=2.5)
    m.add_electrode(shape="none", role="source", node=(4, 4, 0), material=101,
                    waveform="stim")
    m.materials.add(Material.isotropic_rho(101, 1e-7))
    m.ground_nodes = [(4, 4, 8)]
    return m


def _canon(G):
    G = G.tocoo()
    order = np.lexsort((G.col, G.row))
    return G.row[order], G.col[order], G.data[order]


def test_netgen_resistor_values():
    """Reproduce the exact resistor values seen in the lab's netlists."""
    lib = MaterialLibrary([Material.isotropic_rho(143, 5000.0)])
    m = VoxelModel.empty((4, 4, 4), dx=0.25e-3, materials=lib, background=143)
    m.add_electrode(shape="none", role="source", node=(0, 0, 0), material=101,
                    waveform="s")
    m.materials.add(Material.isotropic_rho(101, 1e-7))
    m.ground_nodes = [(4, 4, 4)]

    import io
    from neuroam.mesh import uniform_records
    rec1 = uniform_records(m)[:1]                       # one size-1 voxel
    rec2 = np.array([[0, 0, 0, 2, 2, 2, 143]])          # one size-2 voxel

    import tempfile, os
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        p1 = Path(td) / "a.net"
        write_net(m, rec1, p1)
        txt = p1.read_text()
        assert "8e+07" in txt or "8e+007" in txt or "8e+7" in txt \
            or "8e7" in txt.replace("+0", "").replace("+", ""), txt
        p2 = Path(td) / "b.net"
        write_net(m, rec2, p2)
        assert "4e+07" in p2.read_text()


def test_uniform_equals_netlist_roundtrip(tmp_path):
    m = small_model()
    sys_direct = assemble_uniform(m)

    rec = uniform_records(m)
    net = write_net(m, rec, tmp_path / "tiny.net")
    sys_net = read_net(net, m)

    assert sys_direct.n == sys_net.n
    r1, c1, d1 = _canon(sys_direct.G)
    r2, c2, d2 = _canon(sys_net.G)
    # node orderings coincide (both sorted by packed coordinate)
    assert np.array_equal(r1, r2) and np.array_equal(c1, c2)
    assert np.allclose(d1, d2, rtol=1e-9)


def test_mrm_size1_equals_uniform(tmp_path):
    m = small_model()
    sys_u = assemble_uniform(m)
    rec = uniform_records(m)
    mrm = write_mrm(tmp_path / "tiny.mrm", rec)
    sys_m = assemble_mrm(m, np.loadtxt(mrm, usecols=(1, 2, 3, 4, 5, 6, 7),
                                       dtype=np.int64, ndmin=2))
    assert sys_u.n == sys_m.n
    r1, c1, d1 = _canon(sys_u.G)
    r2, c2, d2 = _canon(sys_m.G)
    assert np.array_equal(r1, r2) and np.array_equal(c1, c2)
    assert np.allclose(d1, d2, rtol=1e-12)


def test_multires_matches_uniform_homogeneous():
    """Coarsened far region reproduces the uniform solution near the source."""
    lib = MaterialLibrary([Material.from_sigma(1, 1.0)])
    n = 16
    m = VoxelModel.empty((n, n, n), dx=1e-4, materials=lib, background=1)
    m.add_electrode(shape="none", role="source", node=(n // 2, n // 2, n // 2),
                    material=101, waveform="s")
    m.materials.add(Material.isotropic_rho(101, 1e-7))
    for gx in (0, n):
        for gy in (0, n):
            for gz in (0, n):
                m.ground_nodes.append((gx, gy, gz))

    sys_u = assemble_uniform(m)
    ru = solve(sys_u, unit_current_vector(sys_u, "s"), method="direct")
    gu = node_grid(sys_u, ru.v)

    # multires: unit voxels in the central 8^3, 2x2x2 clusters elsewhere
    recs = []
    for x in range(0, n, 2):
        for y in range(0, n, 2):
            for z in range(0, n, 2):
                inner = (4 <= x < 12) and (4 <= y < 12) and (4 <= z < 12)
                if inner:
                    for ddx in range(2):
                        for ddy in range(2):
                            for ddz in range(2):
                                recs.append([x + ddx, y + ddy, z + ddz,
                                             1, 1, 1, 1])
                else:
                    recs.append([x, y, z, 2, 2, 2, 1])
    recs = np.asarray(recs, dtype=np.int64)
    sys_m = assemble_mrm(m, recs)
    rm = solve(sys_m, unit_current_vector(sys_m, "s"), method="direct")
    gm = node_grid(sys_m, rm.v)

    c = n // 2
    # Corner-point grounding adds a mesh-dependent constriction offset to the
    # whole solution, so compare potential DIFFERENCES within the fine zone
    # (the offset cancels; near-source physics lives on the shared fine grid).
    ref = (c + 3, c, c)
    pts = [(c + 1, c, c), (c + 2, c, c), (c, c + 2, c), (c, c, c - 2)]
    for p in pts:
        du = gu[p] - gu[ref]
        dm = gm[p] - gm[ref]
        assert abs(dm - du) / abs(du) < 0.05, (p, du, dm)
