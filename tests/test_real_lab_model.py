"""Validation against a real lab model (Sph_80 from the C++ AM distribution).

Runs only when the environment variable ``NEUROAM_REFDATA`` points to a
directory containing ``Sph_80.in``, ``Sph_80.model`` and ``Sph_80.mrm``
(e.g. ``.../Admittance Method/C++ AM/ftdss2``).

Checks:
- the legacy ``.in``/``.model`` pair loads with correct world/materials/nodes;
- the mesher's real ``.mrm`` covers exactly the defined-material voxels;
- direct uniform assembly (with void handling) and assembly from the lab's
  ``.mrm`` produce the *identical* admittance system;
- the system solves with a conservative residual.
"""

import os
from pathlib import Path

import numpy as np
import pytest

REF = os.environ.get("NEUROAM_REFDATA")
pytestmark = pytest.mark.skipif(
    not (REF and (Path(REF) / "Sph_80.in").exists()),
    reason="NEUROAM_REFDATA with Sph_80 files not available")


@pytest.fixture(scope="module")
def sph80():
    from neuroam.model import VoxelModel
    return VoxelModel.from_legacy(Path(REF) / "Sph_80.in")


def test_load_legacy(sph80):
    assert sph80.world == (80, 80, 80)
    assert sph80.dx == pytest.approx(50e-6)
    assert 1 in sph80.materials and 4 in sph80.materials
    assert sph80.sources[0].node == (40, 40, 40)
    assert (40, 40, 79) in sph80.ground_nodes


def test_mrm_covers_defined_voxels(sph80):
    from neuroam.assembly import read_mrm
    rec = read_mrm(Path(REF) / "Sph_80.mrm")
    assert (rec[:, 3:6] == 1).all()          # maximumsize 1 -> all unit voxels
    n_defined = int(np.isin(sph80.labels, [1, 4]).sum())
    assert len(rec) == n_defined


def test_uniform_void_equals_lab_mrm(sph80):
    from neuroam.assembly import assemble_uniform, assemble_mrm, read_mrm

    sys_u = assemble_uniform(sph80)                     # void handling for label 0
    rec = read_mrm(Path(REF) / "Sph_80.mrm")
    sys_m = assemble_mrm(sph80, rec)

    assert sys_u.n == sys_m.n
    a = sys_u.G.tocoo()
    b = sys_m.G.tocoo()
    oa = np.lexsort((a.col, a.row))
    ob = np.lexsort((b.col, b.row))
    assert np.array_equal(a.row[oa], b.row[ob])
    assert np.array_equal(a.col[oa], b.col[ob])
    assert np.allclose(a.data[oa], b.data[ob], rtol=1e-12)


def test_solve_sph80(sph80):
    from neuroam.assembly import assemble_uniform
    from neuroam.solver import solve, unit_current_vector, kcl_report

    system = assemble_uniform(sph80)
    I = unit_current_vector(system, "cur_25us") * 1e-5   # 10 uA
    r = solve(system, I, method="cg", rtol=1e-8)
    rep = kcl_report(system, r.v, I)
    assert rep["relative_residual"] < 1e-6
    # potential should decay away from the source
    from neuroam.fields import node_grid
    g = node_grid(system, r.v)
    assert g[40, 40, 40] > g[40, 40, 60] > 0
