"""Model I/O round-trips, waveforms, coupling files, cache."""

import numpy as np
import pytest

from neuroam.materials import Material, MaterialLibrary
from neuroam.model import VoxelModel, node_name, parse_node_name
from neuroam.waveforms import (Waveform, biphasic_pulse_train, dc,
                               monophasic_pulse_train, sine, from_config)
from neuroam.coupling import (build_v_matrix, place_aligned, read_coordinates,
                              read_unit_field, read_v_file,
                              rotation_between_vectors, sample_node_grid,
                              transform_coordinates, write_coordinates,
                              write_unit_field, write_v_file)
from neuroam.cache import ResultCache, hash_inputs
from neuroam.fields import voxel_average, efield


def test_node_names():
    assert node_name(3, 737, 51) == "000307370051"
    assert parse_node_name("000307370051") == (3, 737, 51)


def test_legacy_roundtrip(tmp_path):
    lib = MaterialLibrary([Material.from_sigma(1, 1.5, name="vitreous"),
                           Material(id=2, rho=(0.5, 1.5, 0.5), name="aniso")])
    m = VoxelModel.empty((6, 5, 4), dx=2.5e-4, materials=lib, background=1,
                         name="rt")
    m.add_box(2, corner=(1, 1, 1), size=(2, 2, 2))
    m.add_electrode(shape="none", role="source", node=(3, 2, 0), material=101,
                    waveform="Cur1")
    m.materials.add(Material.isotropic_rho(101, 1e-7))
    m.ground_nodes = [(3, 2, 4)]

    in_path, model_path = m.save_legacy(tmp_path, "rt")
    m2 = VoxelModel.from_legacy(in_path)

    assert m2.world == m.world
    assert m2.dx == pytest.approx(m.dx)
    assert np.array_equal(m2.labels, m.labels)
    assert m2.sources[0].node == (3, 2, 0)
    assert m2.ground_nodes == [(3, 2, 4)]
    # anisotropic material survived
    assert m2.materials[2].rho == pytest.approx((0.5, 1.5, 0.5))


def test_model_layout_matches_legacy_loader(tmp_path):
    """Our writer/reader agree with AM_v10.2's reshape convention."""
    lib = MaterialLibrary([Material.from_sigma(0, 1.0)])
    m = VoxelModel.empty((3, 4, 5), dx=1e-3, materials=lib, background=0)
    # unique value per voxel to catch axis mixups
    m.labels[:] = np.arange(3 * 4 * 5).reshape(3, 4, 5)
    for v in np.unique(m.labels):
        lib.add(Material.from_sigma(int(v), 1.0))
    _, model_path = m.save_legacy(tmp_path, "layout")

    # legacy loader (verbatim from AM_v10.2_res_multi.py)
    n1, n2, n3 = 3, 4, 5
    m2d = np.loadtxt(model_path).astype(int)
    r, c = m2d.shape
    r = r // n3
    m3d = m2d.reshape(n3, r, c).transpose(2, 1, 0)
    assert np.array_equal(m3d, m.labels)


def test_cur_roundtrip(tmp_path):
    w = biphasic_pulse_train(1e-3, 1e-3, 10e-3, 3, dt=1e-4, ratio=4.0)
    p = w.to_cur(tmp_path / "w.cur")
    w2 = Waveform.from_cur(p)
    assert w2.dt == pytest.approx(w.dt)
    assert np.allclose(w2.samples, w.samples, atol=1e-9)


def test_waveform_charge_balance():
    scb = biphasic_pulse_train(1e-3, 1e-3, 10e-3, 2, dt=1e-5, ratio=1.0)
    acb = biphasic_pulse_train(1e-3, 1e-3, 10e-3, 2, dt=1e-5, ratio=4.0)
    assert abs(scb.charge_balance()) < 1e-9
    assert abs(acb.charge_balance()) < 1e-9
    mono = monophasic_pulse_train(1e-3, 1e-3, 10e-3, 2, dt=1e-5)
    assert mono.charge_balance() == pytest.approx(1.0)


def test_waveform_from_config():
    w = from_config({"type": "sine", "amp_A": 2e-3, "freq_hz": 100,
                     "n_cycles": 2, "dt_s": 1e-5})
    assert w.duration == pytest.approx(0.02)
    assert w.samples.max() == pytest.approx(2e-3, rel=1e-3)


def test_trilinear_sampling_exact():
    """Sampling reproduces a linear field exactly (trilinear property)."""
    nx = ny = nz = 6
    X, Y, Z = np.meshgrid(np.arange(nx + 1), np.arange(ny + 1),
                          np.arange(nz + 1), indexing="ij")
    grid = 2.0 * X - 3.0 * Y + 0.5 * Z + 1.0
    coords = np.array([[1.25, 2.5, 3.75], [0.1, 0.2, 0.3], [4.9, 4.9, 4.9]])
    vals = sample_node_grid(grid, coords)
    expect = 2.0 * coords[:, 0] - 3.0 * coords[:, 1] + 0.5 * coords[:, 2] + 1.0
    assert np.allclose(vals, expect, rtol=1e-12)


def test_coordinate_files_and_v_matrix(tmp_path):
    coords = np.array([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]])
    p = write_coordinates(tmp_path / "coords.txt", coords)
    c2 = read_coordinates(p)
    assert np.allclose(c2, coords)

    unit = np.array([0.5, -0.25])
    up = write_unit_field(tmp_path / "u.v", unit)
    assert np.allclose(read_unit_field(up), unit)

    w = dc(2e-3, dt=1e-4, n_steps=5)
    vmat = build_v_matrix(unit, w, unit_scale=1e3)   # V->mV at 1A per amp
    assert vmat.shape == (5, 2)
    assert vmat[0, 0] == pytest.approx(0.5 * 2e-3 * 1e3)
    vp = write_v_file(tmp_path / "m.v", vmat)
    assert np.allclose(read_v_file(vp), vmat)


def test_transform_coordinates_rigid():
    c = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    out = transform_coordinates(c, translate=(5, 5, 5),
                                rotate_deg=(0, 0, 90), pivot=(0, 0, 0))
    assert np.allclose(out[0], [5, 5, 5], atol=1e-12)
    assert np.allclose(out[1], [5, 6, 5], atol=1e-12)


def test_rotation_between_vectors_aligns_and_preserves_length():
    a = np.array([0.0, 0.0, 1.0])
    b = np.array([1.0, 1.0, 1.0])
    R = rotation_between_vectors(a, b)
    out = R @ a
    assert np.allclose(out, b / np.linalg.norm(b), atol=1e-10)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)   # a real rotation
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-10)


def test_rotation_between_vectors_identity_when_already_aligned():
    a = np.array([1.0, 0.0, 0.0])
    R = rotation_between_vectors(a, a)
    assert np.allclose(R, np.eye(3), atol=1e-10)


def test_rotation_between_vectors_antiparallel():
    a = np.array([0.0, 0.0, 1.0])
    b = np.array([0.0, 0.0, -1.0])
    R = rotation_between_vectors(a, b)
    assert np.allclose(R @ a, b, atol=1e-10)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)


def test_place_aligned_orients_and_translates():
    # a "cell" that is a short stick along local +z, pivoted at its own origin
    coords = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    target_point = np.array([10.0, 20.0, 30.0])
    target_axis = np.array([1.0, 0.0, 0.0])   # want local +z -> world +x
    out = place_aligned(coords, local_axis=(0, 0, 1), target_axis=target_axis,
                        target_point=target_point, pivot=(0, 0, 0))
    assert np.allclose(out[1], target_point, atol=1e-10)          # pivot landed exactly
    assert np.allclose(out[2] - out[1], [1, 0, 0], atol=1e-10)    # +local-z -> +world-x
    assert np.allclose(out[0] - out[1], [-1, 0, 0], atol=1e-10)


def test_cache_roundtrip(tmp_path):
    cache = ResultCache(root=tmp_path / "c")
    key = hash_inputs(np.arange(5), {"rtol": 1e-8})
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return {"v": np.arange(3.0)}

    a = cache.get_or_compute("basis", key, compute)
    b = cache.get_or_compute("basis", key, compute)
    assert calls["n"] == 1
    assert np.allclose(a["v"], b["v"])
    # different inputs -> different key
    assert hash_inputs(np.arange(5), {"rtol": 1e-6}) != key


def test_efield_uniform_gradient():
    """A linear potential gives a constant E field."""
    nx = ny = nz = 5
    X, Y, Z = np.meshgrid(np.arange(nx + 1), np.arange(ny + 1),
                          np.arange(nz + 1), indexing="ij")
    dx = 1e-3
    grid = -3.0 * X * dx          # V = -3 V/m * x
    Ex, Ey, Ez, Em = efield(grid, dx)
    assert np.allclose(Ex, 3.0, rtol=1e-12)
    assert np.allclose(Ey, 0.0, atol=1e-12)
    assert np.allclose(Em, 3.0, rtol=1e-12)
    va = voxel_average(grid)
    assert va.shape == (nx, ny, nz)
