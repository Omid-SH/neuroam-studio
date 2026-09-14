"""SWC I/O, tree connectivity, and Vm-onto-morphology nearest-neighbor mapping."""

import numpy as np
import pytest

from neuroam.morphology import (SWCMorphology, read_swc, write_swc,
                                nearest_values, synthetic_branching_tree,
                                estimate_depth_axis)


def test_swc_round_trip(tmp_path):
    m = SWCMorphology(
        ids=np.array([1, 2, 3]), types=np.array([1, 3, 3]),
        xyz_um=np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]]),
        radius_um=np.array([5.0, 1.0, 1.0]), parent=np.array([-1, 1, 2]))
    path = write_swc(tmp_path / "cell.swc", m)
    back = read_swc(path)
    assert np.array_equal(back.ids, m.ids)
    assert np.array_equal(back.parent, m.parent)
    assert np.allclose(back.xyz_um, m.xyz_um)
    assert np.allclose(back.radius_um, m.radius_um)


def test_swc_skips_comments_and_blank_lines(tmp_path):
    text = "# a comment\n\n1 1 0 0 0 5 -1\n2 3 1 0 0 1 1\n"
    p = tmp_path / "c.swc"
    p.write_text(text)
    m = read_swc(p)
    assert len(m) == 2
    assert m.parent.tolist() == [-1, 1]


def test_edges_is_a_tree():
    m = synthetic_branching_tree(n_generations=3, branching=3, seed=0)
    e = m.edges()
    assert e.shape[1] == 2
    assert e.shape[0] == len(m) - 1        # a tree: n-1 edges
    # single connected component
    import scipy.sparse as sp
    n = len(m)
    A = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    A = A + A.T
    ncomp, _ = sp.csgraph.connected_components(A, directed=False)
    assert ncomp == 1


def test_synthetic_tree_deterministic_given_seed():
    a = synthetic_branching_tree(seed=42)
    b = synthetic_branching_tree(seed=42)
    assert np.array_equal(a.xyz_um, b.xyz_um)
    assert np.array_equal(a.parent, b.parent)


def test_recentered_moves_root_to_target():
    m = synthetic_branching_tree(seed=1)
    target = np.array([10.0, -5.0, 3.0])
    out = m.recentered(at=target)
    assert np.allclose(out[0], target)
    # rigid translation: pairwise distances preserved
    d0 = np.linalg.norm(m.xyz_um - m.xyz_um[0], axis=1)
    d1 = np.linalg.norm(out - out[0], axis=1)
    assert np.allclose(d0, d1)


def test_estimate_depth_axis_recovers_known_orientation():
    # a flat/laminar synthetic cell: soma at the origin, dendrites spread
    # widely in x/y but offset along +z -- thin (small variance) along z,
    # wide along x/y, exactly the shape a real RGC's soma+dendrites have.
    rng = np.random.default_rng(0)
    n = 200
    xy = rng.normal(scale=25.0, size=(n, 2))
    dend_xyz = np.column_stack([xy, 8.0 + rng.normal(scale=0.3, size=n)])
    xyz = np.vstack([[0.0, 0.0, 0.0], dend_xyz])
    types = np.array([1] + [3] * n)
    morph = SWCMorphology(ids=np.arange(1, n + 2), types=types, xyz_um=xyz,
                          radius_um=np.ones(n + 1), parent=np.array([-1] + [1] * n))

    axis = estimate_depth_axis(morph)
    assert axis == pytest.approx(np.array([0.0, 0.0, 1.0]), abs=0.05)


def test_estimate_depth_axis_on_real_rgc_cells():
    from pathlib import Path
    samples = Path(__file__).resolve().parents[1] / "samples/neuron_models"
    for path, expect_sign in [(samples / "rgc_d1/morphology/8.swc", 1),
                              (samples / "rgc_a2i/morphology/9.swc", 1)]:
        if not path.exists():
            pytest.skip(f"{path} not present")
        morph = read_swc(path)
        axis = estimate_depth_axis(morph)
        soma_c = morph.xyz_um[morph.types == 1].mean(axis=0)
        dend_c = morph.xyz_um[morph.types == 3].mean(axis=0)
        # by construction: axis must point from soma toward dendrites
        assert np.dot(dend_c - soma_c, axis) > 0


def test_nearest_values_maps_by_proximity():
    sample_pts = np.array([[0., 0., 0.], [10., 0., 0.]])
    sample_vals = np.array([-65.0, 20.0])
    query_pts = np.array([[0.1, 0, 0], [9.9, 0, 0], [5.0, 0, 0]])
    out = nearest_values(query_pts, sample_pts, sample_vals)
    assert out[0] == pytest.approx(-65.0)
    assert out[1] == pytest.approx(20.0)
    assert out[2] in (-65.0, 20.0)          # midpoint: nearest of the two
