"""Electrode rasterization, grid-aware authoring, fitting, and placement."""

import numpy as np
import pytest

from neuroam import electrodes as el
from neuroam.materials import Material, MaterialLibrary
from neuroam.model import VoxelModel


def test_sphere_and_band_volumes():
    shape = (60, 60, 60)
    s = el.sphere(shape, (30, 30, 30), 10.0)
    # volume within a few percent of 4/3 pi r^3
    assert abs(s.sum() / (4 / 3 * np.pi * 10 ** 3) - 1) < 0.03
    shell = el.spherical_band(shape, (30, 30, 30), 8.0, 10.0)
    assert shell.sum() == pytest.approx(
        4 / 3 * np.pi * (10 ** 3 - 8 ** 3), rel=0.05)
    assert not (shell & el.sphere(shape, (30, 30, 30), 7.9)).any()


def test_spherical_band_cap_vs_ring():
    shape = (60, 60, 60)
    cap = el.spherical_band(shape, (30, 30, 30), 9, 10, axis=(0, 0, 1),
                            theta_min_deg=0, theta_max_deg=30)
    ring = el.spherical_band(shape, (30, 30, 30), 9, 10, axis=(0, 0, 1),
                             theta_min_deg=80, theta_max_deg=100)
    assert cap.any() and ring.any()
    assert not (cap & ring).any()
    # the cap sits on the +z pole; the ring straddles the equator
    assert np.argwhere(cap)[:, 2].mean() > 36
    assert abs(np.argwhere(ring)[:, 2].mean() - 30) < 1.5


def test_tube_sections_and_length():
    shape = (40, 40, 60)
    path = [[20, 20, 5], [20, 20, 55]]
    rnd = el.tube(shape, path, 3.0)
    sq = el.tube(shape, path, 3.0, section="square")
    assert sq.sum() > rnd.sum()                    # square circumscribes round
    assert (rnd & ~sq).sum() == 0
    # cross-section of the square tube is (2r+1)-ish per slice, uniform in z
    counts = [sq[:, :, z].sum() for z in range(10, 50)]
    assert len(set(counts)) == 1


def test_tube_follows_a_bend():
    shape = (60, 60, 60)
    path = [[10, 30, 30], [30, 30, 30], [30, 50, 30]]
    t = el.tube(shape, path, 2.0)
    assert t[12, 30, 30] and t[30, 48, 30]         # both limbs present
    assert not t[12, 48, 30]                       # but not the corner short-cut


def test_grid_rasterization_blocks():
    """grid=n produces uniformly filled n x n x n blocks."""
    shape = (40, 40, 40)
    m = el.rasterize("sphere", shape, grid=2, center=[20, 20, 20], radius=8.0)
    idx = np.argwhere(m)
    coarse, counts = np.unique(idx // 2, axis=0, return_counts=True)
    assert set(counts.tolist()) == {8}
    # and it approximates the same object
    fine = el.rasterize("sphere", shape, grid=1, center=[20, 20, 20], radius=8.0)
    assert el.dice(m, fine) > 0.9


def test_fit_spherical_band_roundtrip():
    shape = (70, 70, 70)
    truth = dict(center=[35.0, 35.0, 35.0], r_in=12.0, r_out=15.0,
                 axis=[0.0, 0.0, 1.0], theta_min_deg=70.0, theta_max_deg=110.0)
    ref = el.spherical_band(shape, **truth)
    fit = el.fit_spherical_band(np.argwhere(ref), axis_hint=[0, 0, 1])
    assert np.allclose(fit["center"], truth["center"], atol=0.6)
    assert fit["r_in"] == pytest.approx(truth["r_in"], abs=1.0)
    assert fit["r_out"] == pytest.approx(truth["r_out"], abs=1.0)
    regen = el.spherical_band(shape, center=fit["center"], r_in=fit["r_in"],
                              r_out=fit["r_out"], axis=fit["axis"],
                              theta_min_deg=fit["theta_min_deg"],
                              theta_max_deg=fit["theta_max_deg"])
    assert el.dice(ref, regen) > 0.9


def test_fit_tube_path_roundtrip_curved():
    shape = (60, 60, 80)
    path = [[30, 20, 10], [30, 20, 50], [30, 34, 68]]
    ref = el.tube(shape, path, 2.5)
    fit = el.fit_tube_path(np.argwhere(ref))
    assert fit["radius"] == pytest.approx(2.5, abs=0.6)
    # the recovered centre-line keeps the lead's full length through the bend
    assert fit["_length"] == pytest.approx(58.0, rel=0.2)
    best = max(el.dice(ref, el.tube(shape, fit["path"], r))
               for r in np.arange(2.0, 3.0, 0.1))
    assert best > 0.85


def test_refine_by_dice_improves():
    shape = (50, 50, 50)
    ref = el.spherical_band(shape, [25, 25, 25], 10.0, 12.0)
    start = dict(center=[25, 25, 25], r_in=9.0, r_out=13.0)
    d0 = el.dice(ref, el.spherical_band(shape, **start))
    best, d1 = el.refine_by_dice(ref, "spherical_band", start,
                                 {"r_in": [-0.5, 0, 0.5, 1.0],
                                  "r_out": [-1.0, -0.5, 0, 0.5]}, shape)
    assert d1 >= d0 and d1 > 0.95


def test_place_paints_and_registers():
    lib = MaterialLibrary([Material.from_sigma(1, 1.0, name="tissue")])
    m = VoxelModel.empty((40, 40, 60), dx=1e-4, materials=lib, background=1)
    specs = [
        el.ElectrodeSpec(name="stim", role="source", material=101,
                         shape="disc",
                         params=dict(center=[20, 20, 3], radius=6.0,
                                     thickness=2.0, normal=[0, 0, 1]),
                         waveform="w"),
        el.ElectrodeSpec(name="ret", role="ground", material=102, shape="tube",
                         params=dict(path=[[20, 20, 56], [20, 20, 30]],
                                     radius=2.0),
                         insulation={"material": 110, "shape": "tube_shell",
                                     "params": {"path": [[20, 20, 56],
                                                         [20, 20, 40]],
                                                "r_in": 2.0, "r_out": 4.0}}),
    ]
    masks = el.place(m, specs)
    assert (m.labels == 101).sum() == masks["stim"].sum() > 0
    assert (m.labels == 102).sum() == masks["ret"].sum() > 0
    assert (m.labels == 110).sum() > 0
    # insulation never overwrites metal
    assert not (masks["ret:insulation"] & masks["ret"]).any()
    assert len(m.sources) == 1 and m.sources[0].name == "w"
    assert len(m.ground_nodes) == 1
    assert m.labels[tuple(m.sources[0].node)] == 101
    assert 101 in m.materials and 110 in m.materials
    # metal is metal, insulation is insulating
    assert m.materials[101].rho[0] < 1e-5
    assert m.materials[110].rho[0] > 1e5


def test_place_respects_protected_materials():
    lib = MaterialLibrary([Material.from_sigma(1, 1.0),
                           Material.from_sigma(77, 0.1, name="retina")])
    m = VoxelModel.empty((30, 30, 30), dx=1e-4, materials=lib, background=1)
    m.labels[10:20, 10:20, 10:20] = 77
    spec = el.ElectrodeSpec(name="s", role="source", material=101,
                            shape="sphere",
                            params=dict(center=[15, 15, 15], radius=8.0))
    el.place(m, [spec], protect=[77])
    assert (m.labels == 77).sum() == 1000          # retina untouched
    assert (m.labels == 101).sum() > 0


def test_spec_json_roundtrip():
    sp = el.ElectrodeSpec(name="cl", role="source", material=101,
                          shape="spherical_band", grid=2,
                          params=dict(center=[1.0, 2.0, 3.0], r_in=4.0,
                                      r_out=5.0, axis=[0, 0, 1],
                                      theta_min_deg=80.0, theta_max_deg=100.0))
    d = sp.to_dict()
    import json
    sp2 = el.ElectrodeSpec.from_dict(json.loads(json.dumps(d)))
    shape = (30, 30, 30)
    assert sp2.grid == 2
    assert el.dice(sp.rasterize(shape), sp2.rasterize(shape)) == 1.0


def test_compare_masks_metrics():
    a = np.zeros((10, 10, 10), bool); a[:5] = True
    b = np.zeros((10, 10, 10), bool); b[:4] = True
    r = el.compare_masks(a, b, "x")
    assert r["dice"] == pytest.approx(2 * 400 / (500 + 400))
    assert r["missed_reference"] == 0 and r["extra_generated"] == 100
