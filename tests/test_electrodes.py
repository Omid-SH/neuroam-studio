"""Electrode design, registration and terminal-model tests."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from neuroam.assembly import assemble_uniform
from neuroam.electrodes import (ElectrodeSpec, Montage, Overlay, Terminal,
                                central_node, compare_masks, dice,
                                electrode_nodes, fit_spherical_band,
                                fit_tube_path, refine_by_dice, register)
from neuroam.frames import Frame, sphere_fit, surface_band, snap_to_surface
from neuroam.geometry import (Annulus, Box, Cylinder, Grid, Needle, Sphere,
                              SphericalBand, SphericalCap, Torus, Polyline,
                              rasterize, rotation_axis_angle)
from neuroam.materials import Material, MaterialLibrary, METAL_RHO
from neuroam.model import VoxelModel
from neuroam.solver import electrode_impedance, solve, unit_current_vector

SAMPLE = (Path(__file__).resolve().parents[1]
          / "samples/ratcc_eye_83um/RatCC_eye_180_160_220_83um_with_retina.in")


def _tissue(n=24, dx=1e-4, rho=1.0, name="box"):
    lib = MaterialLibrary()
    lib.add(Material.isotropic_rho(1, rho, name="tissue"))
    return VoxelModel.empty((n, n, n), dx, lib, background=1, name=name)


def _mask_of(region, grid, **kw):
    idx, _, _ = rasterize(region, grid, **kw)
    m = np.zeros(grid.shape, dtype=bool)
    if len(idx):
        m[tuple(idx.T)] = True
    return m


# ------------------------------------------------------------------ geometry
@pytest.mark.parametrize("shape,exact", [
    (Sphere(2.0), 4 / 3 * math.pi * 8),
    (Box([1.0, 2.0, 3.0]), 6.0),
    (Cylinder(1.0, 2.0), math.pi * 2.0),
    (Torus(2.0, 0.4), 2 * math.pi ** 2 * 2.0 * 0.16),
    (Annulus(1.0, 1.5, 0.4), math.pi * (1.5 ** 2 - 1.0 ** 2) * 0.4),
])
def test_rasterized_volume_matches_analytic(shape, exact):
    grid = Grid((80, 80, 80), 0.1)
    idx, _, _ = rasterize(shape.translate([4, 4, 4]), grid, supersample=4)
    vol = len(idx) * grid.dx_mm ** 3
    assert vol == pytest.approx(exact, rel=0.05)


def test_volume_is_rotation_invariant():
    """A rigid motion must not change how much of a shape exists."""
    grid = Grid((80, 80, 80), 0.1)
    base = Cylinder(1.0, 2.0).translate([4, 4, 4])
    n0 = len(rasterize(base, grid, supersample=4)[0])
    for axis, ang in [((0, 0, 1), 30), ((1, 1, 0), 45), ((1, 2, 3), 100)]:
        n = len(rasterize(base.rotate(axis, ang, about=[4, 4, 4]), grid,
                          supersample=4)[0])
        assert abs(n - n0) / n0 < 0.03


def test_bbox_restriction_matches_bruteforce():
    grid = Grid((30, 30, 30), 0.2)
    s = Sphere(1.0).translate([3, 3, 3])
    idx, _, _ = rasterize(s, grid, supersample=1)
    I, J, K = np.meshgrid(*[np.arange(30)] * 3, indexing="ij")
    C = grid.centers_mm(np.stack([I, J, K], -1))
    brute = np.argwhere(s.contains(C))
    assert set(map(tuple, idx.tolist())) == set(map(tuple, brute.tolist()))


def test_lattice_rasterization_fills_uniform_blocks():
    """lattice=n authors the region on an n x n x n-coarser physical lattice.

    Lab electrodes are frequently hand-drawn this way (e.g. 166 um pitch
    inside an 83 um model); matching it is what lets a generated electrode
    reproduce a hand-built one voxel-for-voxel rather than approximately.
    """
    grid = Grid((40, 40, 40), 0.1)
    s = Sphere(1.6).translate(grid.centers_mm([20, 20, 20]))
    idx, _, _ = rasterize(s, grid, supersample=3, lattice=2)
    coarse, counts = np.unique(idx // 2, axis=0, return_counts=True)
    assert set(counts.tolist()) == {8}          # every 2x2x2 block is whole
    fine, _, _ = rasterize(s, grid, supersample=3, lattice=1)
    A = np.zeros(grid.shape, bool); A[tuple(idx.T)] = True
    B = np.zeros(grid.shape, bool); B[tuple(fine.T)] = True
    assert (A & B).sum() / (A | B).sum() > 0.9   # still approximates the same object


def test_sdf_is_a_true_distance_near_surface():
    s = Sphere(2.0).rotate((1, 1, 0), 33).translate([1, -2, 3])
    P = np.random.default_rng(0).normal(size=(500, 3)) * 3 + np.array([1, -2, 3])
    d = s.sdf(P)
    assert np.allclose(np.abs(d), np.abs(np.linalg.norm(P - np.array([1, -2, 3]),
                                                        axis=1) - 2.0), atol=1e-9)


def test_polyline_and_needle_have_sane_extent():
    grid = Grid((60, 60, 60), 0.1)
    wire = Polyline([[1, 1, 1], [4, 1, 1], [4, 4, 1]], 0.2)
    idx, _, _ = rasterize(wire, grid, supersample=3)
    assert len(idx) > 0
    vol = len(idx) * grid.dx_mm ** 3
    assert vol == pytest.approx(math.pi * 0.04 * 6 + 4 / 3 * math.pi * 0.008,
                                rel=0.15)
    tip = np.array([3.0, 3.0, 1.0])
    nd = Needle(0.1, 2.0, 0.3).place(tip, (0, 0, 1))
    idx, _, _ = rasterize(nd, grid, supersample=3)
    P = grid.centers_mm(idx)
    assert P[:, 2].min() >= tip[2] - grid.dx_mm
    assert P[:, 2].max() <= tip[2] + 2.0 + grid.dx_mm


# -------------------------------------------------------------------- frames
def test_frame_recovers_a_synthetic_globe():
    n, dx = 60, 1e-4
    model = _tissue(n, dx)
    grid = Grid.from_model(model)
    centre = np.array([30.0, 28.0, 31.0])
    I, J, K = np.meshgrid(*[np.arange(n)] * 3, indexing="ij")
    r = np.sqrt((I - centre[0]) ** 2 + (J - centre[1]) ** 2 + (K - centre[2]) ** 2)
    shell = (r >= 18) & (r <= 20)
    # posterior cap only, like a retina
    model.labels[shell & (K < centre[2])] = 77
    f = Frame.eye(model, (77,))
    assert np.allclose(f.origin_vox, centre, atol=0.6)
    assert f.radius_mm == pytest.approx(19 * grid.dx_mm, rel=0.05)
    assert f.e3[2] > 0.9        # anterior points away from the retina cap


def test_frame_place_is_consistent_with_spherical_coords():
    grid = Grid((40, 40, 40), 0.1)
    f = Frame.from_axes([2, 2, 2], (0, 0, 1), grid)
    p = f.point(1.0, 90.0, 0.0)
    assert np.allclose(p, [3, 2, 2])
    r, th, ph = f.spherical_of(np.array([[2, 2, 3.0]]))
    assert r[0] == pytest.approx(1.0) and th[0] == pytest.approx(0.0, abs=1e-6)


def test_snap_to_surface_puts_the_contact_where_asked():
    n, dx = 40, 1e-4
    model = _tissue(n, dx)
    model.labels[:, :, :20] = 0                       # tissue only for z >= 20
    model.labels[:, :, 20:] = 1
    grid = Grid.from_model(model)
    disc = Cylinder(0.3, 0.05).place(grid.centers_mm([20, 20, 10]), (0, 0, 1))
    snapped = snap_to_surface(disc, model, [1], (0, 0, 1), standoff_mm=0.0)
    idx, _, _ = rasterize(snapped, grid, supersample=3)
    top = grid.centers_mm(idx)[:, 2].max()
    surface = grid.centers_mm([0, 0, 20])[2] - 0.5 * grid.dx_mm
    assert top <= surface + grid.dx_mm


# ------------------------------------------------------- registration and QC
def test_central_node_is_always_inside_the_electrode():
    """The legacy 'middle element of argwhere' rule is not."""
    grid = Grid((60, 60, 60), 0.1)
    ring = Torus(2.0, 0.15).translate([3, 3, 3])
    idx, _, _ = rasterize(ring, grid, supersample=3)
    nodes = electrode_nodes(idx, (60, 60, 60))
    c = central_node(idx, (60, 60, 60))
    assert (nodes == np.array(c)).all(1).any()
    legacy = idx[len(idx) // 2]           # what IN_editor_2.py picks
    # a node at the centroid of a ring is in the hole, not in the metal
    centre_vox = np.round(idx.mean(0)).astype(int)
    assert not (idx == centre_vox).all(1).any()
    assert (idx == legacy).all(1).any()   # legacy is a voxel, but arbitrary


def test_register_paints_and_connects():
    model = _tissue(24)
    grid = Grid.from_model(model)
    src = Cylinder(0.3, 0.1).place(grid.centers_mm([12, 12, 2]), (0, 0, 1))
    gnd = Cylinder(0.3, 0.1).place(grid.centers_mm([12, 12, 21]), (0, 0, 1))
    specs = [ElectrodeSpec("s", region=src, role="source", label=101,
                           waveform="I"),
             ElectrodeSpec("g", region=gnd, role="ground", label=102)]
    overlay, qc = register(model, specs)
    assert qc[0].n_voxels > 0 and qc[1].n_voxels > 0
    assert (model.labels == 101).sum() == qc[0].n_voxels
    assert qc[0].terminal_inside and not qc[0].warnings
    assert len(model.terminals) == 2
    assert model.terminals[0].kind == "supernode"
    assert qc[0].surface_area_mm2 > 0
    assert 101 in model.materials and model.materials[101].rho[0] == METAL_RHO


def test_qc_flags_a_terminal_node_outside_the_electrode():
    """The failure mode behind the legacy sample.in ground node."""
    model = _tissue(24)
    grid = Grid.from_model(model)
    disc = Cylinder(0.3, 0.1).place(grid.centers_mm([12, 12, 2]), (0, 0, 1))
    spec = ElectrodeSpec("s", region=disc, role="source", label=101,
                         terminal="node", node=(0, 0, 0), waveform="I")
    _, qc = register(model, [spec])
    assert qc[0].terminal_inside is False
    assert any("NOT a node" in w for w in qc[0].warnings)


def test_qc_flags_shorted_electrodes_and_overwrite_policy():
    model = _tissue(24)
    grid = Grid.from_model(model)
    a = Cylinder(0.3, 0.1).place(grid.centers_mm([12, 12, 5]), (0, 0, 1))
    _, qc = register(model, [
        ElectrodeSpec("a", region=a, role="source", label=101, waveform="I"),
        ElectrodeSpec("b", region=a, role="ground", label=102)])
    assert qc[1].overlaps.get("a", 0) > 0
    assert any("shorted" in w for w in qc[1].warnings)

    model2 = _tissue(24)
    model2.labels[:] = 1                       # no background anywhere
    _, qc2 = register(model2, [ElectrodeSpec("a", region=a, role="source",
                                             label=101, overwrite="background",
                                             waveform="I")])
    assert qc2[0].n_voxels == 0 and qc2[0].n_blocked > 0


def test_overlay_roundtrip_equals_a_baked_model(tmp_path):
    model = _tissue(24)
    grid = Grid.from_model(model)
    disc = Cylinder(0.3, 0.1).place(grid.centers_mm([12, 12, 2]), (0, 0, 1))
    baseline = model.labels.copy()
    overlay, _ = register(model, [ElectrodeSpec("s", region=disc, role="source",
                                                label=101, waveform="I")])
    baked = model.labels.copy()
    p = overlay.save(tmp_path / "o.npz")
    again = Overlay.load(p).apply(baseline, copy=True)
    assert np.array_equal(again, baked)
    assert len(overlay) == (baked == 101).sum()


def test_legacy_export_writes_a_terminal_node_inside_the_electrode(tmp_path):
    model = _tissue(24)
    grid = Grid.from_model(model)
    ring = Torus(0.6, 0.12).place(grid.centers_mm([12, 12, 3]), (0, 0, 1))
    gnd = Cylinder(0.4, 0.1).place(grid.centers_mm([12, 12, 21]), (0, 0, 1))
    mont = Montage("m", [
        ElectrodeSpec("CL", region=ring, role="source", label=101, waveform="Cur1"),
        ElectrodeSpec("G", region=gnd, role="ground", label=102)]).build(model)
    in_p, model_p = mont.write_legacy(model, tmp_path)
    back = VoxelModel.from_legacy(in_p, model_p)
    assert np.array_equal(back.labels, model.labels)
    node = np.array(back.sources[0].node)
    idx = np.argwhere(model.labels == 101)
    assert (electrode_nodes(idx, model.world) == node).all(1).any()


# ------------------------------------------------------------ terminal models
def test_supernode_slab_is_exactly_rho_L_over_A():
    n, dx, rho = 16, 1e-4, 2.0
    model = _tissue(n, dx, rho)
    top = np.array([[i, j, n] for i in range(n + 1) for j in range(n + 1)])
    bot = np.array([[i, j, 0] for i in range(n + 1) for j in range(n + 1)])
    model.terminals = [Terminal("s", "source", "supernode", top, waveform="I"),
                       Terminal("g", "ground", "supernode", bot)]
    S = assemble_uniform(model)
    assert S.merge_inv is not None
    assert S.n < (n + 1) ** 3            # nodes were merged away
    v = solve(S, unit_current_vector(S, "I"), method="cg", rtol=1e-12).v
    R = electrode_impedance(S, v, "I")
    assert R == pytest.approx(rho * (n * dx) / (n * dx) ** 2, rel=1e-9)


def test_supernode_disc_approaches_spreading_resistance():
    """R -> rho/(4a) for a disc on a half-space as the domain grows."""
    ratios = []
    for N in (32, 48):
        dx, rho, a_vox = 1e-4, 1.0, 4
        model = _tissue(N, dx, rho)
        gx, gy = np.meshgrid(np.arange(N + 1), np.arange(N + 1), indexing="ij")
        disc = np.argwhere((gx - N / 2) ** 2 + (gy - N / 2) ** 2 <= a_vox ** 2)
        top = np.c_[disc, np.zeros(len(disc), int)]
        gnd = [[i, j, N] for i in range(N + 1) for j in range(N + 1)]
        for i in (0, N):
            gnd += [[i, j, k] for j in range(N + 1) for k in range(N + 1)]
        for j in (0, N):
            gnd += [[i, j, k] for i in range(N + 1) for k in range(N + 1)]
        model.terminals = [
            Terminal("s", "source", "supernode", top, waveform="I"),
            Terminal("g", "ground", "supernode", np.unique(np.array(gnd), axis=0))]
        S = assemble_uniform(model)
        v = solve(S, unit_current_vector(S, "I"), method="cg", rtol=1e-10).v
        ratios.append(electrode_impedance(S, v, "I") / (rho / (4 * a_vox * dx)))
    assert 0.7 < ratios[0] < 1.2
    assert ratios[1] > ratios[0]        # grounded walls recede -> R rises to rho/4a


def test_supernode_agrees_with_painted_metal_far_from_the_electrode():
    """An ideal equipotential terminal and a 1e-7 ohm*m metal body agree."""
    n, dx, rho = 28, 1e-4, 1.0
    grid = Grid((n, n, n), dx * 1e3)
    disc = Cylinder(0.4, 0.12).place(grid.centers_mm([14, 14, 2]), (0, 0, 1))
    ret = Cylinder(0.4, 0.12).place(grid.centers_mm([14, 14, 25]), (0, 0, 1))

    fields = []
    for kind in ("supernode", "node"):
        m = _tissue(n, dx, rho)
        register(m, [ElectrodeSpec("s", region=disc, role="source", label=101,
                                   terminal=kind, waveform="I"),
                     ElectrodeSpec("g", region=ret, role="ground", label=102,
                                   terminal=kind)])
        S = assemble_uniform(m)
        v = solve(S, unit_current_vector(S, "I"), method="cg", rtol=1e-11).v
        fields.append(S.node_grid(v))
    mid = slice(6, 22)
    a, b = fields[0][mid, mid, 10:20], fields[1][mid, mid, 10:20]
    assert np.abs(a - b).max() / np.abs(b).max() < 0.02


def test_distributed_terminal_splits_the_current():
    n, dx = 20, 1e-4
    m = _tissue(n, dx)
    grid = Grid.from_model(m)
    disc = Cylinder(0.3, 0.1).place(grid.centers_mm([10, 10, 2]), (0, 0, 1))
    ret = Cylinder(0.3, 0.1).place(grid.centers_mm([10, 10, 17]), (0, 0, 1))
    register(m, [ElectrodeSpec("s", region=disc, role="source", label=101,
                               terminal="distributed", waveform="I"),
                 ElectrodeSpec("g", region=ret, role="ground", label=102,
                               terminal="supernode")])
    S = assemble_uniform(m)
    I = unit_current_vector(S, "I")
    assert I.sum() == pytest.approx(1.0)
    assert (I > 0).sum() > 1              # spread over the electrode's nodes


# ------------------------------------------------------------- real anatomy
@pytest.mark.skipif(not SAMPLE.exists(), reason="sample model not present")
def test_parametric_ring_reproduces_the_shipped_contact_lens():
    from scipy import ndimage
    model = VoxelModel.from_legacy(SAMPLE)
    grid = Grid.from_model(model)
    frame = Frame.eye(model, (77,))
    assert frame.radius_mm == pytest.approx(3.155, abs=0.05)
    assert frame.fit_rms_mm < 0.1

    truth = np.argwhere(model.labels == 101)
    idx, _, _ = rasterize(frame.ring(3.360, 53.3, 0.12), grid, supersample=3)
    lo = np.minimum(idx.min(0), truth.min(0)) - 2
    hi = np.maximum(idx.max(0), truth.max(0)) + 3
    A = np.zeros(hi - lo, bool); A[tuple((truth - lo).T)] = True
    B = np.zeros(hi - lo, bool); B[tuple((idx - lo).T)] = True
    dA = ndimage.distance_transform_edt(~A)
    dB = ndimage.distance_transform_edt(~B)
    # the ring is only ~3 voxels thick, so IoU is dominated by its surface;
    # the meaningful statement is that the two rings never separate by more
    # than one and a half voxels anywhere.
    assert max(dB[A].max(), dA[B].max()) <= 1.5
    assert (A & B).sum() / (A | B).sum() > 0.45


@pytest.mark.skipif(not SAMPLE.exists(), reason="sample model not present")
def test_shipped_electrodes_are_single_conductors_and_terminals_are_inside():
    model = VoxelModel.from_legacy(SAMPLE)
    node = model.sources[0].node
    gnode = model.ground_nodes[0]
    _, qc = register(model, [
        ElectrodeSpec("CL", from_label=101, role="source", label=101,
                      terminal="node", node=node, waveform="Cur1"),
        ElectrodeSpec("J", from_label=102, role="ground", label=102,
                      terminal="node", node=gnode)])
    assert qc[0].n_components == 1 and qc[1].n_components == 1
    assert qc[0].terminal_inside and qc[1].terminal_inside


# --------------------------------------------------------------------------
# Clinical analogues: TES-GPS (ocular thread) and VIRON (periorbital skin).
# See docs/CLINICAL_ELECTRODES.md and configs/ratcc_{tes_gps,viron_*}.json.
# --------------------------------------------------------------------------

def test_frame_arc_is_a_partial_ring_with_the_right_contact_length():
    """`Frame.arc` is the open-arc counterpart of `Frame.ring`."""
    model = _tissue(n=40, dx=1e-4)
    frame = Frame.from_axes(np.array([2.0, 2.0, 2.0]), np.array([0.0, 0.0, 1.0]),
                            Grid.from_model(model))
    r, theta, span, tube = 1.5, 60.0, 120.0, 0.06

    assert frame.arc_length_mm(r, theta, span) == pytest.approx(
        r * math.sin(math.radians(theta)) * math.radians(span))

    arc = frame.arc(r, theta, tube, phi0_deg=0.0, span_deg=span)
    pts = np.asarray(arc.points_mm)
    # every sample sits on the sphere of radius r about the frame origin ...
    assert np.allclose(np.linalg.norm(pts - frame.origin_mm, axis=1), r, atol=1e-9)
    # ... at the requested polar angle ...
    _, th, ph = frame.spherical_of(pts)
    assert np.allclose(th, theta, atol=1e-6)
    # ... and spans exactly `span` degrees of azimuth, centred on phi0.
    assert ph.min() == pytest.approx(-span / 2, abs=1e-6)
    assert ph.max() == pytest.approx(+span / 2, abs=1e-6)

    # a full turn closes the loop and matches the ring of the same address
    closed = np.asarray(frame.arc(r, theta, tube, span_deg=360.0).points_mm)
    assert np.allclose(closed[0], closed[-1])


@pytest.mark.skipif(not SAMPLE.exists(), reason="sample model not present")
def test_measured_limbus_and_okuel_thread_lands_on_the_ocular_surface():
    """The TES-GPS thread must sit on cornea/sclera, not in the retina."""
    from neuroam.electrodes import region_from_spec
    model = VoxelModel.from_legacy(SAMPLE)
    model.labels[model.labels == 101] = 66          # strip the lab's CL ring
    grid = Grid.from_model(model)
    frame = Frame.eye(model, (77,))

    # the limbus is measured from the anatomy, not assumed
    shell = {}
    for lab in (66, 33):
        P = grid.centers_mm(np.argwhere(model.labels == lab))
        r, th, _ = frame.spherical_of(P)
        shell[lab] = th[(r > 3.15) & (r < 3.50)]
    edges = np.arange(40.0, 115.0, 5.0)
    nc, _ = np.histogram(shell[66], edges)
    ns, _ = np.histogram(shell[33], edges)
    limbus = float(edges[np.argmax(ns > nc)])
    assert 65.0 <= limbus <= 85.0                   # rat cornea is large

    spec = {"type": "arc", "frame": "eye", "r_mm": 3.46, "theta_deg": 74.0,
            "tube_radius_mm": 0.10, "phi0_deg": 225.0, "span_deg": 100.0}
    idx, _, _ = rasterize(region_from_spec(spec, model, {"eye": frame}), grid,
                          supersample=3)
    displaced = model.labels[tuple(idx.T)]
    # the thread lies on the eye wall; it must never reach the retina (77),
    # the lens (55) or the humour (44) it is supposed to sit outside of
    assert not np.any(displaced == 77)
    assert (displaced == 66).sum() + (displaced == 33).sum() > 0

    _, qc = register(model, [ElectrodeSpec("OkuEl", region=region_from_spec(
        spec, model, {"eye": frame}), role="source", label=101,
        terminal="supernode", waveform="TES", min_voxels=120)])
    assert qc[0].n_components == 1 and qc[0].terminal_inside
    assert qc[0].contact_by_label.get(66, 0) + qc[0].contact_by_label.get(33, 0) > 0


@pytest.mark.skipif(not SAMPLE.exists(), reason="sample model not present")
def test_on_surface_pad_lands_on_skin_and_missing_sites_are_reportable():
    """VIRON periorbital cups: `at.on_surface` addresses the skin, and a site
    the model does not contain fails loudly (or is skipped on request)."""
    from neuroam.electrodes import region_from_spec, specs_from_config
    model = VoxelModel.from_legacy(SAMPLE)
    frame = Frame.eye(model, (77,))

    def pad(phi):
        return {"type": "cylinder", "radius_mm": 1.3, "height_mm": 1.0,
                "frame": "eye",
                "at": {"theta_deg": 85.0, "phi_deg": phi,
                       "on_surface": {"labels": [13], "standoff_mm": 0.0,
                                      "max_mm": 14.0}},
                "conform": {"labels": [13], "offset_mm": -0.10,
                            "thickness_mm": 0.30}}

    _, qc = register(model, [ElectrodeSpec(
        "E_sup", region=region_from_spec(pad(45.0), model, {"eye": frame}),
        role="source", label=121, terminal="supernode",
        overwrite=[0, 13], min_voxels=150)])
    assert qc[0].n_components == 1 and qc[0].terminal_inside
    assert qc[0].contact_by_label.get(13, 0) > 0        # it touches skin

    # phi = 285 deg leaves this crop: without skip_if_missing that is an error
    entry = {"name": "E5", "role": "source", "label": 125,
             "geometry": pad(285.0)}
    with pytest.raises(ValueError, match="cropped|no .13."):
        specs_from_config([entry], model, {"eye": frame})
    with pytest.warns(RuntimeWarning):
        assert specs_from_config([dict(entry, skip_if_missing=True)], model,
                                 {"eye": frame}) == []


# --------------------------------------------------------------------------
# Fitting existing electrodes back into reusable specs, and comparing masks.
# This is what turns a hand-painted montage into a parametric, resolution-
# independent ElectrodeSpec -- see fit_spherical_band/fit_tube_path/
# refine_by_dice and docs/AM_VERIFICATION.md.
# --------------------------------------------------------------------------
def test_dice_and_compare_masks_metrics():
    a = np.zeros((10, 10, 10), bool); a[:5] = True
    b = np.zeros((10, 10, 10), bool); b[:4] = True
    assert dice(a, b) == pytest.approx(2 * 400 / (500 + 400))
    r = compare_masks(a, b, "x")
    assert r["dice"] == pytest.approx(dice(a, b))
    assert r["missed_reference"] == 0 and r["extra_generated"] == 100
    assert dice(np.zeros((3, 3), bool), np.zeros((3, 3), bool)) == 1.0


def test_fit_spherical_band_recovers_a_placed_ring():
    grid = Grid((70, 70, 70), 0.1)
    dx = grid.dx_mm
    center = grid.centers_mm([35, 35, 35])
    truth = SphericalBand(13.5 * dx, 3.0 * dx, 110.0, 70.0).place(center, (0, 0, 1))
    idx, _, _ = rasterize(truth, grid, supersample=3)
    mask = _mask_of(truth, grid, supersample=3)

    region, info = fit_spherical_band(idx, grid, axis_hint=[0, 0, 1])
    assert np.allclose(info["center_mm"], center, atol=0.6 * dx)
    assert info["r_in_mm"] == pytest.approx(12.0 * dx, abs=1.0 * dx)
    assert info["r_out_mm"] == pytest.approx(15.0 * dx, abs=1.0 * dx)

    regen = _mask_of(region, grid, supersample=3)
    assert dice(mask, regen) > 0.9


def test_fit_tube_path_recovers_a_bent_lead():
    grid = Grid((60, 60, 80), 0.1)
    dx = grid.dx_mm
    path_mm = grid.centers_mm([[30, 20, 10], [30, 20, 50], [30, 34, 68]])
    truth = Polyline(path_mm, 2.5 * dx)
    idx, _, _ = rasterize(truth, grid, supersample=3)
    mask = _mask_of(truth, grid, supersample=3)

    region, info = fit_tube_path(idx, grid)
    assert info["radius_mm"] == pytest.approx(2.5 * dx, abs=0.6 * dx)
    # the recovered centre-line keeps the lead's full length through the bend
    assert info["length_mm"] == pytest.approx(58.0 * dx, rel=0.2)

    best = max(dice(mask, _mask_of(Polyline(info["path_mm"], r), grid, supersample=3))
              for r in np.arange(2.0, 3.0, 0.1) * dx)
    assert best > 0.85


def test_refine_by_dice_improves_a_starting_guess():
    grid = Grid((50, 50, 50), 0.1)
    dx = grid.dx_mm
    center = grid.centers_mm([25, 25, 25])

    def build(p):
        return SphericalBand(0.5 * (p["r_in"] + p["r_out"]),
                             p["r_out"] - p["r_in"], 180.0, 0.0
                             ).place(center, (0, 0, 1))

    reference = _mask_of(build({"r_in": 10.0 * dx, "r_out": 12.0 * dx}), grid,
                         supersample=3)
    start = {"r_in": 9.0 * dx, "r_out": 13.0 * dx}
    d0 = dice(reference, _mask_of(build(start), grid, supersample=3))
    best, d1 = refine_by_dice(
        reference, build, start,
        {"r_in": (np.array([-0.5, 0, 0.5, 1.0]) * dx).tolist(),
         "r_out": (np.array([-1.0, -0.5, 0, 0.5]) * dx).tolist()},
        grid, supersample=3)
    assert d1 >= d0 and d1 > 0.95


def test_lattice_electrode_spec_reproduces_an_authoring_grid_shape():
    """A montage authored on a coarser lattice (grid=n) is only reproducible
    by generating on that same lattice -- exactly the situation MERGE_NOTES
    describes for the lab's 166 um-on-83 um contact-lens electrodes."""
    grid = Grid((40, 40, 40), 0.1)
    center = grid.centers_mm([19.3, 20.7, 18.6])   # off-lattice on purpose
    hand_drawn = SphericalCap(1.6, 0.4, 40.0).place(center, (0.2, 0.1, 1))
    truth = _mask_of(hand_drawn, grid, supersample=3, lattice=2)

    model = _tissue(40, dx=1e-4)
    spec_fine = ElectrodeSpec("s", region=hand_drawn, role="source", label=101,
                              waveform="I", lattice=1)
    spec_coarse = ElectrodeSpec("s", region=hand_drawn, role="source", label=101,
                                waveform="I", lattice=2)
    _, qc_fine = register(model, [spec_fine])
    model2 = _tissue(40, dx=1e-4)
    _, qc_coarse = register(model2, [spec_coarse])

    mask_fine = model.labels == 101
    mask_coarse = model2.labels == 101
    assert dice(mask_coarse, truth) == 1.0
    assert dice(mask_fine, truth) < 1.0             # the fine raster differs
