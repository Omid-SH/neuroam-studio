"""3D visualization: surfaces, field colouring, scene assembly, HTML output."""

import numpy as np
import pytest

from neuroam import viz3d
from neuroam.materials import Material, MaterialLibrary
from neuroam.model import VoxelModel

plotly = pytest.importorskip("plotly")
skimage = pytest.importorskip("skimage")


@pytest.fixture
def toy():
    lib = MaterialLibrary([Material.from_sigma(1, 1.0, name="tissue"),
                           Material.from_sigma(2, 0.1, name="target"),
                           Material.isotropic_rho(101, 1e-7, name="stim"),
                           Material.isotropic_rho(102, 1e-7, name="gnd")])
    m = VoxelModel.empty((30, 30, 30), dx=1e-4, materials=lib, background=1,
                         name="toy")
    m.add_sphere(2, center=(15, 15, 15), radius=7)
    m.add_box(101, corner=(13, 13, 0), size=(4, 4, 2))
    m.add_box(102, corner=(13, 13, 28), size=(4, 4, 2))
    field = np.linalg.norm(
        np.stack(np.meshgrid(*[np.arange(30) - 15] * 3, indexing="ij")), axis=0)
    return m, field.astype(float) + 1.0


def test_isosurface_from_mask():
    mask = np.zeros((20, 20, 20), bool)
    mask[5:15, 5:15, 5:15] = True
    v, f = viz3d.isosurface_from_mask(mask)
    assert len(v) > 0 and len(f) > 0
    assert v[:, 0].min() >= 4 and v[:, 0].max() <= 16
    # empty mask returns empty arrays rather than raising
    v2, f2 = viz3d.isosurface_from_mask(np.zeros((10, 10, 10), bool))
    assert len(v2) == 0 and len(f2) == 0


def test_isosurface_decimation_and_spacing():
    mask = np.zeros((40, 40, 40), bool)
    mask[10:30, 10:30, 10:30] = True
    v1, f1 = viz3d.isosurface_from_mask(mask, step=1)
    v2, f2 = viz3d.isosurface_from_mask(mask, step=2)
    assert len(f2) < len(f1)
    v3, _ = viz3d.isosurface_from_mask(mask, spacing=2.0)
    assert v3[:, 0].max() == pytest.approx(2 * v1[:, 0].max())


def test_sample_at_vertices_linear_field():
    field = np.fromfunction(lambda i, j, k: 2.0 * i + 3.0, (10, 10, 10))
    verts = np.array([[2.5, 4.0, 4.0], [6.5, 2.0, 2.0]])
    vals = viz3d.sample_at_vertices(field, verts)
    assert np.allclose(vals, 2.0 * (verts[:, 0] - 0.5) + 3.0, atol=1e-9)


def test_scene_layers_and_html(tmp_path, toy):
    model, field = toy
    sc = viz3d.Scene(dx=model.dx, title="t", subtitle="s", theme="dark")
    sc.add_materials(model.labels, [1], "context", role="context", opacity=0.1,
                     step=2)
    sc.add_field_on_surface(model.labels, [2], field, "target |J|")
    sc.add_electrode(model.labels, [101], "stim", role="source")
    sc.add_electrode(model.labels, [102], "gnd", role="ground")
    sc.add_field_isosurface(field, [8.0, 12.0], "iso")
    sc.add_field_points(field, "hot", mask=model.labels == 2, percentile=90)
    sc.add_slice(field, "z", 15, "cut")
    assert len(sc.traces) >= 7
    # every layer is individually toggleable and named
    assert all(t.showlegend for t in sc.traces)
    assert all(t.name for t in sc.traces)

    p = sc.write_html(tmp_path / "v.html")
    txt = p.read_text()
    assert p.stat().st_size > 10_000
    assert "plotly" in txt.lower()
    for needle in ("context", "target |J|", "stim", "gnd"):
        assert needle in txt


def test_scene_colors_are_role_based(toy):
    model, _ = toy
    sc = viz3d.Scene(dx=model.dx, theme="dark")
    sc.add_electrode(model.labels, [101], "stim", role="source")
    sc.add_electrode(model.labels, [102], "gnd", role="ground")
    assert sc.traces[0].color == viz3d.PALETTE["dark"]["source"]
    assert sc.traces[1].color == viz3d.PALETTE["dark"]["ground"]
    light = viz3d.Scene(dx=model.dx, theme="light")
    light.add_electrode(model.labels, [101], "stim", role="source")
    assert light.traces[0].color == viz3d.PALETTE["light"]["source"]


def test_scene_crop_limits_geometry(toy):
    model, field = toy
    full = viz3d.Scene(dx=model.dx)
    full.add_materials(model.labels, [2], "t", step=1)
    cropped = viz3d.Scene(dx=model.dx, crop=((10, 20), (10, 20), (10, 20)))
    cropped.add_materials(model.labels, [2], "t", step=1)
    assert len(cropped.traces[0].x) < len(full.traces[0].x)
    # marching-cubes vertices sit half a voxel outside the mask, so allow that
    assert min(cropped.traces[0].x) >= (10 - 0.55) * model.dx * 1e3


def test_units_scaling(toy):
    model, _ = toy
    mm = viz3d.Scene(dx=1e-3, units="mm")
    vox = viz3d.Scene(dx=1e-3, units="voxel")
    mm.add_materials(model.labels, [2], "t")
    vox.add_materials(model.labels, [2], "t")
    assert max(mm.traces[0].x) == pytest.approx(max(vox.traces[0].x), rel=1e-9)


def test_view_helpers_write_files(tmp_path, toy):
    model, field = toy
    p1 = viz3d.view_model(model, tmp_path / "m.html",
                          focus={"target": [2]}, source_ids=(101,),
                          ground_ids=(102,), insulation_ids=())
    p2 = viz3d.view_current_density(model, field, tmp_path / "j.html",
                                    focus={"target": [2]}, source_ids=(101,),
                                    ground_ids=(102,), insulation_ids=(),
                                    iso_levels=[10.0], slices=[("z", 15)])
    assert p1.exists() and p2.exists()
    assert p1.stat().st_size > 10_000 and p2.stat().st_size > 10_000


def test_snapshot_png(tmp_path, toy):
    model, field = toy
    p = viz3d.snapshot_png(tmp_path / "s.png", model, field=field,
                           focus_ids=[2], electrode_ids=[101, 102])
    assert p.exists() and p.stat().st_size > 5_000
    p2 = viz3d.snapshot_png(tmp_path / "m.png", model, focus_ids=[2])
    assert p2.exists()
    p3 = viz3d.snapshot_png(tmp_path / "f.png", model, field=field,
                            focus_ids=[2], scale_ids=[2], mask_to_focus=True)
    assert p3.exists()


# --------------------------------------------------------------- neuron views
@pytest.fixture
def morph_vox():
    from neuroam.morphology import synthetic_branching_tree
    m = synthetic_branching_tree(n_generations=2, branching=2, seed=0)
    # local synthetic tree is tens of um across; drop it near the toy
    # model's centre (dx=1e-4 m = 100 um voxels)
    pts_vox = m.xyz_um / 100.0 + np.array([15.0, 15.0, 15.0])
    return pts_vox, m.edges()


def test_add_neuron_flat_color_trace(toy):
    model, field = toy
    pts, edges = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]]), np.array([[0, 1], [1, 2]])
    sc = viz3d.Scene(dx=model.dx, units="mm")
    sc.add_neuron(pts, edges, "cell", role="target")
    assert len(sc.traces) == 1
    tr = sc.traces[0]
    assert tr.mode == "lines"
    # NaN-separated: 3 points/edge * 2 edges = 6 entries
    assert len(tr.x) == 6
    assert np.isnan(tr.x[2]) and np.isnan(tr.x[5])


def test_add_neuron_colored_by_values_has_colorbar(toy):
    model, field = toy
    pts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
    edges = np.array([[0, 1], [1, 2]])
    values = np.array([-65.0, -40.0, 10.0])
    sc = viz3d.Scene(dx=model.dx, units="mm")
    sc.add_neuron(pts, edges, "cell", values=values, colorbar_title="Vm (mV)")
    line = sc.traces[0].line
    assert line.showscale is True
    assert np.nanmax(line.color) == pytest.approx(10.0)
    assert np.nanmin(line.color) == pytest.approx(-65.0)


def test_view_neuron_context_and_field(tmp_path, toy, morph_vox):
    model, field = toy
    pts, edges = morph_vox
    p1 = viz3d.view_neuron_context(model, pts, edges, tmp_path / "ctx.html",
                                   focus={"target": [2]}, source_ids=(101,),
                                   ground_ids=(102,))
    p2 = viz3d.view_neuron_field(model, field, pts, edges, tmp_path / "fld.html",
                                 pad_vox=10,
                                 values=np.linspace(-65, 20, len(pts)))
    assert p1.exists() and p1.stat().st_size > 10_000
    assert p2.exists() and p2.stat().st_size > 10_000


def test_animate_neuron_activity(tmp_path, morph_vox):
    pts, edges = morph_vox
    T = 12
    vals = (np.linspace(-65, 10, T)[:, None]
           + np.zeros(len(pts))[None, :])
    t_ms = np.linspace(0, 5, T)
    out = viz3d.animate_neuron_activity(pts, edges, t_ms, vals,
                                        tmp_path / "act.html", dx=1e-4,
                                        units="mm", max_frames=8)
    assert out.exists() and out.stat().st_size > 5_000
    html = out.read_text(encoding="utf-8")
    assert "Plotly.newPlot" in html
    assert "frames" in html.lower()


def test_view_neuron_placement_editor(tmp_path, toy, morph_vox):
    model, field = toy
    pts, edges = morph_vox
    out = viz3d.view_neuron_placement_editor(
        model, pts, edges, tmp_path / "editor.html", focus={"target": [2]},
        source_ids=(101,), ground_ids=(102,))
    assert out.exists() and out.stat().st_size > 10_000

    html = out.read_text(encoding="utf-8")
    # the live-transform machinery a viewer's browser needs is actually there
    for marker in ("Plotly.restyle", "applyTransform", "rotationMatrix",
                  'id="tx"', 'id="ry"', 'id="readout"', "NEURON_TRACE",
                  "MARKER_TRACE", "transform_coordinates"):
        assert marker in html, f"missing {marker!r}"

    # the embedded point/edge payload matches what was passed in, not a
    # placeholder -- this is what the live JS transform actually operates on
    import json
    import re
    m = re.search(r"const DATA = (\{.*?\});\n", html)
    assert m is not None
    data = json.loads(m.group(1))
    assert len(data["points"]) == len(pts)
    assert np.allclose(data["pivot"], np.asarray(pts).mean(axis=0), atol=1e-6)
    assert len(data["edges"]) == len(edges)
