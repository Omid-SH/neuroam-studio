"""Non-GUI tests for viewer model loading and electrode detection."""

import numpy as np
import pytest

from neuroam.materials import Material, MaterialLibrary
from neuroam.model import VoxelModel
from neuroam.viewer import build_plotter, electrode_labels, load_view_model


def test_electrode_labels_detect_source_and_ground():
    lib = MaterialLibrary([
        Material.from_sigma(1, 0.3, name="tissue"),
        Material.isotropic_rho(101, 1e-7, name="source metal"),
        Material.isotropic_rho(102, 1e-7, name="ground metal"),
    ])
    model = VoxelModel.empty((8, 8, 8), 1e-4, lib, background=1)
    model.add_electrode("box", "source", 101, waveform="pulse",
                        corner=(1, 1, 1), size=(2, 2, 2), node=(2, 2, 2))
    model.add_electrode("box", "ground", 102,
                        corner=(5, 5, 5), size=(2, 2, 2), node=(6, 6, 6))

    assert electrode_labels(model) == {101: "source: pulse", 102: "ground"}


def test_load_view_model_rejects_unknown_extension(tmp_path):
    path = tmp_path / "model.txt"
    path.write_text("not a model")
    with pytest.raises(ValueError, match="legacy .in file or a config .json"):
        load_view_model(path)


def test_load_view_model_from_config(tmp_path):
    config = tmp_path / "tiny.json"
    config.write_text("""{
      "neuroam": "0.1",
      "name": "viewer-test",
      "model": {"world": [3, 4, 5], "unit_voxel_m": 0.001,
                "background": 1},
      "materials": {"inline": [{"id": 1, "sigma": 1.0}]}
    }""")
    model = load_view_model(config)
    assert model.world == (3, 4, 5)


def test_viewer_module_does_not_import_pyvista_eagerly():
    # Solver-only installations must be able to import viewer helpers.
    assert np is not None


def test_build_plotter_off_screen_when_viewer_extra_is_available():
    pytest.importorskip("pyvista")
    lib = MaterialLibrary([
        Material.from_sigma(1, 0.3, name="tissue"),
        Material.isotropic_rho(101, 1e-7, name="source metal"),
    ])
    model = VoxelModel.empty((6, 6, 6), 1e-4, lib, background=1,
                             name="viewer-smoke")
    model.add_electrode("box", "source", 101, waveform="pulse",
                        corner=(2, 2, 0), size=(2, 2, 1), node=(3, 3, 0))
    plotter, grid = build_plotter(model, labels=[1], off_screen=True)
    assert grid.n_cells == model.labels.size
    assert "material" in grid.cell_data
    plotter.close()
