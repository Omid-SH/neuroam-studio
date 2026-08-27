"""End-to-end pipeline runs from JSON configs."""

import json

import numpy as np
import pytest

from neuroam.pipeline import run, load_config, build_model


def sphere_config(tmp_path, with_neuron=False):
    coords = np.stack([np.full(20, 10.0), np.full(20, 10.0),
                       np.linspace(2, 18, 20)], axis=1)
    cpath = tmp_path / "coords.txt"
    np.savetxt(cpath, coords, fmt="%.4f")

    cfg = {
        "neuroam": "0.1",
        "name": "demo",
        "model": {
            "world": [20, 20, 20],
            "unit_voxel_m": 1e-4,
            "background": 1,
            "shapes": [
                {"type": "sphere", "material": 2, "center": [10, 10, 10],
                 "radius": 6},
            ],
        },
        "materials": {
            "inline": [
                {"id": 1, "sigma": 1.5, "name": "saline"},
                {"id": 2, "sigma": 0.1, "name": "tissue"},
            ]
        },
        "electrodes": [
            {"name": "stim", "role": "source", "material": 101,
             "shape": {"type": "box", "corner": [9, 9, 0], "size": [2, 2, 1]},
             "node": [10, 10, 0], "waveform": "pulse"},
            {"name": "ret", "role": "ground", "material": 100,
             "shape": {"type": "box", "corner": [9, 9, 19], "size": [2, 2, 1]},
             "node": [10, 10, 20]},
        ],
        "waveforms": {
            "pulse": {"type": "biphasic", "amp_A": 1e-3, "pulse_width_s": 1e-3,
                      "period_s": 5e-3, "n_pulses": 2, "dt_s": 1e-4,
                      "ratio": 1.0}
        },
        "solve": {"method": "direct", "cache": True},
        "outputs": {"dir": str(tmp_path / "out"),
                    "save": ["npz", "png", "vof", "vavg"],
                    "slices": [{"axis": "z", "index": 10}],
                    "rois": [{"name": "tissue", "labels": [2]}]},
    }
    if with_neuron:
        cfg["neuron"] = {"coordinates": str(cpath), "unit_scale": 1e3}
    p = tmp_path / "demo.json"
    p.write_text(json.dumps(cfg))
    return p


def test_pipeline_end_to_end(tmp_path):
    p = sphere_config(tmp_path, with_neuron=True)
    manifest = run(p, progress=lambda *a: None)

    out = tmp_path / "out"
    assert (out / "run_manifest.json").exists()
    assert (out / "demo_fields.npz").exists()
    assert (out / "demo_unit.vof").exists()
    assert (out / "demo_vext.v").exists()

    assert manifest["conservation"]["relative_residual"] < 1e-8
    rois = {r["name"]: r for r in manifest["roi_metrics_per_unit_A"]}
    assert rois["tissue"]["voxels"] > 0
    assert rois["tissue"]["E_mean_V_per_m"] > 0

    # coupling matrix: T x n_compartments
    T = 2 * int(5e-3 / 1e-4)
    assert manifest["coupling"]["v_matrix_shape"] == [T, 20]
    assert manifest["coupling"]["n_outside_domain"] == 0

    with np.load(out / "demo_fields.npz") as z:
        assert z["Emag"].shape == (20, 20, 20)
        assert np.isfinite(z["Emag"]).all()


def test_pipeline_cache_reuse(tmp_path):
    p = sphere_config(tmp_path)
    m1 = run(p, progress=lambda *a: None)
    assert "cached" not in next(iter(m1["solver"].values()))
    m2 = run(p, progress=lambda *a: None)
    assert next(iter(m2["solver"].values())).get("cached") is True


def test_validate_config(tmp_path):
    p = sphere_config(tmp_path)
    cfg = load_config(p)
    model = build_model(cfg)
    assert model.world == (20, 20, 20)
    assert len(model.sources) == 1
    assert len(model.ground_nodes) == 1
