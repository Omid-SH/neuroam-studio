"""NeuronRunResult save/load round-trip -- pure numpy/json I/O, no NEURON
needed to read a saved run back."""

import numpy as np

from neuroam.neuron_link import NeuronRunResult, save_neuron_run, load_neuron_run


def _sample_result(with_swc: bool = True) -> NeuronRunResult:
    T, n_seg = 5, 4
    rng = np.random.default_rng(0)
    r = NeuronRunResult(
        t_ms=np.linspace(0, 1, T),
        vm_mV=rng.normal(-65, 5, size=(T, n_seg)),
        spikes_ms=np.array([0.3, 0.7]),
        seg_vox=rng.normal(size=(n_seg, 3)),
        edges=np.array([[0, 1], [1, 2], [2, 3]]),
        unit_v=rng.normal(size=n_seg),
        meta={"model": "toy", "amp_A": 2e-4, "dx": 1e-5},
    )
    if with_swc:
        r.swc_vox = rng.normal(size=(6, 3))
        r.swc_edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]])
        r.swc_to_seg = np.array([0, 0, 1, 2, 3, 3])
        r.field_crop = rng.random((8, 8, 8))
        r.meta["field_crop_origin_vox"] = [10, 10, 10]
    return r


def test_round_trip_with_swc(tmp_path):
    r = _sample_result(with_swc=True)
    p = save_neuron_run(tmp_path / "run.npz", r)
    back = load_neuron_run(p)

    assert np.allclose(back.t_ms, r.t_ms)
    assert np.allclose(back.vm_mV, r.vm_mV)
    assert np.allclose(back.spikes_ms, r.spikes_ms)
    assert np.allclose(back.seg_vox, r.seg_vox)
    assert np.array_equal(back.edges, r.edges)
    assert np.allclose(back.unit_v, r.unit_v)
    assert np.allclose(back.swc_vox, r.swc_vox)
    assert np.array_equal(back.swc_edges, r.swc_edges)
    assert np.array_equal(back.swc_to_seg, r.swc_to_seg)
    assert np.allclose(back.field_crop, r.field_crop)
    assert back.meta == r.meta


def test_round_trip_without_swc(tmp_path):
    r = _sample_result(with_swc=False)
    p = save_neuron_run(tmp_path / "run.npz", r)
    back = load_neuron_run(p)

    assert back.swc_vox is None
    assert back.swc_edges is None
    assert back.swc_to_seg is None
    assert back.field_crop is None
    assert np.allclose(back.vm_mV, r.vm_mV)


def test_no_spikes_round_trips_as_empty_array(tmp_path):
    r = _sample_result(with_swc=False)
    r.spikes_ms = []
    p = save_neuron_run(tmp_path / "run.npz", r)
    back = load_neuron_run(p)
    assert len(back.spikes_ms) == 0
