"""NEURON coupling test — runs only if the neuron package is installed."""

import numpy as np
import pytest

from neuroam.neuron_link import (neuron_available, make_ball_and_stick,
                                 get_segment_coordinates, run_extracellular)

pytestmark = pytest.mark.skipif(not neuron_available(),
                                reason="NEURON not installed")


def test_extracellular_drive_and_spike():
    sections = make_ball_and_stick(axon_len_um=600.0, nseg=31)
    coords = get_segment_coordinates(sections, dx_um=50.0)
    n_seg = len(coords)
    assert n_seg == 1 + 31

    dt = 0.025
    T = int(20.0 / dt)
    v = np.zeros((T, n_seg))

    # strong focal cathodic pulse near the axon midpoint, 1-2 ms
    dist = np.linalg.norm(coords - coords[n_seg // 2], axis=1)
    profile = -80.0 / (1.0 + dist ** 2)          # mV
    i0, i1 = int(1.0 / dt), int(2.0 / dt)
    v[i0:i1, :] = profile

    run = run_extracellular(sections, v, dt_ms=dt, tstop_ms=20.0,
                            record_segments=[0, n_seg - 1],
                            spike_section_index=n_seg - 1)
    assert run.vm_mV.shape[1] == 2
    # resting before stimulus
    assert abs(run.vm_mV[5, 0] + 65.0) < 5.0
    # the cathodic shock should elicit at least one propagated spike
    assert len(run.spikes_ms) >= 1, "no spike detected at distal axon"


def test_no_drive_stays_at_rest():
    sections = make_ball_and_stick(axon_len_um=300.0, nseg=11)
    coords = get_segment_coordinates(sections, dx_um=50.0)
    dt = 0.025
    T = int(5.0 / dt)
    v = np.zeros((T, len(coords)))
    run = run_extracellular(sections, v, dt_ms=dt, tstop_ms=5.0,
                            record_segments=[0])
    assert np.all(np.abs(run.vm_mV + 65.0) < 6.0)
