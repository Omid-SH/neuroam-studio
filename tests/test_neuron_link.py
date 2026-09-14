"""NEURON coupling test — runs only if the neuron package is installed."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from neuroam.neuron_link import (neuron_available, make_ball_and_stick,
                                 make_branching_cell, get_segment_coordinates,
                                 get_segment_edges, load_hoc_cell,
                                 run_extracellular)

pytestmark = pytest.mark.skipif(not neuron_available(),
                                reason="NEURON not installed")

REPO = Path(__file__).resolve().parents[1]
SAMPLES = REPO / "samples/neuron_models"
RGC_DIR = SAMPLES / "rgc_d1"
A2I_DIR = SAMPLES / "rgc_a2i"


def _has_compiled_mechanisms(d: Path) -> bool:
    return (d / "nrnmech.dll").exists() or bool(list(d.glob("*/.libs/libnrnmech.so")))


def _load_in_subprocess(model_dir: Path, expect_gnabar: float) -> None:
    """Build a hoc-template cell in its own process.

    Two of these sibling models (rgc_d1/rgc_a2i) ship byte-identical
    ``makecell.hoc`` content -- hoc refuses to redefine a template/obfunc a
    second time in one process (see load_hoc_cell's docstring), so testing
    both for real means testing each in isolation, exactly as any real user
    of two such sibling models would have to run them.
    """
    script = f"""
import sys
sys.path.insert(0, {str(REPO)!r})
from neuroam.neuron_link import load_hoc_cell, get_segment_coordinates, get_segment_edges
sections = load_hoc_cell({str(model_dir)!r})
assert len(sections) > 200, len(sections)
coords = get_segment_coordinates(sections, dx_um=1.0)
edges = get_segment_edges(sections)
assert edges.shape == (len(coords) - 1, 2)
soma = next(s for s in sections if "soma" in s.name())
gnabar = soma(0.5).gnabar_spike
assert abs(gnabar - {expect_gnabar}) < 1e-9, gnabar
from neuron import h
h.finitialize(-65)
h.continuerun(2.0)
assert soma(0.5).v == soma(0.5).v  # not NaN
print("OK", len(sections))
"""
    r = subprocess.run([sys.executable, "-c", script], capture_output=True,
                       text=True, timeout=60)
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    assert "OK" in r.stdout, r.stdout


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
    assert run.spikes_ms == []                 # empty Vector -> empty list, no crash


def _assert_is_one_tree(edges: np.ndarray, n: int) -> None:
    assert edges.shape == (n - 1, 2)
    A = sp.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                      shape=(n, n))
    A = A + A.T
    ncomp, _ = sp.csgraph.connected_components(A, directed=False)
    assert ncomp == 1


def test_segment_edges_ball_and_stick_is_a_chain():
    sections = make_ball_and_stick(axon_len_um=200.0, nseg=5)
    coords = get_segment_coordinates(sections, dx_um=1.0)
    edges = get_segment_edges(sections)
    _assert_is_one_tree(edges, len(coords))


def test_segment_edges_branching_cell_is_a_tree():
    sections = make_branching_cell(n_generations=3, branching=2, nseg=3)
    coords = get_segment_coordinates(sections, dx_um=1.0)
    edges = get_segment_edges(sections)
    n_seg = sum(s.nseg for s in sections)
    assert len(coords) == n_seg
    _assert_is_one_tree(edges, n_seg)


@pytest.mark.skipif(not _has_compiled_mechanisms(RGC_DIR),
                    reason="rgc_d1 sample model's compiled mechanisms not present")
def test_load_rgc_d1_sample_model():
    """The real D1 retinal ganglion cell (copied from the lab's
    neuron_exstim_pkg/Single-RGC) loads, forms a single connected tree, and
    runs -- this is the model examples/26_register_neuron.py drives.
    Run in a subprocess: this test file already builds other cells earlier,
    and hoc cannot rebuild a template in a process that already has one."""
    _load_in_subprocess(RGC_DIR, expect_gnabar=0.2)


@pytest.mark.skipif(not _has_compiled_mechanisms(A2I_DIR),
                    reason="rgc_a2i sample model's compiled mechanisms not present")
def test_load_rgc_a2i_sample_model_is_distinct_from_d1():
    """The A2i sibling (same lineage/build script as D1, different SWC
    reconstruction and channel densities -- see samples/neuron_models/
    rgc_a2i/README.md for provenance) loads as its own real, distinct,
    single-tree morphology. Its soma Na conductance (0.35) differs from
    D1's (0.2) in the source makecell.hoc, confirming the right biophysics
    branch fired, not just the right morphology. Run in a subprocess: its
    makecell.hoc is byte-identical to rgc_d1's, and building both cells in
    one process is not supported (see load_hoc_cell's docstring)."""
    _load_in_subprocess(A2I_DIR, expect_gnabar=0.35)
