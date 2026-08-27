"""Optional NEURON coupling: play extracellular potentials into a cell.

Requires ``pip install neuron``.  All imports are guarded so the rest of
NeuroAM works without NEURON installed.

The driver reproduces the lab's hoc pattern (``Stim_*_TimeCourse.hoc``):
insert ``extracellular`` in every section, then for each segment play the
precomputed potential time series into ``e_extracellular``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def neuron_available() -> bool:
    try:
        import neuron  # noqa: F401
        return True
    except Exception:
        return False


def get_segment_coordinates(sections, dx_um: float,
                            origin_um=(0.0, 0.0, 0.0)) -> np.ndarray:
    """3D coordinates of every segment center, converted to unit-voxel units.

    ``dx_um`` is the field's unit voxel size in micrometers; NEURON section
    coordinates are in micrometers.
    """
    from neuron import h
    pts = []
    for sec in sections:
        n3d = int(h.n3d(sec=sec))
        if n3d < 2:
            raise ValueError(f"section {sec} has no 3D points; call define_shape()")
        xs = np.array([h.x3d(i, sec=sec) for i in range(n3d)])
        ys = np.array([h.y3d(i, sec=sec) for i in range(n3d)])
        zs = np.array([h.z3d(i, sec=sec) for i in range(n3d)])
        ln = np.concatenate([[0.0], np.cumsum(np.sqrt(np.diff(xs) ** 2
                                                      + np.diff(ys) ** 2
                                                      + np.diff(zs) ** 2))])
        total = ln[-1] if ln[-1] > 0 else 1.0
        for seg in sec:
            s = seg.x * total
            pts.append([np.interp(s, ln, xs), np.interp(s, ln, ys),
                        np.interp(s, ln, zs)])
    pts_um = np.asarray(pts) - np.asarray(origin_um)
    return pts_um / dx_um


@dataclass
class ExtracellularRun:
    t_ms: np.ndarray
    vm_mV: np.ndarray            # (T, n_segments)
    spikes_ms: List[float]
    segment_names: List[str]


def run_extracellular(sections, v_matrix_mV: np.ndarray, dt_ms: float,
                      tstop_ms: Optional[float] = None, v_init: float = -65.0,
                      record_segments: Optional[Sequence[int]] = None,
                      spike_section_index: int = 0,
                      spike_threshold_mV: float = 0.0,
                      celsius: Optional[float] = None) -> ExtracellularRun:
    """Apply per-segment extracellular potentials and run NEURON.

    ``v_matrix_mV``: (T, n_segments) — column order must match the segment
    order of ``get_segment_coordinates`` over the same section list.
    """
    from neuron import h
    h.load_file("stdrun.hoc")

    segs = [seg for sec in sections for seg in sec]
    n_seg = len(segs)
    if v_matrix_mV.shape[1] != n_seg:
        raise ValueError(f"v matrix has {v_matrix_mV.shape[1]} columns, "
                         f"model has {n_seg} segments")

    for sec in sections:
        if not sec.has_membrane("extracellular"):
            sec.insert("extracellular")

    T = v_matrix_mV.shape[0]
    tvec = h.Vector(np.arange(T) * dt_ms)
    players = []
    for j, seg in enumerate(segs):
        vec = h.Vector(np.ascontiguousarray(v_matrix_mV[:, j]))
        vec.play(seg._ref_e_extracellular, tvec, 1)   # linear interp
        players.append(vec)

    rec_idx = list(record_segments) if record_segments is not None \
        else list(range(n_seg))
    rec_t = h.Vector().record(h._ref_t)
    recs = [h.Vector().record(segs[j]._ref_v) for j in rec_idx]

    apc_seg = segs[spike_section_index]
    apc = h.APCount(apc_seg)
    apc.thresh = spike_threshold_mV
    spikes = h.Vector()
    apc.record(spikes)

    if celsius is not None:
        h.celsius = celsius
    h.dt = dt_ms
    h.finitialize(v_init)
    h.continuerun(tstop_ms if tstop_ms is not None else T * dt_ms)

    return ExtracellularRun(
        t_ms=np.array(rec_t),
        vm_mV=np.column_stack([np.array(r) for r in recs]),
        spikes_ms=list(np.array(spikes)),
        segment_names=[str(segs[j]) for j in rec_idx],
    )


def make_ball_and_stick(axon_len_um: float = 1000.0, nseg: int = 51):
    """A simple demo cell (HH soma + axon), for coupling tests and examples."""
    from neuron import h
    h.load_file("stdrun.hoc")
    soma = h.Section(name="soma")
    soma.L = soma.diam = 20.0
    soma.insert("hh")
    axon = h.Section(name="axon")
    axon.L = axon_len_um
    axon.diam = 2.0
    axon.nseg = nseg
    axon.insert("hh")
    axon.connect(soma(1))
    h.define_shape()
    return [soma, axon]
