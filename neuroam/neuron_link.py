"""Optional NEURON coupling: play extracellular potentials into a cell.

Requires ``pip install neuron``.  All imports are guarded so the rest of
NeuroAM works without NEURON installed.

The driver reproduces the lab's hoc pattern (``Stim_*_TimeCourse.hoc``):
insert ``extracellular`` in every section, then for each segment play the
precomputed potential time series into ``e_extracellular``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def neuron_available() -> bool:
    try:
        import neuron  # noqa: F401
        return True
    except Exception:
        return False


#: fingerprints (sorted .mod filenames) of mechanism sets already loaded in
#: this process -- NEURON identifies mechanisms by name (a mod file's
#: SUFFIX, not necessarily its filename), and loading a second, separately
#: -compiled DLL that redefines an already-registered name does not just
#: raise a Python exception to catch: it leaves the hoc interpreter unable
#: to build anything afterward, silently. Two sibling cell models copied
#: from the same source (e.g. rgc_d1/rgc_a2i) ship byte-identical .mod
#: files under different paths -- this skips the redundant, unsafe reload
#: rather than relying on error handling after the fact.
_LOADED_MECHANISM_SETS: set = set()

#: content hashes of hoc build scripts already run this process (see the
#: docstring below — a script that ``begintemplate``s a class or declares
#: an ``obfunc`` cannot be run a second time in the same process).
_LOADED_HOC_SCRIPTS: Dict[str, str] = {}


def load_hoc_cell(dirpath, hoc_file: str = "makecell.hoc",
                  mechanisms_dir: Optional[str] = None) -> List:
    """Build a cell from a self-contained hoc model directory.

    The directory is expected to hold a build script (default
    ``makecell.hoc``) that instantiates exactly one cell — e.g. via
    ``Import3d_SWC_read`` from a co-located ``.swc`` — using paths relative
    to itself (this is the convention of the lab's ``neuron_exstim_pkg``
    sample models: ``makecell.hoc`` + ``CellTypes.txt`` + ``morphology/*.swc``
    + compiled mechanisms). Compiled mechanisms (``nrnmech.dll`` /
    ``<arch>/.libs/libnrnmech.so``) are loaded first if present —
    ``mechanisms_dir`` overrides where to look for them (default: the same
    directory), and skipped if a directory with the identical set of
    ``.mod`` filenames was already loaded this process (see
    :data:`_LOADED_MECHANISM_SETS`). ``h.allsec()`` is global NEURON process
    state — anything else already built in this process (another cell, a
    prior test) stays there too — so this returns only the sections newly
    created by this call, not the full registry.

    **Two *different* cells built from a byte-identical build script cannot
    share one process.** A hoc ``begintemplate``/``obfunc`` is a top-level,
    process-global declaration hoc refuses to redefine; running the same
    script content twice does not raise a catchable error, it silently
    aborts partway through the second run (no cell, no clear exception).
    This is a property of hoc itself, not something this function can paper
    over — build sibling cells that share a script (e.g. ``rgc_d1`` and
    ``rgc_a2i``, both copies of the lab's ``makecell.hoc``) in **separate
    processes**. This function detects the specific case of re-running
    identical script content and raises a clear error instead of returning
    an empty, confusing result.
    """
    from neuron import h, load_mechanisms

    dirpath = Path(dirpath)
    mech_dir = Path(mechanisms_dir or dirpath)
    fingerprint = tuple(sorted(p.name for p in mech_dir.glob("*.mod")))
    if fingerprint and fingerprint in _LOADED_MECHANISM_SETS:
        pass  # an identical mechanism set is already registered; loading
              # another compiled copy of it would corrupt hoc's interpreter
              # state (NEURON refuses to redefine a mechanism name, and does
              # not leave the interpreter usable afterward even if the
              # resulting error is caught)
    else:
        try:
            load_mechanisms(str(mech_dir))
            _LOADED_MECHANISM_SETS.add(fingerprint)
        except RuntimeError:
            pass  # already loaded via a path we didn't fingerprint-match
    h.load_file("stdrun.hoc")

    hoc_path = dirpath / hoc_file
    content = hoc_path.read_text()
    if content in _LOADED_HOC_SCRIPTS:
        raise RuntimeError(
            f"{hoc_path} is byte-identical to a build script already run "
            f"in this process (from {_LOADED_HOC_SCRIPTS[content]}). hoc "
            "cannot redefine a template/obfunc a second time -- build this "
            "cell in a separate process instead of a second load_hoc_cell() "
            "call here.")
    _LOADED_HOC_SCRIPTS[content] = str(dirpath)

    before = set(h.allsec())
    cwd = os.getcwd()
    os.chdir(dirpath)
    try:
        h.load_file(hoc_file)
    finally:
        os.chdir(cwd)
    return [sec for sec in h.allsec() if sec not in before]


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


def get_segment_edges(sections) -> np.ndarray:
    """(n_edges, 2) index pairs into the flat segment list of
    :func:`get_segment_coordinates` over the same ``sections`` — the tree
    connectivity needed to draw the morphology as a real branching structure
    rather than a point cloud. Segments within a section form a chain; a
    section's first segment links to whichever of its parent section's
    segments its 0-end actually attaches to (``parentseg()``).
    """
    from neuron import h

    starts: Dict = {}
    start = 0
    for sec in sections:
        starts[sec] = start
        start += sec.nseg

    edges: List[Tuple[int, int]] = []
    known = set(sections)
    for sec in sections:
        base = starts[sec]
        n = sec.nseg
        for i in range(n - 1):
            edges.append((base + i, base + i + 1))
        try:
            pseg = sec.parentseg()
        except Exception:
            pseg = None
        if pseg is None or pseg.sec not in known:
            continue
        psec = pseg.sec
        # nearest actual segment (by seg.x) in the parent section to the
        # arclength position this section actually attaches at
        xs = np.array([seg.x for seg in psec])
        j = int(np.argmin(np.abs(xs - pseg.x)))
        edges.append((base, starts[psec] + j))
    return np.asarray(edges, dtype=np.int64).reshape(-1, 2)


@dataclass
class ExtracellularRun:
    t_ms: np.ndarray
    vm_mV: np.ndarray            # (T, n_segments)
    spikes_ms: List[float]
    segment_names: List[str]


@dataclass
class NeuronRunResult:
    """Everything needed to answer a *different* question about a run, or
    regenerate any of the three neuron views, without touching NEURON or
    re-solving the AM field again.

    This is the save/reload boundary: build one from a live
    :class:`ExtracellularRun` plus the registered geometry right after
    simulating, write it once, and every later question about *this run*
    (peak depolarization and where, spike timing in a specific branch,
    activity replayed with a different colormap or time window, a fresh
    crop for the field view) is answered by loading the file back — the
    expensive parts (AM solve, ``continuerun``) never need to happen again.
    A *different* stimulus/placement/model is a new run, not something this
    can answer after the fact — ``unit_v`` is kept specifically so a new
    *waveform* at the same placement can be re-scaled without a new AM
    solve, but a new NEURON run is still needed to see its Vm response.
    """
    t_ms: np.ndarray
    vm_mV: np.ndarray                          # (T, n_seg)
    spikes_ms: np.ndarray
    seg_vox: np.ndarray                        # (n_seg, 3) registered, voxel units
    edges: np.ndarray                          # (n_seg-1, 2) segment tree
    unit_v: np.ndarray                         # (n_seg,) V per A, field at placement
    swc_vox: Optional[np.ndarray] = None       # (n_swc, 3) dense trace, same frame
    swc_edges: Optional[np.ndarray] = None
    swc_to_seg: Optional[np.ndarray] = None    # (n_swc,) nearest-segment index
    field_crop: Optional[np.ndarray] = None
    """A small per-voxel field array (e.g. ``|J|``) cropped tight around the
    cell -- not the whole model, however big it is -- so the *field* view
    is reproducible too without a fresh AM solve. ``field_crop_origin_vox``
    in ``meta`` is this crop's origin in the original model's voxel frame."""
    meta: Dict = field(default_factory=dict)
    """Free-form provenance: model name/world/dx, cell dir, waveform params,
    registration target, solve stats, spike count -- whatever answers "what
    scenario produced this" later without guessing."""


def save_neuron_run(path, result: NeuronRunResult) -> Path:
    """Write a :class:`NeuronRunResult` to one ``.npz`` file (arrays +
    ``meta`` as an embedded JSON string) -- self-contained, no NEURON
    needed to read it back."""
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = dict(t_ms=result.t_ms, vm_mV=result.vm_mV,
                 spikes_ms=np.asarray(result.spikes_ms, dtype=float),
                 seg_vox=result.seg_vox, edges=result.edges,
                 unit_v=result.unit_v, meta=np.asarray(json.dumps(result.meta)))
    if result.swc_vox is not None:
        arrays["swc_vox"] = result.swc_vox
    if result.swc_edges is not None:
        arrays["swc_edges"] = result.swc_edges
    if result.swc_to_seg is not None:
        arrays["swc_to_seg"] = result.swc_to_seg
    if result.field_crop is not None:
        arrays["field_crop"] = result.field_crop
    np.savez_compressed(path, **arrays)
    return path


def load_neuron_run(path) -> NeuronRunResult:
    """Read back a :class:`NeuronRunResult` written by :func:`save_neuron_run`."""
    import json

    with np.load(path, allow_pickle=False) as d:
        return NeuronRunResult(
            t_ms=d["t_ms"], vm_mV=d["vm_mV"], spikes_ms=d["spikes_ms"],
            seg_vox=d["seg_vox"], edges=d["edges"], unit_v=d["unit_v"],
            swc_vox=d["swc_vox"] if "swc_vox" in d else None,
            swc_edges=d["swc_edges"] if "swc_edges" in d else None,
            swc_to_seg=d["swc_to_seg"] if "swc_to_seg" in d else None,
            field_crop=d["field_crop"] if "field_crop" in d else None,
            meta=json.loads(str(d["meta"])) if "meta" in d else {})


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
        # h.Vector.__iter__ always yields floats, even when empty (no
        # spikes) -- np.array(spikes) on an empty hoc Vector raises
        # "setting an array element with a sequence" on some NEURON/numpy
        # combinations, so iterate the Vector directly rather than round
        # -tripping through numpy.
        spikes_ms=list(spikes),
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


def make_branching_cell(n_generations: int = 3, branching: int = 2,
                        branch_len_um: float = 60.0, nseg: int = 5,
                        axon_len_um: float = 500.0,
                        seed: int = 0) -> List:
    """A synthetic multi-branch cell (HH soma + dendritic tree + axon), pure
    ``h.Section`` — no ``.swc``, no compiled mechanisms, nothing to install
    beyond NEURON itself. For exercising :func:`get_segment_edges` and the
    morphology viewer against a real branching topology fast, in CI or
    offline from any specific lab cell model.
    """
    from neuron import h
    rng = np.random.default_rng(seed)
    h.load_file("stdrun.hoc")

    soma = h.Section(name="soma")
    soma.L = soma.diam = 15.0
    soma.insert("hh")
    sections = [soma]

    axon = h.Section(name="axon")
    axon.L = axon_len_um
    axon.diam = 1.5
    axon.nseg = nseg * 3
    axon.insert("hh")
    axon.connect(soma(0.0))
    sections.append(axon)

    def grow(parent, gen, prefix):
        if gen > n_generations:
            return
        for k in range(branching):
            d = h.Section(name=f"{prefix}{k}")
            d.L = branch_len_um * (0.85 ** gen)
            d.diam = max(0.3, 1.2 * (0.8 ** gen))
            d.nseg = nseg
            d.insert("hh")
            d.gnabar_hh = 0.0  # dendrites: passive-ish, spikes stay axonal
            frac = float(rng.uniform(0.3, 1.0))
            d.connect(parent(frac))
            sections.append(d)
            grow(d, gen + 1, f"{prefix}{k}_")

    grow(soma, 1, "dend_")
    h.define_shape()
    return sections
