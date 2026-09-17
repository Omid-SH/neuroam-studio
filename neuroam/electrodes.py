"""Electrode design, registration and quality control.

An electrode is a :class:`~neuroam.geometry.Region` in millimetres plus an
electrical *terminal model*.  Registering it onto a model paints its label
into a **sparse overlay** rather than rewriting the anatomy, so a montage is
``anatomy + overlays`` and a 6.9 GB ``.model`` never has to be duplicated per
montage.

Terminal models
---------------
``supernode``   All lattice nodes of the electrode body are merged into a
                single unknown -- an ideal equipotential metal contact.  The
                solve then yields the electrode potential directly, hence its
                access impedance, and the result no longer depends on which
                node the current happens to be injected at.  Default.
``node``        Legacy behaviour: current is injected at one node and the
                metal resistivity (1e-7 ohm*m) does the equipotentializing.
                Kept so archived runs reproduce bit-for-bit.
``distributed`` Current is split over the electrode's nodes in proportion to
                their share of the exposed surface (no equipotential
                constraint) -- a current-source array rather than a metal.

Quality control
---------------
:func:`register` returns an :class:`ElectrodeQC` per electrode: painted
volume, connected components, exposed surface area, which tissues it actually
touches, overlap with other electrodes, and -- the check that would have
caught the legacy ``sample.in`` ground-node bug -- whether the terminal node
is genuinely inside the electrode it claims to drive.

:meth:`ElectrodeQC.check_contact` verifies which of two physically distinct
placements a spec actually produced -- ``"surface"`` (resting on a tissue,
e.g. a skin pad) vs. ``"inserted"`` (the footprint itself replaces that
tissue, e.g. a subdermal or intracorneal contact) -- rather than assuming the
geometry spec did what it was meant to. See ``neuroam.safety`` for the
charge-density safety check this QC feeds into.
"""

from __future__ import annotations

import hashlib
import math
import warnings
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from .geometry import (Grid, Region, SDF, MaskRegion, rasterize,
                       Sphere, Box, Cylinder, Capsule, Cone, Torus, Disc,
                       Annulus, SphericalShell, SphericalBand, SphericalCap,
                       Needle, Cuff, Polyline)
from .materials import Material, METAL_RHO, INSULATOR_RHO
from .model import VoxelModel, Source, node_name

__all__ = ["ElectrodeSpec", "ElectrodeQC", "ContactCheck", "Overlay", "Montage",
           "Terminal", "register", "region_from_spec", "electrode_nodes",
           "central_node", "dice", "compare_masks", "fit_sphere",
           "fit_spherical_band", "fit_tube_path", "refine_by_dice"]

_EPS = 1e-12


# ------------------------------------------------------------------ terminal
@dataclass
class Terminal:
    """Electrical connection of an electrode to the admittance system."""

    name: str
    role: str                      # 'source' | 'ground'
    kind: str                      # 'supernode' | 'node' | 'distributed'
    nodes: np.ndarray              # (M, 3) lattice node coordinates
    waveform: Optional[str] = None
    weights: Optional[np.ndarray] = None    # distributed only, sums to 1

    @property
    def node(self) -> Tuple[int, int, int]:
        return tuple(int(v) for v in self.nodes[0])


# ------------------------------------------------------------------- specs
@dataclass
class ElectrodeSpec:
    """A designed electrode: where it is, what it is made of, how it connects."""

    name: str
    region: Optional[Region] = None
    role: str = "source"                 # source | ground | recording
    label: int = 101
    terminal: str = "supernode"          # supernode | node | distributed
    node: Optional[Sequence[int]] = None  # explicit terminal node (kind='node')
    waveform: Optional[str] = None
    rho: float = METAL_RHO
    insulation: Optional[Region] = None
    insulation_label: int = 110
    insulation_rho: float = INSULATOR_RHO
    overwrite: Union[str, Sequence[int]] = "any"   # any | background | [labels]
    from_label: Optional[int] = None     # adopt voxels already painted with this
    supersample: int = 3
    lattice: int = 1                     # authoring lattice, see geometry.rasterize
    min_voxels: int = 1
    notes: str = ""

    def key(self) -> str:
        """Content hash of the spec, for overlay caching."""
        d = {"name": self.name, "role": self.role, "label": self.label,
             "terminal": self.terminal, "node": list(self.node) if self.node else None,
             "rho": self.rho, "overwrite": self.overwrite,
             "from_label": self.from_label, "supersample": self.supersample,
             "lattice": self.lattice,
             "region": self.region.describe() if self.region is not None else None,
             "insulation": self.insulation.describe() if self.insulation is not None else None}
        return hashlib.sha1(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]


@dataclass
class ElectrodeQC:
    """What actually got painted, and whether it makes sense."""

    name: str
    role: str
    label: int
    terminal: str
    n_voxels: int
    volume_mm3: float
    bbox_vox: Tuple[Tuple[int, int, int], Tuple[int, int, int]]
    centroid_vox: Tuple[float, float, float]
    centroid_mm: Tuple[float, float, float]
    n_components: int
    exposed_faces: int
    surface_area_mm2: float
    contact_by_label: Dict[int, int]
    contact_area_mm2: Dict[int, float]
    n_requested: int
    n_blocked: int
    overlaps: Dict[str, int]
    terminal_node: Optional[Tuple[int, int, int]]
    n_terminal_nodes: int
    terminal_inside: bool
    warnings: List[str] = field(default_factory=list)
    replaced_by_label: Dict[int, int] = field(default_factory=dict)
    """What tissue occupied the electrode's *own* footprint before painting
    (voxel counts, pre-overwrite) -- the "inserted into" signal, distinct
    from :attr:`contact_by_label` (what borders the painted footprint from
    outside -- the "resting on" signal). Together these are what
    :meth:`contact_mode` and :meth:`check_contact` classify."""

    def charge_density(self, current_A: float, pulse_width_s: float) -> float:
        """Charge density per phase in uC/cm^2 over the exposed surface."""
        if self.surface_area_mm2 <= 0:
            return float("nan")
        area_cm2 = self.surface_area_mm2 * 1e-2
        return abs(current_A) * pulse_width_s * 1e6 / area_cm2

    def current_density(self, current_A: float) -> float:
        """Mean current density at the exposed surface, A/m^2."""
        if self.surface_area_mm2 <= 0:
            return float("nan")
        return abs(current_A) / (self.surface_area_mm2 * 1e-6)

    def contact_mode(self, target_label: int, min_fraction: float = 0.3) -> str:
        """How this electrode relates to ``target_label`` tissue.

        - ``"inserted"``  -- the electrode's own footprint *replaced*
          ``target_label`` voxels (``replaced_by_label``) for at least
          ``min_fraction`` of its volume: a subdermal/intracorneal contact.
        - ``"surface"``   -- ``target_label`` borders the painted footprint
          (``contact_by_label``) but the electrode did not replace it: a
          contact resting on top of the tissue.
        - ``"floating"``  -- ``target_label`` appears in neither: there is
          an air/gap buffer (or some other tissue) between the electrode and
          this target, so it is not actually contacting it at all.
        - ``"absent"``    -- the electrode itself painted 0 voxels (see
          ``warnings``); no contact judgement is possible.
        """
        if self.n_voxels == 0:
            return "absent"
        n_replaced = self.replaced_by_label.get(target_label, 0)
        if self.n_voxels and n_replaced / self.n_voxels >= min_fraction:
            return "inserted"
        if self.contact_by_label.get(target_label, 0) > 0:
            return "surface"
        return "floating"

    def check_contact(self, target_label: int, mode: str = "surface",
                      min_area_mm2: float = 0.0, min_fraction: float = 0.3
                      ) -> "ContactCheck":
        """Verify this electrode is actually in the requested contact mode
        with ``target_label`` -- the check the user-facing request asked
        for: "make sure the electrode is completely touching the skin, or
        inside it" as two distinct, checkable scenarios, not an assumption
        baked into the geometry spec.

        ``mode``: ``"surface"`` (resting on ``target_label``, not replacing
        it) or ``"inserted"`` (the footprint itself is carved out of
        ``target_label``). Returns a :class:`ContactCheck`; nothing raises --
        inspect ``.ok`` and ``.reason``, or use it as a QC gate before a
        solve (see ``neuroam.safety``).
        """
        actual = self.contact_mode(target_label, min_fraction=min_fraction)
        area = self.contact_area_mm2.get(target_label, 0.0)
        if mode not in ("surface", "inserted"):
            raise ValueError(f"mode must be 'surface' or 'inserted', got {mode!r}")
        if actual == "absent":
            ok, reason = False, f"{self.name}: electrode painted 0 voxels"
        elif actual == "floating":
            ok, reason = False, (f"{self.name}: not touching label {target_label} at all "
                                 f"(0 contact area) -- there is a gap")
        elif actual != mode:
            ok, reason = False, (f"{self.name}: touches label {target_label} as "
                                 f"{actual!r}, not the requested {mode!r}")
        elif mode == "surface" and area < min_area_mm2:
            ok, reason = False, (f"{self.name}: surface contact area {area:.4g} mm^2 "
                                 f"is below the required {min_area_mm2:.4g} mm^2")
        else:
            ok, reason = True, (f"{self.name}: {actual} contact with label "
                                f"{target_label}, {area:.4g} mm^2")
        return ContactCheck(name=self.name, target_label=target_label,
                            requested_mode=mode, actual_mode=actual,
                            contact_area_mm2=area, ok=ok, reason=reason)

    def summary(self) -> str:
        lo, hi = self.bbox_vox
        contacts = ", ".join(f"{k}:{v}" for k, v in
                             sorted(self.contact_by_label.items(), key=lambda kv: -kv[1])[:6])
        s = (f"{self.name} [{self.role}/{self.terminal}] label {self.label}: "
             f"{self.n_voxels:,} vox ({self.volume_mm3:.4g} mm^3), "
             f"area {self.surface_area_mm2:.4g} mm^2, {self.n_components} component(s)\n"
             f"    bbox {list(lo)}..{list(hi)}  centroid {np.round(self.centroid_vox, 1).tolist()}\n"
             f"    touches [{contacts}]  terminal {self.terminal_node} "
             f"({self.n_terminal_nodes} node(s), inside={self.terminal_inside})")
        if self.n_blocked:
            s += f"\n    {self.n_blocked:,} voxel(s) blocked by the overwrite policy"
        for w in self.warnings:
            s += f"\n    !! {w}"
        return s


@dataclass
class ContactCheck:
    """Result of :meth:`ElectrodeQC.check_contact` -- pass/fail plus why."""

    name: str
    target_label: int
    requested_mode: str
    actual_mode: str
    contact_area_mm2: float
    ok: bool
    reason: str

    def __bool__(self) -> bool:
        return self.ok


# ----------------------------------------------------------------- overlay
@dataclass
class Overlay:
    """Sparse label patch: the montage's electrodes, without touching anatomy."""

    idx: np.ndarray                       # (N, 3) int32 voxel indices
    labels: np.ndarray                    # (N,) int32
    owner: np.ndarray                     # (N,) int16 index into names
    names: List[str] = field(default_factory=list)
    base: str = ""                        # provenance of the anatomy it targets
    world: Optional[Tuple[int, int, int]] = None

    def __len__(self) -> int:
        return len(self.idx)

    @classmethod
    def empty(cls, world=None, base: str = "") -> "Overlay":
        return cls(np.zeros((0, 3), np.int32), np.zeros(0, np.int32),
                   np.zeros(0, np.int16), [], base, world)

    def add(self, idx: np.ndarray, label: int, name: str) -> None:
        if name not in self.names:
            self.names.append(name)
        o = self.names.index(name)
        idx = np.asarray(idx, np.int32).reshape(-1, 3)
        self.idx = np.vstack([self.idx, idx])
        self.labels = np.concatenate([self.labels,
                                      np.full(len(idx), label, np.int32)])
        self.owner = np.concatenate([self.owner, np.full(len(idx), o, np.int16)])

    def of(self, name: str) -> np.ndarray:
        if name not in self.names:
            return np.zeros((0, 3), np.int32)
        return self.idx[self.owner == self.names.index(name)]

    def apply(self, labels: np.ndarray, copy: bool = False) -> np.ndarray:
        out = labels.copy() if copy else labels
        if len(self.idx):
            out[self.idx[:, 0], self.idx[:, 1], self.idx[:, 2]] = self.labels
        return out

    def save(self, path) -> Path:
        path = Path(path)
        np.savez_compressed(path, idx=self.idx, labels=self.labels,
                            owner=self.owner, names=np.array(self.names, dtype=object),
                            base=self.base,
                            world=np.array(self.world if self.world else (0, 0, 0)))
        return path

    @classmethod
    def load(cls, path) -> "Overlay":
        z = np.load(path, allow_pickle=True)
        w = tuple(int(v) for v in z["world"])
        return cls(z["idx"], z["labels"], z["owner"], list(z["names"]),
                   str(z["base"]), None if w == (0, 0, 0) else w)


# ------------------------------------------------------------- node helpers
def electrode_nodes(idx: np.ndarray, world: Sequence[int]) -> np.ndarray:
    """All lattice nodes touched by a set of voxels (the 8 corners of each)."""
    if len(idx) == 0:
        return np.zeros((0, 3), np.int64)
    offs = np.array([[a, b, c] for a in (0, 1) for b in (0, 1) for c in (0, 1)])
    n = (idx[:, None, :] + offs[None, :, :]).reshape(-1, 3)
    nx, ny, nz = world
    packed = (n[:, 0].astype(np.int64) * (ny + 1) + n[:, 1]) * (nz + 1) + n[:, 2]
    _, first = np.unique(packed, return_index=True)
    return n[np.sort(first)].astype(np.int64)


def central_node(idx: np.ndarray, world: Sequence[int]) -> Tuple[int, int, int]:
    """The electrode node nearest the electrode centroid.

    Deterministic and always inside the body -- unlike the legacy
    "middle element of ``argwhere``" rule, which depends on scan order and
    silently lands outside non-convex electrodes.
    """
    nodes = electrode_nodes(idx, world)
    c = idx.mean(0) + 0.5
    return tuple(int(v) for v in nodes[np.argmin(((nodes - c) ** 2).sum(1))])


def _components(idx: np.ndarray, shape) -> int:
    """Electrically connected pieces of an electrode body.

    26-connectivity is the right rule here: in the admittance formulation two
    voxels that share only an edge or a corner still share *lattice nodes*, so
    they are one conductor.  Counting with 6-connectivity would report a
    diagonally-linked electrode as several floating pieces.
    """
    from scipy import ndimage
    if len(idx) == 0:
        return 0
    lo = idx.min(0); hi = idx.max(0) + 1
    sub = np.zeros(hi - lo, dtype=bool)
    sub[idx[:, 0] - lo[0], idx[:, 1] - lo[1], idx[:, 2] - lo[2]] = True
    return int(ndimage.label(sub, structure=np.ones((3, 3, 3), bool))[1])


def _exposed(idx: np.ndarray, member: np.ndarray, labels: np.ndarray):
    """Faces of the electrode that face something else; count by neighbour label."""
    faces = 0
    by_label: Dict[int, int] = {}
    shape = np.asarray(labels.shape)
    for d in ([1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]):
        q = idx + np.array(d)
        inb = np.all((q >= 0) & (q < shape), axis=1)
        qi = q[inb]
        own = member[qi[:, 0], qi[:, 1], qi[:, 2]]
        out = qi[~own]
        faces += int((~own).sum()) + int((~inb).sum())
        if len(out):
            u, c = np.unique(labels[out[:, 0], out[:, 1], out[:, 2]], return_counts=True)
            for k, v in zip(u.tolist(), c.tolist()):
                by_label[int(k)] = by_label.get(int(k), 0) + int(v)
    return faces, by_label


# -------------------------------------------------------------- registration
def register(model: VoxelModel, specs: Sequence[ElectrodeSpec],
             overlay: Optional[Overlay] = None, apply: bool = True,
             set_terminals: bool = True, clear_existing: bool = True
             ) -> Tuple[Overlay, List[ElectrodeQC]]:
    """Rasterize, paint and connect a set of electrodes.

    Returns the sparse :class:`Overlay` and one :class:`ElectrodeQC` per
    electrode.  With ``apply=True`` the labels are composited into
    ``model.labels`` and (``set_terminals=True``) the model's sources, ground
    nodes and terminals are set from the specs.
    """
    grid = Grid.from_model(model)
    world = model.world
    overlay = overlay or Overlay.empty(world=world, base=model.name)
    base_labels = model.labels.copy()
    qcs: List[ElectrodeQC] = []
    painted: Dict[str, np.ndarray] = {}
    terminals: List[Terminal] = []

    if clear_existing and set_terminals:
        model.sources = []
        model.ground_nodes = []

    for spec in specs:
        # ---- geometry -> voxels
        if spec.from_label is not None:
            idx = np.argwhere(base_labels == spec.from_label).astype(np.int64)
            n_req = len(idx)
        elif spec.region is not None:
            idx, _, _ = rasterize(spec.region, grid, supersample=spec.supersample,
                                  lattice=spec.lattice)
            n_req = len(idx)
        else:
            raise ValueError(f"electrode {spec.name!r}: give region= or from_label=")

        # ---- overwrite policy
        n_blocked = 0
        if len(idx) and spec.from_label is None:
            here = base_labels[idx[:, 0], idx[:, 1], idx[:, 2]]
            if spec.overwrite == "background":
                allow = (here == 0)
            elif spec.overwrite == "any":
                allow = np.ones(len(idx), bool)
            else:
                allow = np.isin(here, list(spec.overwrite))
            n_blocked = int((~allow).sum())
            idx = idx[allow]

        warnings: List[str] = []
        if len(idx) < spec.min_voxels:
            warnings.append(f"only {len(idx)} voxel(s) painted "
                            f"(min_voxels={spec.min_voxels}) — check units/placement")

        # ---- collisions with electrodes already placed
        overlaps: Dict[str, int] = {}
        if len(idx):
            key = set(map(tuple, idx.tolist()))
            for other, oidx in painted.items():
                n = len(key & set(map(tuple, oidx.tolist())))
                if n:
                    overlaps[other] = n
                    warnings.append(f"shares {n} voxel(s) with {other!r} — "
                                    f"electrodes are shorted")
        painted[spec.name] = idx

        # ---- paint
        if len(idx):
            overlay.add(idx, spec.label, spec.name)
            if apply:
                model.labels[idx[:, 0], idx[:, 1], idx[:, 2]] = spec.label
        if spec.label not in model.materials:
            model.materials.add(Material.isotropic_rho(
                spec.label, spec.rho, name=f"{spec.name} ({spec.role})"))

        if spec.insulation is not None:
            iidx, _, _ = rasterize(spec.insulation, grid, supersample=spec.supersample,
                                   lattice=spec.lattice)
            if len(iidx):
                keep = ~np.isin(
                    (iidx[:, 0] * world[1] + iidx[:, 1]) * world[2] + iidx[:, 2],
                    (idx[:, 0] * world[1] + idx[:, 1]) * world[2] + idx[:, 2]
                    if len(idx) else np.zeros(0, np.int64))
                iidx = iidx[keep]
                overlay.add(iidx, spec.insulation_label, spec.name + ":insulation")
                if apply:
                    model.labels[iidx[:, 0], iidx[:, 1], iidx[:, 2]] = spec.insulation_label
            if spec.insulation_label not in model.materials:
                model.materials.add(Material.isotropic_rho(
                    spec.insulation_label, spec.insulation_rho,
                    name=f"{spec.name} insulation"))

        # ---- terminal
        term_node = None
        nodes = np.zeros((0, 3), np.int64)
        inside = False
        if len(idx):
            nodes = electrode_nodes(idx, world)
            if spec.node is not None:
                term_node = tuple(int(v) for v in spec.node)
                inside = bool((nodes == np.array(term_node)).all(1).any())
                if not inside:
                    warnings.append(
                        f"declared terminal node {term_node} is NOT a node of "
                        f"electrode {spec.name!r} — current would be injected "
                        f"into whatever tissue is there")
            else:
                term_node = central_node(idx, world)
                inside = True

        if set_terminals and len(idx) and spec.role in ("source", "ground"):
            kind = spec.terminal
            t = Terminal(name=spec.name, role=spec.role, kind=kind,
                         nodes=nodes if kind != "node" else np.array([term_node]),
                         waveform=spec.waveform or spec.name)
            if kind == "distributed":
                t.weights = np.full(len(t.nodes), 1.0 / len(t.nodes))
            terminals.append(t)
            if spec.role == "source":
                model.sources.append(Source(name=t.waveform, node=term_node))
            else:
                model.ground_nodes.extend(
                    [tuple(int(v) for v in n) for n in t.nodes]
                    if kind != "node" else [term_node])

        # ---- QC
        if len(idx):
            member = np.zeros(model.labels.shape, dtype=bool)
            member[idx[:, 0], idx[:, 1], idx[:, 2]] = True
            faces, by_label = _exposed(idx, member, base_labels)
            ncomp = _components(idx, model.labels.shape)
            # what tissue this electrode's own footprint replaced (pre-paint) --
            # the "inserted into" signal, vs. by_label above (the "rests on")
            replaced_here = base_labels[idx[:, 0], idx[:, 1], idx[:, 2]]
            ru, rc = np.unique(replaced_here, return_counts=True)
            replaced_by_label = {int(k): int(v) for k, v in zip(ru.tolist(), rc.tolist())}
            if ncomp > 1:
                if spec.terminal == "node":
                    warnings.append(
                        f"{ncomp} electrically separate pieces, but the "
                        f"'node' terminal drives only the one containing "
                        f"{term_node}; the rest float")
                else:
                    warnings.append(f"{ncomp} electrically separate pieces "
                                    f"(tied together by the {spec.terminal} terminal)")
            a = grid.dx_mm ** 2
            qc = ElectrodeQC(
                name=spec.name, role=spec.role, label=spec.label,
                terminal=spec.terminal, n_voxels=len(idx),
                volume_mm3=len(idx) * grid.dx_mm ** 3,
                bbox_vox=(tuple(idx.min(0).tolist()), tuple(idx.max(0).tolist())),
                centroid_vox=tuple(idx.mean(0).tolist()),
                centroid_mm=tuple(grid.centers_mm(idx).mean(0).tolist()),
                n_components=ncomp, exposed_faces=faces,
                surface_area_mm2=faces * a,
                contact_by_label={k: v for k, v in by_label.items()},
                contact_area_mm2={k: v * a for k, v in by_label.items()},
                n_requested=n_req, n_blocked=n_blocked, overlaps=overlaps,
                terminal_node=term_node, n_terminal_nodes=len(nodes),
                terminal_inside=inside, warnings=warnings,
                replaced_by_label=replaced_by_label)
        else:
            qc = ElectrodeQC(
                name=spec.name, role=spec.role, label=spec.label,
                terminal=spec.terminal, n_voxels=0, volume_mm3=0.0,
                bbox_vox=((0, 0, 0), (0, 0, 0)), centroid_vox=(0, 0, 0),
                centroid_mm=(0, 0, 0), n_components=0, exposed_faces=0,
                surface_area_mm2=0.0, contact_by_label={}, contact_area_mm2={},
                n_requested=n_req, n_blocked=n_blocked, overlaps={},
                terminal_node=None, n_terminal_nodes=0, terminal_inside=False,
                warnings=warnings + ["electrode is empty"], replaced_by_label={})
        qcs.append(qc)

    if set_terminals:
        model.terminals = terminals
    return overlay, qcs


# ------------------------------------------------------------------ montage
@dataclass
class Montage:
    """Anatomy + electrode overlays + terminal assignment, as one artefact."""

    name: str
    electrodes: List[ElectrodeSpec]
    overlay: Optional[Overlay] = None
    qc: List[ElectrodeQC] = field(default_factory=list)
    base_in: str = ""

    def build(self, model: VoxelModel, **kw) -> "Montage":
        self.overlay, self.qc = register(model, self.electrodes, **kw)
        self.base_in = self.base_in or model.name
        return self

    def report(self) -> str:
        head = f"montage {self.name!r} — {len(self.electrodes)} electrode(s)"
        return "\n".join([head] + [q.summary() for q in self.qc])

    @property
    def warnings(self) -> List[str]:
        return [f"{q.name}: {w}" for q in self.qc for w in q.warnings]

    def save(self, out_dir) -> Path:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        if self.overlay is not None:
            self.overlay.save(out_dir / f"{self.name}.overlay.npz")
        meta = {"name": self.name, "base": self.base_in,
                "electrodes": [{"name": s.name, "role": s.role, "label": s.label,
                                "terminal": s.terminal, "waveform": s.waveform,
                                "key": s.key(),
                                "region": s.region.describe() if s.region is not None else None}
                               for s in self.electrodes],
                "qc": [{k: v for k, v in asdict(q).items()} for q in self.qc]}
        p = out_dir / f"{self.name}.montage.json"
        p.write_text(json.dumps(meta, indent=2, default=str))
        return p

    def write_legacy(self, model: VoxelModel, out_dir, name: Optional[str] = None):
        """Bake anatomy+overlay to ``.model``/``.in`` for the legacy mesher."""
        return model.save_legacy(out_dir, name or self.name)


# ------------------------------------------------------- config -> geometry
_PRIMS = {
    "sphere": lambda d: Sphere(d["radius_mm"]),
    "box": lambda d: Box(d["size_mm"]),
    "cylinder": lambda d: Cylinder(d["radius_mm"], d["height_mm"]),
    "disc": lambda d: Cylinder(d["radius_mm"], d.get("thickness_mm", d.get("height_mm"))),
    "capsule": lambda d: Capsule(d["radius_mm"], d["length_mm"]),
    "cone": lambda d: Cone(d["radius_mm"], d["height_mm"]),
    "torus": lambda d: Torus(d["ring_radius_mm"], d["tube_radius_mm"]),
    "ring": lambda d: Torus(d["ring_radius_mm"], d["tube_radius_mm"]),
    "annulus": lambda d: Annulus(d["inner_radius_mm"], d["outer_radius_mm"],
                                 d["thickness_mm"]),
    "spherical_shell": lambda d: SphericalShell(d["radius_mm"], d["thickness_mm"]),
    "spherical_cap": lambda d: SphericalCap(d["radius_mm"], d["thickness_mm"],
                                            d["half_angle_deg"]),
    "spherical_band": lambda d: SphericalBand(d["radius_mm"], d["thickness_mm"],
                                              d["theta_max_deg"],
                                              d.get("theta_min_deg", 0.0)),
    "needle": lambda d: Needle(d["radius_mm"], d["length_mm"],
                               d.get("tip_length_mm", 0.0)),
    "cuff": lambda d: Cuff(d["inner_radius_mm"], d["thickness_mm"],
                           d["length_mm"], d.get("gap_deg", 0.0)),
    "polyline": lambda d: Polyline(d["points_mm"], d["radius_mm"]),
}


def region_from_spec(d: dict, model: VoxelModel,
                     frames: Optional[Dict[str, "object"]] = None) -> Region:
    """Build a region from a JSON-friendly dict.

    Example::

        {"type": "torus", "ring_radius_mm": 2.7, "tube_radius_mm": 0.15,
         "frame": "eye", "at": {"axial_mm": 2.0}}

    Composite forms: ``"type"`` one of ``"union"``, ``"intersect"``,
    ``"difference"``, each with ``"a"``/``"b"`` sub-specs; anatomy:
    ``{"type": "labels", "labels": [...]}`` and
    ``{"type": "surface_band", "labels": [...], "inner_mm": 0, "outer_mm": 0.2}``.
    Any region may also carry ``"conform"``/``"snap"`` post-processing.
    """
    from .frames import Frame, surface_band, conform, label_mask, snap_to_surface

    frames = frames or {}
    t = d["type"]

    if t in ("union", "intersect", "difference"):
        a = region_from_spec(d["a"], model, frames)
        b = region_from_spec(d["b"], model, frames)
        return {"union": lambda: a | b, "intersect": lambda: a & b,
                "difference": lambda: a - b}[t]()
    if t == "labels":
        return MaskRegion(label_mask(model, d["labels"]), Grid.from_model(model),
                          name=f"labels{d['labels']}")
    if t == "surface_band":
        return surface_band(model, d["labels"], d.get("inner_mm", 0.0),
                            d.get("outer_mm", 0.2))

    if t == "arc":
        # An arc is defined *on* a frame's sphere, so it comes out already
        # placed in world coordinates; skip the generic placement below.
        fr = frames.get(d.get("frame", ""))
        if fr is None:
            raise ValueError("arc regions need a 'frame' (e.g. the eye frame)")
        region = fr.arc(d["r_mm"], d["theta_deg"], d["tube_radius_mm"],
                        d.get("phi0_deg", 0.0), d.get("span_deg", 360.0),
                        d.get("n_points", 81))
        if d.get("conform"):
            c = d["conform"]
            region = conform(region, model, c["labels"], c.get("offset_mm", 0.0),
                             c.get("thickness_mm", 0.1))
        return region

    if t not in _PRIMS:
        raise ValueError(f"unknown region type {t!r}")
    sdf: SDF = _PRIMS[t](d)

    # -- placement
    frame = frames.get(d["frame"]) if d.get("frame") else None
    at = d.get("at", {})
    if frame is not None and at.get("on_surface"):
        # Anatomical address for a *surface* contact: march out from the frame
        # origin along (theta, phi), land on the named tissue, and orient the
        # body along that tissue's outward normal.  Unlike a fixed r_mm this
        # carries across resolutions and species, where the skin sits at a
        # different distance from the globe centre.
        from .frames import surface_point, normal_at
        os_ = at["on_surface"]
        direction = frame.direction(at.get("theta_deg", 0.0), at.get("phi_deg", 0.0))
        P = surface_point(model, os_["labels"], frame.origin_mm, direction,
                          max_mm=os_.get("max_mm"))
        if P is None:
            raise ValueError(
                f"no {list(os_['labels'])} tissue along theta="
                f"{at.get('theta_deg', 0.0)} phi={at.get('phi_deg', 0.0)} "
                f"in frame {frame.name!r} — the model may be cropped there")
        n = normal_at(model, os_["labels"], P, os_.get("smooth_vox", 1.5))
        sdf = sdf.place(P + n * float(os_.get("standoff_mm", 0.0)), n,
                        at.get("roll_deg", 0.0))
    elif frame is not None:
        if "theta_deg" in at or "r_mm" in at:
            sdf = frame.place(sdf, r_mm=at.get("r_mm", 0.0),
                              theta_deg=at.get("theta_deg", 0.0),
                              phi_deg=at.get("phi_deg", 0.0),
                              roll_deg=at.get("roll_deg", 0.0),
                              axis=at.get("axis"))
        else:
            origin = frame.along(at.get("axial_mm", 0.0),
                                 at.get("lateral_mm", 0.0), at.get("phi_deg", 0.0))
            axis = at.get("axis", frame.e3)
            sdf = sdf.place(origin, axis, at.get("roll_deg", 0.0))
    else:
        origin_mm = at.get("origin_mm")
        if origin_mm is None and at.get("origin_vox") is not None:
            origin_mm = Grid.from_model(model).centers_mm(
                np.asarray(at["origin_vox"], float) - 0.5 + 0.5)
        if origin_mm is not None:
            sdf = sdf.place(origin_mm, at.get("axis", (0, 0, 1)),
                            at.get("roll_deg", 0.0))

    if d.get("snap"):
        s = d["snap"]
        direction = s.get("direction")
        if direction is None and frame is not None:
            direction = frame.direction(at.get("theta_deg", 0.0), at.get("phi_deg", 0.0))
        sdf = snap_to_surface(sdf, model, s["labels"], direction,
                              standoff_mm=s.get("standoff_mm", 0.0))

    region: Region = sdf
    if d.get("conform"):
        c = d["conform"]
        region = conform(region, model, c["labels"], c.get("offset_mm", 0.0),
                         c.get("thickness_mm", 0.1))
    return region


def specs_from_config(entries: Sequence[dict], model: VoxelModel,
                      frames: Optional[Dict[str, object]] = None
                      ) -> List[ElectrodeSpec]:
    """Build :class:`ElectrodeSpec` objects from the config ``electrodes`` block."""
    out: List[ElectrodeSpec] = []
    for e in entries:
        region = None
        if e.get("geometry") is not None:
            try:
                region = region_from_spec(e["geometry"], model, frames)
            except ValueError as exc:
                # One montage, many anatomies: a contact that has no landing site
                # in *this* model (a crop that stops short, a species whose orbit
                # is shaped differently) is dropped rather than failing the run,
                # so the same montage file can be carried across models.
                if not e.get("skip_if_missing"):
                    raise
                warnings.warn(f"electrode {e['name']!r} skipped: {exc}",
                              RuntimeWarning, stacklevel=2)
                continue
        insulation = None
        if e.get("insulation") is not None:
            insulation = region_from_spec(e["insulation"], model, frames)
        out.append(ElectrodeSpec(
            name=e["name"], region=region, role=e.get("role", "source"),
            label=int(e.get("label", 101)),
            terminal=e.get("terminal", "supernode"),
            node=e.get("node"), waveform=e.get("waveform"),
            rho=float(e.get("rho", METAL_RHO)),
            insulation=insulation,
            insulation_label=int(e.get("insulation_label", 110)),
            overwrite=e.get("overwrite", "any"),
            from_label=e.get("from_label"),
            supersample=int(e.get("supersample", 3)),
            lattice=int(e.get("lattice", 1)),
            min_voxels=int(e.get("min_voxels", 1)),
            notes=e.get("notes", "")))
    return out


# --------------------------------------------------------- fitting & compare
def close_eyelid(model: VoxelModel, frame, apex_radius_mm: float,
                 theta_max_deg: float, thickness_mm: float = 0.3,
                 r_offset_mm: float = 0.02, skin_label: int = 13,
                 overwrite: Sequence[int] = (0,)) -> int:
    """Paint a thin eyelid-skin shell over the exposed cornea -- the literal
    "eye closed" anatomy variant: a montage built on the model *before* this
    call sees an open eye (cornea exposed to air/skin-pad electrodes reach it
    directly); the identical montage re-registered *after* this call sees the
    same electrodes now separated from the cornea by a layer of eyelid skin,
    the way they would be with the lid shut.

    Geometry: a :class:`~neuroam.geometry.SphericalBand` dome (radius
    ``apex_radius_mm + r_offset_mm``, angular half-width ``theta_max_deg``
    from the corneal axis -- pass the same limbus angle used to place the
    montage's own corneal-contact electrodes, so the lid exactly covers the
    palpebral aperture) centred at the eye frame's origin, oriented along
    ``frame.e3``. Only relabels voxels currently in ``overwrite`` (background/
    air by default) to ``skin_label`` -- it never touches the cornea, sclera
    or an electrode already registered there.

    There is no CT/MRI of a closed rat eyelid to measure a thickness from;
    ``thickness_mm=0.3`` matches the skin-conform thickness already used
    elsewhere in this project's clinical electrode designs (the ``pad()``
    helper in ``examples/07_clinical_analogues.py``), not a literature value
    for this specific tissue -- treat results from this variant as showing
    the *direction and rough scale* of the effect of closing the eye, not a
    validated absolute number.

    Returns the number of voxels relabelled (0 if the geometry landed
    entirely on existing tissue -- worth checking, not assuming, since an
    already-closed crop or a wrong ``theta_max_deg`` would silently no-op).
    """
    from .geometry import SphericalBand
    grid = Grid.from_model(model)
    lid = SphericalBand(radius_mm=apex_radius_mm + r_offset_mm,
                        thickness_mm=thickness_mm,
                        theta_max_deg=theta_max_deg).place(frame.origin_mm, frame.e3)
    idx, _, _ = rasterize(lid, grid)
    if len(idx) == 0:
        return 0
    here = model.labels[idx[:, 0], idx[:, 1], idx[:, 2]]
    allow = np.isin(here, list(overwrite))
    idx = idx[allow]
    if len(idx) == 0:
        return 0
    if skin_label not in model.materials:
        # match whatever resistivity the model already uses for skin, if any
        existing = [m for m in model.materials if getattr(m, "name", "") and
                   "skin" in m.name.lower()]
        rho = existing[0].rho[0] if existing else 4.0
        model.materials.add(Material.isotropic_rho(skin_label, rho, name="eyelid (closed)"))
    model.labels[idx[:, 0], idx[:, 1], idx[:, 2]] = skin_label
    return len(idx)


def dice(a: np.ndarray, b: np.ndarray) -> float:
    """Sorensen-Dice overlap of two boolean voxel masks (1.0 = identical)."""
    a = a.astype(bool); b = b.astype(bool)
    s = int(a.sum()) + int(b.sum())
    return 1.0 if s == 0 else float(2 * int((a & b).sum()) / s)


def compare_masks(a: np.ndarray, b: np.ndarray, name: str = "") -> Dict:
    """Geometric agreement report between a generated mask ``a`` and a
    reference mask ``b`` -- voxel counts, Dice/Jaccard, and which voxels
    disagree and in which direction."""
    a = a.astype(bool); b = b.astype(bool)
    inter = int((a & b).sum())
    union = int((a | b).sum())
    return {
        "name": name,
        "generated_voxels": int(a.sum()),
        "reference_voxels": int(b.sum()),
        "intersection": inter,
        "dice": dice(a, b),
        "jaccard": float(inter / max(union, 1)),
        "missed_reference": int((b & ~a).sum()),
        "extra_generated": int((a & ~b).sum()),
    }


def fit_sphere(P_mm: np.ndarray) -> Tuple[np.ndarray, float]:
    """Algebraic least-squares sphere fit to points (mm), returns (center, r)."""
    A = np.c_[2 * P_mm, np.ones(len(P_mm))]
    b = (P_mm ** 2).sum(1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = sol[:3]
    return c, float(np.sqrt(sol[3] + (c ** 2).sum()))


def fit_spherical_band(idx: np.ndarray, grid: Grid,
                       axis_hint: Optional[Sequence[float]] = None
                       ) -> Tuple[SphericalBand, Dict]:
    """Recover a placed :class:`~neuroam.geometry.SphericalBand` from an
    electrode's existing voxels.

    Fits a sphere to the voxel centres, then measures the band's angular
    extent about ``axis_hint`` (fix this to an anatomical axis, e.g.
    globe-centre -> cornea-centre, for a ring; without it the axis is taken
    from the band centroid, which is only meaningful for a cap). This is how
    a hand-painted contact-lens electrode becomes a reusable, resolution- and
    species-independent :class:`ElectrodeSpec` region, and it composes with
    :func:`refine_by_dice` to tighten the fit against the original voxels.

    Returns ``(region, info)``; ``info`` carries the raw fitted numbers
    (center_mm, r_in_mm, r_out_mm, axis, theta_min_deg, theta_max_deg) for
    inspection, JSON export, or as a starting point for ``refine_by_dice``.
    """
    P = grid.centers_mm(idx)
    c, r_fit = fit_sphere(P)
    rad = np.linalg.norm(P - c, axis=1)
    u = np.asarray(axis_hint, float) if axis_hint is not None else (P.mean(0) - c)
    u = u / max(np.linalg.norm(u), _EPS)
    cos = np.clip(((P - c) @ u) / np.maximum(rad, _EPS), -1.0, 1.0)
    ang = np.degrees(np.arccos(cos))
    r_in, r_out = float(rad.min()), float(rad.max())
    theta_min, theta_max = float(ang.min()), float(ang.max())
    region = SphericalBand(0.5 * (r_in + r_out), r_out - r_in,
                           theta_max, theta_min).place(c, u)
    info = {"center_mm": c.tolist(), "r_in_mm": r_in, "r_out_mm": r_out,
            "axis": u.tolist(), "theta_min_deg": theta_min,
            "theta_max_deg": theta_max, "fit_radius_mm": r_fit,
            "n_voxels": int(len(idx))}
    return region, info


def fit_tube_path(idx: np.ndarray, grid: Grid,
                  simplify_tol_mm: Optional[float] = None,
                  max_nodes: int = 40, bin_width_mm: Optional[float] = None
                  ) -> Tuple[Polyline, Dict]:
    """Recover a centre-line :class:`~neuroam.geometry.Polyline` and radius
    from a tube-like electrode's voxels -- a needle, wire, or hooked "J" lead.

    Builds a 26-connectivity graph over the voxel centres (bridging any
    disconnected pieces), finds the two geodesic extremes, then takes the
    centroid of each geodesic-distance shell as a centre-line node. This
    tracks a curved lead correctly -- no single cylinder or straight-line fit
    would -- and needs no skeletonization. ``simplify_tol_mm`` and
    ``bin_width_mm`` default to fractions of the model's voxel size.
    """
    from scipy import sparse
    from scipy.sparse import csgraph
    from scipy.spatial import cKDTree

    dx = grid.dx_mm
    simplify_tol_mm = simplify_tol_mm if simplify_tol_mm is not None else 0.8 * dx
    bin_width_mm = bin_width_mm if bin_width_mm is not None else 2.0 * dx

    P = grid.centers_mm(idx)                 # voxel centres, world mm

    # 26-connectivity graph, weighted by euclidean distance
    tree = cKDTree(P)
    pairs = np.array(sorted(tree.query_pairs(r=math.sqrt(3) * dx + 1e-6 * dx)))
    if len(pairs) == 0:
        raise ValueError("cannot build a path: skeleton has no adjacent points")
    w = np.linalg.norm(P[pairs[:, 0]] - P[pairs[:, 1]], axis=1)
    n = len(P)
    A = sparse.coo_matrix((np.r_[w, w], (np.r_[pairs[:, 0], pairs[:, 1]],
                                         np.r_[pairs[:, 1], pairs[:, 0]])),
                          shape=(n, n)).tocsr()

    # skeletons of curved tubes often fragment; bridge components along their
    # closest points so the walk can traverse the whole lead
    ncomp, lab = csgraph.connected_components(A, directed=False)
    if ncomp > 1:
        extra_r, extra_c, extra_w = [], [], []
        groups = [np.nonzero(lab == k)[0] for k in range(ncomp)]
        trees = [cKDTree(P[g]) for g in groups]
        for i in range(ncomp):
            for j in range(i + 1, ncomp):
                d, sub = trees[i].query(P[groups[j]])
                k = int(np.argmin(d))
                a_, b_ = int(groups[i][sub[k]]), int(groups[j][k])
                extra_r += [a_, b_]; extra_c += [b_, a_]
                extra_w += [float(d[k])] * 2
        A = (A + sparse.coo_matrix((extra_w, (extra_r, extra_c)),
                                   shape=(n, n)).tocsr())

    def farthest(src: int):
        d = csgraph.dijkstra(A, directed=False, indices=src)
        finite = np.isfinite(d)
        return int(np.argmax(np.where(finite, d, -1))), d

    # the two geodesic extremes of the lead
    a, _ = farthest(0)
    b, d_from_a = farthest(a)

    # Centre-line = centroid of each geodesic-distance shell. This tracks a
    # curved lead correctly and needs no skeletonization.
    dv = np.where(np.isfinite(d_from_a), d_from_a, np.nan)
    span = float(np.nanmax(dv))
    nbins = max(2, int(np.ceil(span / max(bin_width_mm, _EPS))) + 1)
    edges = np.linspace(0, span + 1e-9, nbins + 1)
    which = np.digitize(dv, edges) - 1
    nodes = []
    for k in range(nbins):
        sel = which == k
        if sel.any():
            nodes.append(P[sel].mean(0))
    path = np.asarray(nodes)
    if len(path) < 2:
        path = np.vstack([P[a], P[b]])
    # bin centroids stop about half a bin short of each tip; anchor the ends
    # on the true geodesic extremes so the lead keeps its full length
    if np.linalg.norm(path[0] - P[a]) > 1e-9:
        path = np.vstack([P[a], path])
    if np.linalg.norm(path[-1] - P[b]) > 1e-9:
        path = np.vstack([path, P[b]])

    path = _rdp(path, simplify_tol_mm)
    if len(path) > max_nodes:
        keep = np.linspace(0, len(path) - 1, max_nodes).astype(int)
        path = path[keep]

    # radius from cross-sectional area: n_vox * dx^3 (mm^3) ~= pi r^2 * length
    length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
    radius = float(np.sqrt(max(len(idx), 1) * dx ** 3 / (np.pi * max(length, _EPS))))
    region = Polyline(path, radius)
    info = {"path_mm": path.tolist(), "radius_mm": radius,
            "length_mm": length, "n_voxels": int(len(idx))}
    return region, info


def _rdp(points: np.ndarray, tol: float) -> np.ndarray:
    """Ramer-Douglas-Peucker polyline simplification."""
    if len(points) < 3:
        return points
    a, b = points[0], points[-1]
    ab = b - a
    L = np.linalg.norm(ab)
    if L < _EPS:
        d = np.linalg.norm(points - a, axis=1)
    else:
        t = np.clip(((points - a) @ ab) / (L * L), 0, 1)
        proj = a + t[:, None] * ab
        d = np.linalg.norm(points - proj, axis=1)
    i = int(np.argmax(d))
    if d[i] <= tol:
        return np.vstack([a, b])
    return np.vstack([_rdp(points[:i + 1], tol)[:-1], _rdp(points[i:], tol)])


def refine_by_dice(reference_mask: np.ndarray, build,
                   params: Dict, sweep: Dict[str, Sequence[float]], grid: Grid,
                   supersample: int = 2, lattice: int = 1, passes: int = 2
                   ) -> Tuple[Dict, float]:
    """Coordinate-descent refinement of region parameters to maximise Dice
    against a reference voxel mask -- e.g. tightening a :func:`fit_spherical_band`
    or :func:`fit_tube_path` result until it reproduces a hand-built electrode
    voxel-for-voxel.

    ``build(params) -> Region`` constructs the trial region from a parameter
    dict; ``sweep`` maps a parameter name to a sequence of offsets tried each
    pass (added to the current value, scalar or array-valued). Returns the
    best parameters found and the Dice they achieve.
    """
    shape = reference_mask.shape

    def _mask_of(p: Dict) -> np.ndarray:
        idx, _, _ = rasterize(build(p), grid, supersample=supersample,
                              lattice=lattice)
        m = np.zeros(shape, dtype=bool)
        if len(idx):
            keep = np.all((idx >= 0) & (idx < np.asarray(shape)), axis=1)
            idx = idx[keep]
            m[idx[:, 0], idx[:, 1], idx[:, 2]] = True
        return m

    best = dict(params)
    best_d = dice(reference_mask, _mask_of(best))
    for _ in range(max(1, passes)):
        improved = False
        for key, deltas in sweep.items():
            if key not in best:
                continue
            base = best[key]
            base_arr = np.asarray(base, float)
            for d in deltas:
                trial = dict(best)
                bumped = base_arr + np.asarray(d, float)
                trial[key] = bumped.tolist() if base_arr.ndim else float(bumped)
                try:
                    sc = dice(reference_mask, _mask_of(trial))
                except Exception:
                    continue
                if sc > best_d + 1e-12:
                    best_d, best, improved = sc, trial, True
        if not improved:
            break
    return best, best_d
