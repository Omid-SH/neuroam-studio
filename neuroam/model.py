"""Voxel model container and legacy ``.in``/``.model`` I/O.

Legacy layout (matches ``AM_v10.2_res_multi.py`` exactly):

- ``world nx ny nz`` in the ``.in`` file gives model size in unit voxels.
- The ``.model`` file is ASCII, headerless, laid out as ``nz`` blocks of
  ``ny`` lines with ``nx`` values per line.  The legacy loader does::

      m2d = np.loadtxt(f)              # (nz*ny, nx)
      m3d = m2d.reshape(nz, ny, nx).transpose(2, 1, 0)   # -> [x, y, z]

- Node names are ``xxxxyyyyzzzz`` (4 digits per coordinate) for the node at
  unit-lattice coordinate (x, y, z); the ground node is renamed ``0``.
- Sources come from ``spice I <cur> [node x y z] [node x y z] ...`` lines;
  ``nodename x y z 0`` declares the ground node.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .materials import MaterialLibrary, Material, METAL_RHO, INSULATOR_RHO


def _abbrev(seq, n: int = 3) -> str:
    """Short repr for node lists — an equipotential electrode has thousands."""
    seq = list(seq)
    if len(seq) <= n:
        return repr(seq)
    return f"{seq[:n]!r}... ({len(seq):,} total)"


def node_name(x: int, y: int, z: int) -> str:
    return f"{x:04d}{y:04d}{z:04d}"


def parse_node_name(name: str) -> Tuple[int, int, int]:
    return int(name[0:4]), int(name[4:8]), int(name[8:12])


@dataclass
class Source:
    """A current source between a node and the ground reference."""

    name: str                       # waveform name (legacy: .cur basename)
    node: Tuple[int, int, int]      # unit-lattice node coordinate


@dataclass
class VoxelModel:
    labels: np.ndarray                     # (nx, ny, nz) integer material ids
    dx: float                              # unit voxel size in meters
    materials: MaterialLibrary
    sources: List[Source] = field(default_factory=list)
    ground_nodes: List[Tuple[int, int, int]] = field(default_factory=list)
    name: str = "model"
    terminals: List = field(default_factory=list)
    """Optional :class:`neuroam.electrodes.Terminal` list.

    A terminal generalizes the legacy single-node source: it names the *set*
    of lattice nodes belonging to an electrode and how they connect
    (``supernode`` = merged into one equipotential unknown, ``node`` = legacy
    single injection node, ``distributed`` = weighted current split).  When
    empty, assembly uses ``sources``/``ground_nodes`` exactly as before.
    """

    # ------------------------------------------------------------------ basic
    @property
    def world(self) -> Tuple[int, int, int]:
        return tuple(self.labels.shape)  # type: ignore[return-value]

    @property
    def n_nodes(self) -> int:
        nx, ny, nz = self.world
        return (nx + 1) * (ny + 1) * (nz + 1)

    def summary(self) -> str:
        nx, ny, nz = self.world
        uniq, cnt = np.unique(self.labels, return_counts=True)
        mats = ", ".join(f"{u}({c})" for u, c in zip(uniq.tolist(), cnt.tolist()))
        return (f"{self.name}: world {nx}x{ny}x{nz} ({self.labels.size:,} voxels, "
                f"{self.n_nodes:,} nodes), dx={self.dx:g} m\n"
                f"  materials: {mats}\n"
                f"  sources: {[(s.name, s.node) for s in self.sources]}\n"
                f"  ground nodes: {_abbrev(self.ground_nodes)}"
                + (f"\n  terminals: "
                   f"{[(t.name, t.role, t.kind, len(t.nodes)) for t in self.terminals]}"
                   if self.terminals else ""))

    # ------------------------------------------------------------ constructors
    @classmethod
    def empty(cls, world: Sequence[int], dx: float,
              materials: MaterialLibrary, background: int = 0,
              name: str = "model") -> "VoxelModel":
        labels = np.full(tuple(world), background, dtype=np.int32)
        if background not in materials:
            materials.add(Material.isotropic_rho(background, INSULATOR_RHO,
                                                 name="background"))
        return cls(labels=labels, dx=dx, materials=materials, name=name)

    # ------------------------------------------------------------- primitives
    def _grids(self):
        nx, ny, nz = self.world
        return np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz),
                           indexing="ij")

    def add_box(self, material: int, corner, size) -> None:
        x0, y0, z0 = corner
        sx, sy, sz = size
        self.labels[x0:x0 + sx, y0:y0 + sy, z0:z0 + sz] = material

    def add_sphere(self, material: int, center, radius: float) -> None:
        X, Y, Z = self._grids()
        cx, cy, cz = center
        mask = (X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2 <= radius ** 2
        self.labels[mask] = material

    def add_cylinder(self, material: int, center, radius: float,
                     axis: str = "z", extent: Optional[Tuple[int, int]] = None) -> None:
        X, Y, Z = self._grids()
        cx, cy, cz = center
        ax = {"x": (Y - cy, Z - cz, X), "y": (X - cx, Z - cz, Y),
              "z": (X - cx, Y - cy, Z)}[axis]
        mask = ax[0] ** 2 + ax[1] ** 2 <= radius ** 2
        if extent is not None:
            mask &= (ax[2] >= extent[0]) & (ax[2] < extent[1])
        self.labels[mask] = material

    def add_slab(self, material: int, axis: str, start: int, thickness: int) -> None:
        sl = [slice(None)] * 3
        sl["xyz".index(axis)] = slice(start, start + thickness)
        self.labels[tuple(sl)] = material

    # -------------------------------------------------------------- electrodes
    def add_electrode(self, shape: str, role: str, material: int,
                      waveform: str = "stim", node: Optional[Sequence[int]] = None,
                      **shape_kw) -> None:
        """Paint an electrode and register its terminal node.

        ``role`` is "source" or "ground".  ``node`` is the injection node in
        unit-lattice coordinates; defaults to the corner of the electrode's
        first voxel (metal equipotentializes the electrode body, matching the
        legacy single-node injection convention).
        """
        if material not in self.materials:
            self.materials.add(Material.isotropic_rho(material, METAL_RHO,
                                                      name=f"{role} electrode"))
        before = self.labels.copy()
        if shape == "box":
            self.add_box(material, **shape_kw)
        elif shape == "sphere":
            self.add_sphere(material, **shape_kw)
        elif shape == "cylinder":
            self.add_cylinder(material, **shape_kw)
        elif shape == "slab":
            self.add_slab(material, **shape_kw)
        elif shape == "none":
            pass  # electrode body painted elsewhere / bare node
        else:
            raise ValueError(f"unknown electrode shape {shape!r}")

        if node is None:
            changed = np.argwhere((self.labels == material) & (before != material))
            if changed.size == 0:
                changed = np.argwhere(self.labels == material)
            if changed.size == 0:
                raise ValueError("electrode painted no voxels and no node given")
            node = tuple(int(v) for v in changed[0])
        node = tuple(int(v) for v in node)

        if role == "source":
            self.sources.append(Source(name=waveform, node=node))
        elif role == "ground":
            self.ground_nodes.append(node)
        else:
            raise ValueError("role must be 'source' or 'ground'")

    # ------------------------------------------------------------ legacy I/O
    @classmethod
    def from_legacy(cls, in_path, model_path=None, name: Optional[str] = None
                    ) -> "VoxelModel":
        in_path = Path(in_path)
        text = in_path.read_text()
        world = None
        dx = 1.0
        matrix_file = None
        sources: List[Source] = []
        grounds: List[Tuple[int, int, int]] = []

        node_re = re.compile(r"\[node\s+(\d+)\s+(\d+)\s+(\d+)\]")
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line[0] in "#%":
                continue
            parts = line.split()
            key = parts[0]
            if key == "world":
                world = (int(parts[1]), int(parts[2]), int(parts[3]))
            elif key == "unitvoxelsize":
                dx = float(parts[1])
            elif key == "matrix":
                matrix_file = parts[1]
            elif key == "spice" and len(parts) > 2 and parts[1] == "I":
                nodes = node_re.findall(line)
                if nodes:
                    sources.append(Source(name=parts[2],
                                          node=tuple(int(v) for v in nodes[0])))
            elif key == "nodename" and parts[-1] == "0":
                grounds.append((int(parts[1]), int(parts[2]), int(parts[3])))

        if world is None:
            raise ValueError(f"{in_path}: no 'world' declaration")
        materials = MaterialLibrary.from_in_file(in_path)

        if model_path is None:
            cand = in_path.with_suffix(".model")
            if matrix_file is not None:
                for c in (in_path.parent / matrix_file,
                          (in_path.parent / matrix_file).with_suffix(".model")):
                    if c.exists():
                        cand = c
                        break
            model_path = cand
        labels = cls.load_model_array(model_path, world)
        return cls(labels=labels, dx=dx, materials=materials, sources=sources,
                   ground_nodes=grounds, name=name or in_path.stem)

    @staticmethod
    def load_model_array(path, world: Sequence[int]) -> np.ndarray:
        nx, ny, nz = world
        flat = np.loadtxt(path)
        m3d = flat.reshape(nz, ny, nx).transpose(2, 1, 0)
        return np.ascontiguousarray(m3d).astype(np.int32)

    def save_legacy(self, out_dir, name: Optional[str] = None,
                    cur_names: Optional[Sequence[str]] = None) -> Tuple[Path, Path]:
        """Write ``<name>.in`` and ``<name>.model`` in the legacy layout."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        name = name or self.name
        nx, ny, nz = self.world

        # .model: nz blocks of ny rows with nx integers each
        m2d = self.labels.transpose(2, 1, 0).reshape(nz * ny, nx)
        model_path = out_dir / f"{name}.model"
        np.savetxt(model_path, m2d, fmt="%d")

        lines = ["#" * 50,
                 "# NeuroAM Studio generated parameter file",
                 "#" * 50,
                 f"unitvoxelsize {self.dx:g}",
                 "maximumsize 1"]
        lines += self.materials.to_in_lines()
        lines += [f"matrix {name}.model",
                  f"meshfile {name}.mrm",
                  f"networkfile {name}.net",
                  f"nodevoltagefile {name}.vof"]
        grounds = self.ground_nodes or []
        gnode = grounds[0] if grounds else None
        for i, s in enumerate(self.sources):
            cur = (cur_names[i] if cur_names else s.name)
            gtxt = f" [node {gnode[0]} {gnode[1]} {gnode[2]}]" if gnode else ""
            lines.append(f"spice I {cur} [node {s.node[0]} {s.node[1]} {s.node[2]}]{gtxt}")
            lines.append(f"nodename {s.node[0]} {s.node[1]} {s.node[2]} "
                         f"{node_name(*s.node)}")
        if gnode:
            lines.append(f"nodename {gnode[0]} {gnode[1]} {gnode[2]} 0")
        lines.append(f"world {nx} {ny} {nz}")
        in_path = out_dir / f"{name}.in"
        in_path.write_text("\n".join(lines) + "\n")
        return in_path, model_path
