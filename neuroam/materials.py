"""Material definitions and libraries.

Legacy convention (verified against the Cela 3D Impedance Method reference,
Appendix A, and real ``.in`` files): each material line stores per-axis
**resistivity** in ohm-meter, with an imaginary/capacitive part per axis::

    material <id> rho_x im_x rho_y im_y rho_z im_z

Insulators are ~1e7 ohm*m, metal electrodes ~1e-7 ohm*m.
NeuroAM keeps resistivity as the stored quantity and exposes conductivity
(S/m) as a derived property.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

INSULATOR_RHO = 1e7
METAL_RHO = 1e-7


@dataclass
class Material:
    """One material: per-axis resistivity (ohm*m) plus optional metadata."""

    id: int
    rho: Tuple[float, float, float]           # (rho_x, rho_y, rho_z), ohm*m
    rho_im: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    name: str = ""
    reference: str = ""                        # literature source for the value
    species: str = ""
    frequency_hz: Optional[float] = None
    notes: str = ""

    # ---- derived ----
    @property
    def sigma(self) -> Tuple[float, float, float]:
        """Per-axis conductivity in S/m."""
        return tuple(1.0 / r for r in self.rho)  # type: ignore[return-value]

    @property
    def isotropic(self) -> bool:
        return self.rho[0] == self.rho[1] == self.rho[2]

    @property
    def is_resistive(self) -> bool:
        return all(v == 0.0 for v in self.rho_im)

    @classmethod
    def from_sigma(cls, id: int, sigma, **kw) -> "Material":
        """Create from conductivity (S/m); scalar or per-axis triple."""
        if isinstance(sigma, (int, float)):
            sigma = (float(sigma),) * 3
        rho = tuple(1.0 / s for s in sigma)
        return cls(id=id, rho=rho, **kw)

    @classmethod
    def isotropic_rho(cls, id: int, rho: float, **kw) -> "Material":
        return cls(id=id, rho=(rho, rho, rho), **kw)


class MaterialLibrary:
    """A named collection of materials, JSON-serializable and .in-compatible."""

    def __init__(self, materials: Optional[Iterable[Material]] = None,
                 default_material: Optional[int] = None):
        self._by_id: Dict[int, Material] = {}
        self.default_material = default_material
        for m in materials or ():
            self.add(m)

    # ---- container behavior ----
    def add(self, m: Material) -> None:
        self._by_id[m.id] = m

    def __getitem__(self, mid: int) -> Material:
        try:
            return self._by_id[mid]
        except KeyError:
            if self.default_material is not None and self.default_material in self._by_id:
                return self._by_id[self.default_material]
            raise KeyError(
                f"material {mid} not defined and no default_material set"
            ) from None

    def __contains__(self, mid: int) -> bool:
        return mid in self._by_id

    def __iter__(self):
        return iter(sorted(self._by_id.values(), key=lambda m: m.id))

    def __len__(self) -> int:
        return len(self._by_id)

    def ids(self):
        return sorted(self._by_id)

    # ---- legacy .in I/O ----
    _IN_MAT = re.compile(r"^material\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)")

    @classmethod
    def from_in_file(cls, path) -> "MaterialLibrary":
        lib = cls()
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            m = cls._IN_MAT.match(line)
            if m:
                mid = int(m.group(1))
                vals = [float(m.group(i)) for i in range(2, 8)]
                lib.add(Material(id=mid,
                                 rho=(vals[0], vals[2], vals[4]),
                                 rho_im=(vals[1], vals[3], vals[5])))
            elif line.startswith("defaultmaterial"):
                lib.default_material = int(line.split()[1])
        return lib

    def to_in_lines(self) -> list:
        lines = []
        for m in self:
            r, i = m.rho, m.rho_im
            lines.append(
                f"material {m.id:04d} {r[0]:g} {i[0]:g} {r[1]:g} {i[1]:g} {r[2]:g} {i[2]:g}"
            )
        if self.default_material is not None:
            lines.append(f"defaultmaterial {self.default_material}")
        return lines

    # ---- JSON I/O ----
    def to_json(self, path=None) -> str:
        payload = {
            "schema": "neuroam-materials-1",
            "default_material": self.default_material,
            "materials": [asdict(m) for m in self],
        }
        text = json.dumps(payload, indent=2)
        if path is not None:
            Path(path).write_text(text)
        return text

    @classmethod
    def from_json(cls, src) -> "MaterialLibrary":
        if isinstance(src, (str, Path)) and Path(src).exists():
            data = json.loads(Path(src).read_text())
        else:
            data = json.loads(src)
        lib = cls(default_material=data.get("default_material"))
        for d in data["materials"]:
            d = dict(d)
            d["rho"] = tuple(d["rho"])
            d["rho_im"] = tuple(d.get("rho_im", (0.0, 0.0, 0.0)))
            lib.add(Material(**d))
        return lib

    # ---- vectorized lookup used by assembly ----
    def rho_arrays(self, ids):
        """Per-axis resistivity arrays for an integer label array.

        Returns (rho_x, rho_y, rho_z) float arrays shaped like ``ids``.
        Unknown labels fall back to default_material if set, else raise.
        """
        import numpy as np

        ids = np.asarray(ids)
        uniq = np.unique(ids)
        lut_size = int(uniq.max()) + 1
        lut = np.full((lut_size, 3), np.nan)
        for u in uniq:
            mat = self[int(u)]
            lut[int(u)] = mat.rho
        out = lut[ids]
        if np.isnan(out).any():
            raise KeyError("undefined material id in label array")
        return out[..., 0], out[..., 1], out[..., 2]


def default_electrode_materials(lib: MaterialLibrary,
                                source_id: int = 101,
                                ground_id: int = 100) -> MaterialLibrary:
    """Ensure the conventional electrode labels exist (metal, 1e-7 ohm*m)."""
    for mid, nm in ((source_id, "source electrode"), (ground_id, "ground electrode")):
        if mid not in lib:
            lib.add(Material.isotropic_rho(mid, METAL_RHO, name=nm))
    return lib
