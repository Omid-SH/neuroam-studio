"""Shared helpers for the stimulation-parameter sweep (examples/31_*.py).

Not a numbered example itself -- imported by the sweep runners. Reuses the
already-solved, already-cached fields from examples/28 (same cache-key
fingerprint) so none of this re-solves anything; a full sweep of hundreds
of trials costs NEURON time only, no AM solves.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from neuroam.cache import ResultCache, hash_inputs
from neuroam.electrodes import ElectrodeSpec, register
from neuroam.frames import Frame
from neuroam.model import VoxelModel

REPO = Path(__file__).resolve().parent.parent
SAMP = REPO / "samples/ratcc_eye_83um"
RETINA = 77

CONFIGS = {
    "SCL-ON": "RatCC_full_CLStim_JGND_912_900_504_Ring_Res_83um_with_retina",
    "SCL-IntraCranial": "RatCC_full_CLStim_NeedleGND_912_900_504_Ring_Res_83um_with_retina",
    "SCL-TransCranial": "RatCC_full_CLStim_PatchGND_912_900_504_Ring_Res_83um_with_retina",
    "ON-IntraCranial": "RatCC_full_JStim_NeedleGND_912_900_504_Ring_Res_83um_with_retina",
    "ON-TransCranial": "RatCC_full_JStim_PatchGND_912_900_504_Ring_Res_83um_with_retina",
    "IntraCranial-TransCranial": "RatCC_full_NeedleStim_PlateGND_912_900_504_Res_83um_with_retina",
}


class CachedMontage:
    """One config's already-solved field, loaded once and reused for many
    trials at different locations -- cropping is cheap array slicing."""

    def __init__(self, cfg: str):
        fname = CONFIGS[cfg]
        in_path = SAMP / f"{fname}.in"
        mrm_path = SAMP / f"{fname}.mrm"
        self.cfg = cfg
        self.model = VoxelModel.from_legacy(in_path)
        register(self.model, [
            ElectrodeSpec("stim", from_label=101, role="source", label=101,
                          terminal="supernode", waveform="Cur1"),
            ElectrodeSpec("gnd", from_label=102, role="ground", label=102,
                          terminal="supernode"),
        ])

        def _fingerprint(p: Path) -> tuple:
            st = p.stat()
            return (str(p), st.st_size, int(st.st_mtime))

        cache = ResultCache(REPO / ".neuroam_cache")
        key = hash_inputs(_fingerprint(in_path), _fingerprint(mrm_path),
                          "stim=101,gnd=102,supernode,Cur1", "rtol=1e-7,amg")
        full = cache.get_array("scl_on_field", key)
        if full is None:
            raise RuntimeError(
                f"{cfg} has no cached field -- run "
                f"examples/28_solve_scl_on_for_registration.py {cfg!r} first")
        self.grid = full["grid"].astype(np.float64)
        self.jmag = full["jmag"].astype(np.float64)
        self.dx = float(self.model.dx)
        self.world = np.asarray(self.model.world)

        self.fr = Frame.eye(self.model, (RETINA,))
        retina_mask = self.model.labels == RETINA
        self.retina_idx = np.argwhere(retina_mask)
        pts_mm = self.fr.grid.centers_mm(self.retina_idx)
        self._r, self._theta, self._phi = self.fr.spherical_of(pts_mm)
        self._ecc = 180.0 - self._theta

    def voxel_near(self, ecc_deg: float, phi_deg: float) -> np.ndarray:
        """The real retina voxel closest (angularly) to a target
        (eccentricity, azimuth) -- never an arbitrary point in empty space."""
        dphi = np.abs(((self._phi - phi_deg + 180) % 360) - 180)
        d = (self._ecc - ecc_deg) ** 2 + dphi ** 2
        return self.retina_idx[np.argmin(d)].astype(float)

    def radial_direction(self, anchor_vox: np.ndarray) -> np.ndarray:
        anchor_mm = self.fr.grid.centers_mm(anchor_vox.reshape(1, 3))[0]
        v = anchor_mm - self.fr.origin_mm
        return v / np.linalg.norm(v)

    def crop(self, anchor_vox: np.ndarray, pad_vox: float = 40.0):
        lo = np.clip(np.floor(anchor_vox - pad_vox).astype(int), 0, None)
        hi_vox = np.minimum(np.ceil(anchor_vox + pad_vox).astype(int) + 1, self.world)
        hi_grid = np.minimum(hi_vox + 1, np.asarray(self.grid.shape))
        grid_crop = self.grid[lo[0]:hi_grid[0], lo[1]:hi_grid[1], lo[2]:hi_grid[2]]
        jmag_crop = self.jmag[lo[0]:hi_vox[0], lo[1]:hi_vox[1], lo[2]:hi_vox[2]]
        return grid_crop, jmag_crop, lo


def rotate_about_tangent(radial: np.ndarray, deviation_deg: float) -> np.ndarray:
    """A unit vector ``deviation_deg`` off ``radial`` -- for the orientation
    sweep: 0 = anatomically correct (perpendicular to the retina), 180 =
    fully inverted. Rotates about a fixed tangent direction (arbitrary but
    consistent), not a random axis, so a sweep of deviations traces one
    great circle through the true radial direction."""
    radial = radial / np.linalg.norm(radial)
    ref = np.array([1.0, 0.0, 0.0]) if abs(radial[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    tangent = np.cross(radial, ref)
    tangent /= np.linalg.norm(tangent)
    th = np.deg2rad(deviation_deg)
    return np.cos(th) * radial + np.sin(th) * tangent
