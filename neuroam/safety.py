"""Electrode-surface stimulation safety: the Shannon charge-density
criterion, plus a small set of named literature reference points to judge a
result against.

Two genuinely different quantities show up in this project and are easy to
conflate:

- **Electrode-surface dose** (this module): how much charge/current is
  crammed through the electrode's own contact area -- what causes
  electrochemical tissue damage at the interface. A function of the
  electrode's geometry (:class:`neuroam.electrodes.ElectrodeQC`), not of the
  anatomy it is sitting on.
- **Delivered target dose** (``neuroam.fields``' ROI metrics, and the
  ``docs/STIMULATION_PARAMETER_STUDY.md`` sweep): the field that actually
  reaches the retina/brain once current has spread through tissue -- a
  function of the whole solved field, orders of magnitude smaller than the
  electrode-surface density because most tissue is far more resistive than
  the electrode metal.

A montage can fail the first (unsafe electrode) while being far too weak on
the second (no therapeutic effect) -- they are not the same axis, and this
module only ever answers the first question.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from .electrodes import ElectrodeQC

__all__ = ["ShannonLimit", "SafetyReport", "shannon_k", "charge_per_phase_C",
           "safety_report", "LITERATURE_CURRENT_DENSITY_REFERENCE"]


@dataclass(frozen=True)
class ShannonLimit:
    """A named point on Shannon's (1992) charge-density safety curve.

    ``k = log10(Q^2 / A)``, ``Q`` in microcoulombs per phase, ``A`` in cm^2.
    Higher ``k`` is more aggressive (more charge for less area). Originally
    fit to cortical/peripheral-nerve damage thresholds in animal studies; it
    is a screening heuristic, not a guarantee, and says nothing about
    frequency, duty cycle, or chronic (repeated-session) exposure.
    """

    name: str
    k: float
    note: str


#: k <= 1.5 is Shannon's own conservative "below this, no damage observed"
#: line. k <= 1.75 is looser but is the line most contemporary DBS devices
#: are designed against (~30 uC/cm^2 at a 1 mm^2-scale contact). Neither
#: number is specific to ocular/periocular stimulation -- there is no
#: published Shannon-style curve for the cornea, sclera or eyelid -- so both
#: are carried as named reference lines, not a single hardcoded pass/fail.
SHANNON_CONSERVATIVE = ShannonLimit(
    "Shannon 1992 conservative", 1.5,
    "k<=1.5: Shannon's own 'no observed damage' line (animal cortical/nerve data)")
SHANNON_DBS = ShannonLimit(
    "Shannon 1992 / DBS convention", 1.75,
    "k<=1.75 (~30 uC/cm^2 at mm^2 scale): the looser line most contemporary "
    "DBS devices are designed against")

#: Electrode-surface *current* density actually used by real periocular/
#: transcorneal devices in the literature this project models -- for sanity-
#: checking a new montage's own current density against, not a hard limit
#: (these are continuous-current or per-pulse figures at the specific pulse
#: parameters each device uses; see each source, not a universal rule).
#: Values in A/m^2 (1 A/m^2 = 0.1 mA/cm^2).
LITERATURE_CURRENT_DENSITY_REFERENCE: Dict[str, tuple] = {
    "TES-GPS OkuEl M (device warning threshold)": (
        200.0, "Instructions for use warn current density at the active "
              "electrode 'can exceed 2 mA/cm^2' -- a stated caution line, "
              "not a device output figure -- docs/CLINICAL_ELECTRODES.md"),
    "Eyetronic Nextwave / rACS (clinical use)": (
        100.0, "~1.00 mA/cm^2 mean current density at the felt-buffer "
              "electrode (0.35 cm^2, ~352 uA), in routine clinical use "
              "(Eyetronic Nextwave System, retinofugal ACS)"),
    "VIRON target retinal field, standard montage": (
        0.09, "0.09 +/- 0.01 A/m^2 -- this is the *delivered target* field "
             "at the retina, not an electrode-surface density; listed here "
             "only to make the >1000x gap between source and target vivid"),
    "VIRON target retinal field, optimised montage": (
        0.18, "0.18 +/- 0.02 A/m^2 -- same caveat as above"),
}


def charge_per_phase_C(amp_A: float, pulse_width_s: float) -> float:
    """Charge delivered in one phase of a rectangular pulse, in coulombs."""
    return abs(amp_A) * pulse_width_s


def shannon_k(charge_uC: float, area_cm2: float) -> float:
    """Shannon's k = log10(Q^2 / A), Q in uC, A in cm^2.

    Returns -inf if either charge or area is non-positive (degenerate --
    an electrode with no contact area cannot be scored, not "infinitely
    safe").
    """
    import math
    if charge_uC <= 0 or area_cm2 <= 0:
        return float("-inf")
    return math.log10(charge_uC ** 2 / area_cm2)


@dataclass
class SafetyReport:
    """Electrode-surface safety verdict for one electrode at one waveform."""

    name: str
    amp_A: float
    pulse_width_s: float
    surface_area_mm2: float
    charge_per_phase_uC: float
    charge_density_uC_cm2: float
    current_density_A_m2: float
    k: float
    limit: ShannonLimit
    ok: bool
    margin_db: float
    """20*log10(limit_charge / actual_charge) at this area -- positive means
    headroom, negative means over the line, in a unit that reads sensibly
    across orders of magnitude."""

    def __bool__(self) -> bool:
        return self.ok

    def summary(self) -> str:
        verdict = "OK" if self.ok else "OVER LIMIT"
        return (f"{self.name}: {self.amp_A * 1e6:.4g} uA x {self.pulse_width_s * 1e6:.4g} us "
               f"over {self.surface_area_mm2:.4g} mm^2 -> "
               f"{self.charge_density_uC_cm2:.4g} uC/cm^2, "
               f"{self.current_density_A_m2:.4g} A/m^2, k={self.k:.3f} "
               f"vs. {self.limit.name} (k<={self.limit.k}): {verdict} "
               f"({self.margin_db:+.1f} dB margin)")


def safety_report(qc: ElectrodeQC, amp_A: float, pulse_width_s: float,
                  limit: ShannonLimit = SHANNON_CONSERVATIVE) -> SafetyReport:
    """Score one registered electrode's own drive against a Shannon limit.

    Takes an already-built :class:`~neuroam.electrodes.ElectrodeQC` (so the
    contact area used is the electrode's *actual* exposed surface at this
    voxel size and placement, not a nominal geometric area) and the
    waveform's amplitude/phase-width. Reuses ``qc.charge_density`` /
    ``qc.current_density`` rather than recomputing area handling.

    Example::

        overlay, qcs = register(model, specs)
        rep = safety_report(qcs[0], amp_A=1e-3, pulse_width_s=5e-3)
        if not rep.ok:
            print(rep.summary())
    """
    import math
    charge_C = charge_per_phase_C(amp_A, pulse_width_s)
    charge_uC = charge_C * 1e6
    charge_density = qc.charge_density(amp_A, pulse_width_s)     # uC/cm^2
    current_density = qc.current_density(amp_A)                  # A/m^2
    area_cm2 = qc.surface_area_mm2 * 1e-2
    k = shannon_k(charge_uC, area_cm2)
    ok = k <= limit.k if math.isfinite(k) else False
    if math.isfinite(k) and area_cm2 > 0:
        limit_charge_uC = math.sqrt((10 ** limit.k) * area_cm2)
        margin_db = 20 * math.log10(limit_charge_uC / charge_uC) if charge_uC > 0 else float("inf")
    else:
        margin_db = float("-inf")
    return SafetyReport(
        name=qc.name, amp_A=amp_A, pulse_width_s=pulse_width_s,
        surface_area_mm2=qc.surface_area_mm2, charge_per_phase_uC=charge_uC,
        charge_density_uC_cm2=charge_density, current_density_A_m2=current_density,
        k=k, limit=limit, ok=ok, margin_db=margin_db)
