"""Stimulus waveforms and legacy ``.cur`` I/O.

``.cur`` format (verified): first line ``% <dt_seconds> <x>``, then one
current amplitude (A) per line, one per time step.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


@dataclass
class Waveform:
    """A sampled current waveform: amplitudes (A) on a fixed time step (s)."""

    dt: float
    samples: np.ndarray

    @property
    def t(self) -> np.ndarray:
        return np.arange(len(self.samples)) * self.dt

    @property
    def duration(self) -> float:
        return len(self.samples) * self.dt

    def charge_balance(self) -> float:
        """Net charge divided by total absolute charge (0 = balanced)."""
        q = np.sum(self.samples)
        qa = np.sum(np.abs(self.samples))
        return float(q / qa) if qa > 0 else 0.0

    # ------------------------------------------------------------- legacy IO
    @classmethod
    def from_cur(cls, path) -> "Waveform":
        dt = None
        vals = []
        for line in Path(path).read_text().splitlines():
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "%":
                dt = float(parts[1])
            else:
                vals.append(float(parts[0]))
        if dt is None:
            raise ValueError(f"{path}: missing '%' timestep header")
        return cls(dt=dt, samples=np.asarray(vals))

    def to_cur(self, path, extra: float = 0.0) -> Path:
        path = Path(path)
        lines = [f"% {self.dt:g} {extra:g}"]
        lines += [f"{v:g}" for v in self.samples]
        path.write_text("\n".join(lines) + "\n")
        return path


# ----------------------------------------------------------------- factories
def dc(amp_A: float, dt: float, n_steps: int = 1) -> Waveform:
    return Waveform(dt=dt, samples=np.full(n_steps, amp_A))


def monophasic_pulse_train(amp_A: float, pulse_width_s: float,
                           period_s: float, n_pulses: int, dt: float,
                           delay_s: float = 0.0) -> Waveform:
    n = int(round((delay_s + n_pulses * period_s) / dt))
    s = np.zeros(n)
    pw = max(1, int(round(pulse_width_s / dt)))
    for k in range(n_pulses):
        i0 = int(round((delay_s + k * period_s) / dt))
        s[i0:i0 + pw] = amp_A
    return Waveform(dt=dt, samples=s)


def biphasic_pulse_train(amp_A: float, pulse_width_s: float, period_s: float,
                         n_pulses: int, dt: float, ratio: float = 1.0,
                         interphase_s: float = 0.0, delay_s: float = 0.0,
                         cathodic_first: bool = True) -> Waveform:
    """Charge-balanced biphasic train.

    ``ratio`` is the asymmetric charge-balance ratio: the second phase has
    amplitude ``amp/ratio`` and width ``pulse_width*ratio`` (SCB: ratio=1;
    ACB 1:4 -> ratio=4), keeping phase charges equal.
    """
    pw1 = max(1, int(round(pulse_width_s / dt)))
    pw2 = max(1, int(round(pulse_width_s * ratio / dt)))
    ip = int(round(interphase_s / dt))
    a1 = -amp_A if cathodic_first else amp_A
    a2 = -a1 / ratio
    n = int(round((delay_s + n_pulses * period_s) / dt))
    s = np.zeros(n)
    for k in range(n_pulses):
        i0 = int(round((delay_s + k * period_s) / dt))
        s[i0:i0 + pw1] = a1
        s[i0 + pw1 + ip:i0 + pw1 + ip + pw2] = a2
    return Waveform(dt=dt, samples=s)


def sine(amp_A: float, freq_hz: float, n_cycles: float, dt: float,
         delay_s: float = 0.0) -> Waveform:
    n = int(round((delay_s + n_cycles / freq_hz) / dt))
    t = np.arange(n) * dt
    s = np.where(t >= delay_s,
                 amp_A * np.sin(2 * np.pi * freq_hz * (t - delay_s)), 0.0)
    return Waveform(dt=dt, samples=s)


def from_config(cfg: dict) -> Waveform:
    """Build a waveform from a JSON config dict (see pipeline schema)."""
    kind = cfg.get("type", "dc")
    dt = float(cfg.get("dt_s", 1e-4))
    if kind == "dc":
        return dc(float(cfg["amp_A"]), dt, int(cfg.get("n_steps", 1)))
    if kind == "monophasic":
        return monophasic_pulse_train(
            float(cfg["amp_A"]), float(cfg["pulse_width_s"]),
            float(cfg["period_s"]), int(cfg.get("n_pulses", 1)), dt,
            float(cfg.get("delay_s", 0.0)))
    if kind == "biphasic":
        return biphasic_pulse_train(
            float(cfg["amp_A"]), float(cfg["pulse_width_s"]),
            float(cfg["period_s"]), int(cfg.get("n_pulses", 1)), dt,
            float(cfg.get("ratio", 1.0)), float(cfg.get("interphase_s", 0.0)),
            float(cfg.get("delay_s", 0.0)),
            bool(cfg.get("cathodic_first", True)))
    if kind == "sine":
        return sine(float(cfg["amp_A"]), float(cfg["freq_hz"]),
                    float(cfg.get("n_cycles", 1)), dt,
                    float(cfg.get("delay_s", 0.0)))
    if kind == "cur_file":
        return Waveform.from_cur(cfg["path"])
    raise ValueError(f"unknown waveform type {kind!r}")
