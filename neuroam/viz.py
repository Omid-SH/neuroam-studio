"""Headless-safe matplotlib visualization (PNG outputs)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _slice(arr: np.ndarray, axis: str, index: int) -> np.ndarray:
    a = "xyz".index(axis)
    sl = [slice(None)] * 3
    sl[a] = index
    plane = arr[tuple(sl)]
    return plane.T  # display: first remaining axis horizontal


def save_slice(path, arr: np.ndarray, axis: str = "z", index: Optional[int] = None,
               title: str = "", cmap: str = "viridis", log: bool = False,
               units: str = "") -> Path:
    if index is None:
        index = arr.shape["xyz".index(axis)] // 2
    plane = _slice(arr, axis, index)
    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=130)
    data = np.log10(np.abs(plane) + 1e-30) if log else plane
    im = ax.imshow(data, origin="lower", cmap=cmap, interpolation="nearest")
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label(("log10 " if log else "") + units)
    rem = [c for c in "xyz" if c != axis]
    ax.set_xlabel(rem[0]); ax.set_ylabel(rem[1])
    ax.set_title(title or f"{axis}={index}")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return Path(path)


def save_labels_slice(path, labels: np.ndarray, axis: str = "z",
                      index: Optional[int] = None, title: str = "") -> Path:
    return save_slice(path, labels.astype(float), axis, index,
                      title or "materials", cmap="tab20", units="material id")


def save_waveform(path, waveform, title: str = "stimulus") -> Path:
    fig, ax = plt.subplots(figsize=(7, 3), dpi=130)
    ax.step(waveform.t * 1e3, waveform.samples * 1e3, where="post")
    ax.set_xlabel("time (ms)"); ax.set_ylabel("current (mA)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return Path(path)


def save_vm_traces(path, t_ms: np.ndarray, vm_mV: np.ndarray,
                   labels: Optional[Sequence[str]] = None,
                   title: str = "membrane potential", max_traces: int = 8) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4), dpi=130)
    n = vm_mV.shape[1]
    idx = np.linspace(0, n - 1, min(n, max_traces)).astype(int)
    for j in idx:
        lbl = labels[j] if labels else f"seg {j}"
        ax.plot(t_ms, vm_mV[:, j], lw=0.9, label=lbl)
    ax.set_xlabel("time (ms)"); ax.set_ylabel("Vm (mV)")
    ax.set_title(title)
    if len(idx) <= 8:
        ax.legend(fontsize=7, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return Path(path)


def save_vm_comparison(path, traces: dict, title: str = "membrane potential",
                       xlabel: str = "time (ms)", ylabel: str = "Vm (mV)",
                       ax=None):
    """Overlay several *named, single-segment* Vm(t) traces on one axes --
    e.g. the same soma under different drives (single pulse vs. pulse
    train vs. a zero-drive control), rather than ``save_vm_traces``'s many
    segments of *one* run. ``traces`` maps a legend label to a
    ``(t_ms, vm_mV)`` pair (each 1-D). Pass ``ax`` to draw into an existing
    subplot (e.g. building a composite figure) instead of saving standalone
    -- ``path`` is then ignored and the caller owns saving the figure."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 4), dpi=130)
    for label, (t, v) in traces.items():
        ax.plot(t, v, lw=1.1, label=label)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    if standalone:
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return Path(path)
    return ax


def save_dose_response(path, series: dict, title: str = "dose-response",
                       xlabel: str = "amplitude (uA)", ylabel: str = "peak deviation (mV)",
                       xlog: bool = True, ylog: bool = False,
                       spike_markers: Optional[dict] = None, ax=None):
    """A log-x amplitude sweep with one or more named series.

    ``series`` maps a legend label to an ``(x, y)`` pair (matched
    amplitude and response arrays, e.g. one condition's dose-response
    curve). ``spike_markers``, if given, maps the same labels to an ``x``
    value to mark with a star (e.g. the lowest amplitude that produced a
    spike) -- drawn on top of that series' own line/color. Pass ``ax`` to
    draw into an existing subplot instead of saving standalone."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6, 4.5), dpi=130)
    for label, (x, y) in series.items():
        line, = ax.plot(x, y, marker="o", ms=4, lw=1.3, label=label)
        if spike_markers and label in spike_markers:
            xm = spike_markers[label]
            ym = np.interp(xm, x, y) if xm <= max(x) else max(y)
            ax.plot([xm], [ym], marker="*", ms=14, mec="black",
                   mfc=line.get_color(), zorder=5)
    if xlog:
        ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3, which="both")
    if standalone:
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return Path(path)
    return ax
