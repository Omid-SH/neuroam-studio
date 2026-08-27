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
