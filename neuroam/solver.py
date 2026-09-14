"""Sparse solvers, unit-current basis fields, and superposition."""

from __future__ import annotations

import inspect
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

from .assembly import System

# 3D Laplacians incur heavy fill-in in sparse LU; beyond ~4e4 unknowns CG with
# a diagonal preconditioner is much faster, so "auto" switches early.
_DIRECT_MAX_N = 40_000


@dataclass
class SolveResult:
    v: np.ndarray                  # reduced node voltages
    info: int
    iterations: int
    residual: float
    seconds: float
    method: str


def _iterative(name: str):
    return {"cg": spla.cg, "bicgstab": spla.bicgstab, "gmres": spla.gmres}[name]


def _tol_kwargs(fn: Callable, rtol: float):
    """SciPy renamed tol->rtol in 1.12/1.14; support both."""
    params = inspect.signature(fn).parameters
    if "rtol" in params:
        return {"rtol": rtol, "atol": 0.0}
    return {"tol": rtol, "atol": 0.0}


def make_preconditioner(G: sparse.csr_matrix, kind: str = "diag"):
    if kind in (None, "none"):
        return None
    if kind == "diag":
        d = G.diagonal()
        d[d == 0] = 1.0
        return sparse.diags(1.0 / d)
    if kind == "ilu":
        ilu = spla.spilu(G.tocsc(), drop_tol=1e-5, fill_factor=10)
        return spla.LinearOperator(G.shape, ilu.solve)
    if kind == "amg":
        try:
            import pyamg
        except ImportError as exc:
            raise ImportError(
                "precond='amg' needs the optional 'pyamg' package "
                "(pip install pyamg)") from exc
        # Smoothed aggregation on the (SPD) conductance matrix: on a 6.44M-
        # unknown model this took CG from ~2400 iterations to ~50.
        ml = pyamg.smoothed_aggregation_solver(G.tocsr())
        return ml.aspreconditioner()
    raise ValueError(f"unknown preconditioner {kind!r}")


def solve(system: System, I: np.ndarray, method: str = "auto",
          rtol: float = 1e-8, maxiter: Optional[int] = None,
          precond: str = "diag", x0: Optional[np.ndarray] = None) -> SolveResult:
    """Solve G v = I on the reduced system."""
    G = system.G
    n = G.shape[0]
    if method == "auto":
        method = "direct" if n <= _DIRECT_MAX_N else "cg"

    t0 = time.time()
    if method == "direct":
        v = spla.spsolve(G.tocsc(), I)
        res = float(np.linalg.norm(G @ v - I) / max(np.linalg.norm(I), 1e-300))
        return SolveResult(v=v, info=0, iterations=1, residual=res,
                           seconds=time.time() - t0, method="direct")

    fn = _iterative(method)
    M = make_preconditioner(G, precond)
    iters = {"n": 0}

    def cb(xk):
        iters["n"] += 1

    kw = _tol_kwargs(fn, rtol)
    if method == "gmres":
        v, info = fn(G, I, x0=x0, M=M, maxiter=maxiter or 20000,
                     callback=cb, callback_type="pr_norm", **kw)
    else:
        v, info = fn(G, I, x0=x0, M=M, maxiter=maxiter or 200_000,
                     callback=cb, **kw)
    res = float(np.linalg.norm(G @ v - I) / max(np.linalg.norm(I), 1e-300))
    if info > 0 and res > rtol * 10:
        raise RuntimeError(
            f"{method} did not converge: info={info}, residual={res:.3e}")
    return SolveResult(v=v, info=int(info), iterations=iters["n"],
                       residual=res, seconds=time.time() - t0, method=method)


# ------------------------------------------------------------------ basis fields
def unit_current_vector(system: System, source_name: str) -> np.ndarray:
    """1 A injected at a source.

    For a ``distributed`` terminal the ampere is split across the electrode's
    nodes by their weights; for ``node`` and ``supernode`` terminals it enters
    a single row (for a supernode that row *is* the whole equipotential
    electrode).
    """
    I = np.zeros(system.n)
    w = getattr(system, "terminal_weights", {}).get(source_name)
    if w is not None:
        rows, weights = w
        np.add.at(I, rows, weights / weights.sum())
        return I
    I[system.source_rows[source_name]] = 1.0
    return I


def electrode_impedance(system: System, v: np.ndarray, source_name: str,
                        current_A: float = 1.0) -> float:
    """Access impedance of a source electrode, ohm (V_electrode / I).

    Only meaningful for an equipotential (supernode) terminal, where the
    electrode has a single well-defined potential; for a legacy single-node
    terminal this returns the potential of that one node, which includes the
    grid's local spreading resistance and is mesh-dependent.
    """
    row = system.source_rows.get(source_name)
    if row is None:
        row = system.terminal_rows[source_name]
    return float(v[row] / current_A)


def solve_basis(system: System, source_names: Optional[List[str]] = None,
                **solve_kw) -> Dict[str, SolveResult]:
    """One unit-current (1 A) solve per source electrode.

    Any waveform or multi-electrode pattern is then a superposition —
    ``V(t) = sum_k a_k(t) * V_k`` — with no further linear solves.
    """
    names = source_names or list(system.source_rows)
    out: Dict[str, SolveResult] = {}
    for nm in names:
        out[nm] = solve(system, unit_current_vector(system, nm), **solve_kw)
    return out


def superpose(basis: Dict[str, np.ndarray], amplitudes: Dict[str, np.ndarray]
              ) -> np.ndarray:
    """Combine unit fields with per-source amplitude time series.

    ``basis[name]`` is a (n,) unit-current field; ``amplitudes[name]`` is a
    (T,) amplitude array in amperes.  Returns (T, n).
    """
    names = list(basis)
    T = len(next(iter(amplitudes.values())))
    n = len(next(iter(basis.values())))
    out = np.zeros((T, n))
    for nm in names:
        a = np.asarray(amplitudes[nm])
        if len(a) != T:
            raise ValueError("amplitude series lengths differ")
        out += np.outer(a, basis[nm])
    return out


def kcl_report(system: System, v: np.ndarray, I: np.ndarray) -> Dict[str, float]:
    """Current-conservation diagnostics for a solved field."""
    r = system.G @ v - I
    inj = float(np.sum(np.abs(I)))
    return {
        "max_kcl_residual_A": float(np.max(np.abs(r))),
        "sum_kcl_residual_A": float(np.sum(np.abs(r))),
        "relative_residual": float(np.linalg.norm(r) / max(np.linalg.norm(I), 1e-300)),
        "injected_A": inj,
    }
