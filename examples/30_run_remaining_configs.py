"""Solve the other 5 of 6 OEC-RGC montages, then register both RGCs in each.

SCL-ON is already done (examples/28 + examples/29). This runs, for each
remaining config: the AM solve + cache (examples/28), then D1 and A2i
registration (examples/29, each its own process -- see
neuroam.neuron_link.load_hoc_cell's docstring for why they can't share one).

Each of these subprocess calls is independently resumable: examples/28
skips straight to a cached reload if that config was already solved (see
its own module docstring), so re-running this script after a partial
failure does not redo completed work.

Run:  python examples/30_run_remaining_configs.py
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable

REMAINING = ["SCL-IntraCranial", "SCL-TransCranial", "ON-IntraCranial",
            "ON-TransCranial", "IntraCranial-TransCranial"]
CELLS = ["d1", "a2i"]


def run(args: list) -> None:
    t0 = time.time()
    print(f">>> {' '.join(args)}", flush=True)
    r = subprocess.run([PY] + args, cwd=str(REPO))
    print(f"<<< exit {r.returncode} ({time.time()-t0:.0f}s)", flush=True)
    if r.returncode != 0:
        raise SystemExit(f"failed: {' '.join(args)} (exit {r.returncode})")


def main() -> int:
    for cfg in REMAINING:
        run(["examples/28_solve_scl_on_for_registration.py", cfg])
        for cell in CELLS:
            run(["examples/29_register_rgc_in_scl_on.py", cell, cfg])
        print(f"=== {cfg} complete ===\n", flush=True)
    print("ALL 5 REMAINING CONFIGS COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
