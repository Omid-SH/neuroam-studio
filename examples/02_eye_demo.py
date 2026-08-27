"""End-to-end demo: simplified eye model, contact-lens vs needle return,
retinal dose metrics — a miniature of the cross-species study workflow.

Run:  python examples/02_eye_demo.py
Writes PNGs + manifest into examples/out/eye_demo/
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neuroam.pipeline import run

HERE = Path(__file__).parent
OUT = HERE / "out" / "eye_demo"

cfg = {
    "neuroam": "0.1",
    "name": "eye_demo",
    "model": {
        "world": [60, 60, 60],
        "unit_voxel_m": 2.5e-4,     # 0.25 mm voxels -> 15 mm cube
        "background": 1,            # head tissue
        "shapes": [
            # eye globe: sclera shell, vitreous interior, retina shell at back
            {"type": "sphere", "material": 3, "center": [30, 30, 22],
             "radius": 12},                                     # sclera
            {"type": "sphere", "material": 2, "center": [30, 30, 22],
             "radius": 10.5},                                   # vitreous
            {"type": "sphere", "material": 4, "center": [30, 30, 22],
             "radius": 11.5},                                   # retina shell...
            {"type": "sphere", "material": 2, "center": [30, 30, 22],
             "radius": 10.4},                                   # ...re-carve vitreous
        ],
    },
    "materials": {
        "inline": [
            {"id": 1, "sigma": 0.3, "name": "head tissue"},
            {"id": 2, "sigma": 1.5, "name": "vitreous"},
            {"id": 3, "sigma": 0.01, "name": "sclera"},
            {"id": 4, "sigma": 0.7, "name": "retina"},
        ]
    },
    "electrodes": [
        {"name": "lens", "role": "source", "material": 101, "waveform": "pulse",
         "shape": {"type": "cylinder", "center": [30, 30, 4], "radius": 6,
                   "axis": "z", "extent": [2, 4]},
         "node": [30, 30, 2]},
        {"name": "ret", "role": "ground", "material": 100,
         "shape": {"type": "box", "corner": [26, 26, 58], "size": [8, 8, 2]},
         "node": [30, 30, 60]},
    ],
    "waveforms": {
        "pulse": {"type": "biphasic", "amp_A": 1e-3, "pulse_width_s": 1e-3,
                  "period_s": 10e-3, "n_pulses": 3, "dt_s": 1e-4, "ratio": 4.0}
    },
    "solve": {"method": "cg", "rtol": 1e-8, "cache": True},
    "outputs": {
        "dir": str(OUT),
        "save": ["npz", "png", "vof", "vavg", "legacy_model"],
        "slices": [{"axis": "y", "index": 30}, {"axis": "z", "index": 22}],
        "rois": [
            {"name": "retina", "labels": [4]},
            {"name": "vitreous", "labels": [2]},
            {"name": "off_target_head", "labels": [1]},
        ],
    },
}

if __name__ == "__main__":
    cfg_path = HERE / "out" / "eye_demo.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(cfg, indent=2))
    manifest = run(cfg_path)
    print("\nretinal dose per 1 A (scale by waveform amplitude):")
    for r in manifest["roi_metrics_per_unit_A"]:
        if r.get("voxels"):
            print(f"  {r['name']:>16}: E_mean={r['E_mean_V_per_m']:.3g} V/m, "
                  f"E_p95={r['E_p95_V_per_m']:.3g} V/m")
