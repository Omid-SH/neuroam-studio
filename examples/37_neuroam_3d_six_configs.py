"""A single interactive 3D scene: current density on the retina, optic
nerve, and occipital cortex, for each of the 6 electrode-pair montages, at
200 uA -- built entirely from NeuroAM's own solved fields (the same
``.neuroam_cache`` basis fields ``examples/28`` produced and
``examples/33_verify_against_oecrgc.py`` cross-validated against the legacy
OEC-RGC pipeline to ~1%). No re-solving.

One scene, one crop tight enough to hold all three regions (a ~282x316x182
voxel box -- small even though the regions themselves are anatomically
spread out, because none of the three is individually large). Every
region/config pair is its own legend entry -- only SCL-ON's three regions
are visible by default; click any other config's entries to compare. The
retina is rendered as a real isosurface (it is a thin, well-defined shell);
the optic nerve and occipital cortex are rendered as colored point clouds
(a tube and a diffuse volume respectively -- points read better than a
marching-cubes shell for either). Each trace auto-scales its own color
range to its own data -- this view is for spatial *pattern* within and
across configs, not a strict shared-colorbar magnitude comparison (see
``examples/36_neuroam_six_config_distribution.py`` / its box-plot figure
for that).

Run:  python examples/37_neuroam_3d_six_configs.py
Writes: samples/ratcc_eye_83um/verification/views/
       neuroam_six_config_current_density_3d.html
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from neuroam import viz3d
from _sweep_common import CachedMontage, CONFIGS

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT = REPO / "samples/ratcc_eye_83um/verification/views"
OUT.mkdir(parents=True, exist_ok=True)

RETINA, OPTIC_NERVE, BRAIN = 77, 24, 18
STIM, GND = 101, 102
CUR1_AMP_A = 200e-6           # see examples/33's cross-validation
SPLIT_DEG = 5.0

# union bounding box of retina(77) + optic nerve(24) + occipital(brain(18) &
# y>550 & z>300), + a little padding -- computed once from the shared
# anatomy (identical across all 6 configs; only electrode placement differs)
PAD = 8
CROP = ((282 - PAD, 564 + PAD), (338 - PAD, 654 + PAD), (202 - PAD, 384 + PAD))


def main():
    sc = viz3d.Scene(dx=83e-6, crop=CROP, theme="dark", units="mm",
                     title="Current density: retina, optic nerve, occipital cortex",
                     subtitle="6 electrode-pair montages at 200 uA — NeuroAM's own "
                              "solve, cross-validated to the OEC-RGC legacy pipeline to "
                              "~1% (docs/VALIDATION.md) — click legend entries to "
                              "toggle regions/configs; only SCL-ON shown by default")

    for i, cfg in enumerate(CONFIGS):
        print(f"[{cfg}] loading ...", flush=True)
        m = CachedMontage(cfg)
        labels = m.model.labels
        J = m.jmag * CUR1_AMP_A          # A/m^2 at 200 uA

        on_mask = labels == OPTIC_NERVE
        Yg, Zg = np.indices((labels.shape[1], labels.shape[2]))
        occ_mask = (labels == BRAIN) & ((Yg > 550) & (Zg > 300))[None, :, :]

        visible = (cfg == "SCL-ON")
        vis = True if visible else "legendonly"

        sc.add_field_on_surface(labels, [RETINA], J, f"{cfg} — retina |J|",
                                log=True, colorbar_title="log10 |J| (A/m2)",
                                visible=vis)
        sc.add_field_points(J, f"{cfg} — optic nerve |J|", mask=on_mask,
                            percentile=0, size=2.5, log=True,
                            colorbar_title="log10 |J| (A/m2)", visible=vis)
        sc.add_field_points(J, f"{cfg} — occipital |J|", mask=occ_mask,
                            percentile=95, size=1.6, log=True,
                            colorbar_title="log10 |J| (A/m2)", visible=vis)
        sc.add_electrode(labels, [STIM], f"{cfg} — stim electrode",
                         role="source", visible=vis)
        sc.add_electrode(labels, [GND], f"{cfg} — gnd electrode",
                         role="ground", visible=vis)
        print(f"[{cfg}] added ({i+1}/{len(CONFIGS)})", flush=True)
        del m, J, labels, on_mask, occ_mask

    sc.add_note("retina: isosurface, colored by |J| · optic nerve / occipital: "
               "point clouds (optic nerve: all voxels; occipital: hottest 5%) · "
               "each trace auto-scales its own color range")
    out = OUT / "neuroam_six_config_current_density_3d.html"
    sc.write_html(out)
    print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
