"""Build the interactive 3D views: model + electrodes, then current density.

Produces self-contained HTML files you open in a browser and rotate; every
legend entry is a toggle, so you can go

    whole head  ->  eye only  ->  retina only

without regenerating anything.  Also writes static PNG snapshots for reports.

One of the project's earliest viz3d scripts, from before the full 6-montage
912x900x504 work existed -- it's kept because the toggle-layered "whole
head -> eye -> retina" view it builds is still a genuinely useful shape for
a single montage, but it predates a consistent way to regenerate its input.
It needs a ``fields.npz`` (``Ex``, ``Ey``, ``Ez``, ``Emag``, ``Jmag``,
``labels_ref`` for the small ``RatCC_eye_180_160_220_83um_with_retina``
crop) that this repo does not currently ship or regenerate automatically --
solve that model the way ``examples/22_retina_distribution_study.py`` does
and save those arrays to reproduce it. For the current, complete
equivalent across all 6 full-head montages, see
``examples/37_neuroam_3d_six_configs.py`` instead.

Run (once fields.npz exists next to this script's SAMP/verification dir):
    python examples/20_view_model_and_field.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from neuroam import viz3d
from neuroam.materials import MaterialLibrary
from neuroam.model import VoxelModel

HERE = Path(__file__).resolve().parent
SAMP = HERE.parent / "samples" / "ratcc_eye_83um"
VER = SAMP / "verification"
OUT = VER / "views"
OUT.mkdir(parents=True, exist_ok=True)

DX = 83e-6
# rat eye tissue labels in this model
RETINA, VITREOUS, SCLERA_ISH, LENS_ISH = 77, 55, 66, 33
STIM, GND, INSUL = 101, 102, 110

FOCUS = {"retina": [RETINA], "vitreous": [VITREOUS],
         "cornea / outer eye": [SCLERA_ISH], "lens region": [LENS_ISH]}


def main():
    z = np.load(VER / "fields.npz")
    labels = z["labels_ref"]
    Jmag = z["Jmag_A"]
    Emag = z["Emag_A"]

    lib = MaterialLibrary.from_in_file(
        SAMP / "RatCC_eye_180_160_220_83um_with_retina.in")
    model = VoxelModel(labels=labels, dx=DX, materials=lib,
                       name="RatCC eye (83 µm, CL-stim / J-gnd)")

    stim_ctr = np.argwhere(labels == STIM).mean(0).astype(int)
    eye_ctr = np.argwhere(np.isin(labels, [RETINA, VITREOUS])).mean(0).astype(int)

    # ---------------------------------------------------------------- model
    print("building model view ...")
    viz3d.view_model(
        model, OUT / "01_model_and_electrodes.html",
        focus=FOCUS, step=1,
        title="Rat head & eye — model and electrodes",
        subtitle="83 µm voxels · contact-lens ring stimulator (red) · "
                 "J return electrode (blue) · toggle layers in the legend",
        theme="dark")

    # ------------------------------------------------- current density views
    print("building current-density view ...")
    retina = labels == RETINA
    jr = Jmag[retina]
    lo, hi = np.percentile(jr, [2, 99.5])
    viz3d.view_current_density(
        model, Jmag, OUT / "02_current_density.html",
        focus=FOCUS, step=1,
        iso_levels=[float(np.percentile(jr, p)) for p in (50, 90, 99)],
        slices=[("y", int(eye_ctr[1])), ("z", int(eye_ctr[2]))],
        log=True,
        title="Current density |J| — CL-stim / J-ground montage",
        subtitle="per 1 A injected · retina surface coloured by |J| · "
                 "start with 'retina' alone, then add layers",
        theme="dark")

    # retina-only close-up: the view the dose numbers come from
    print("building retina close-up ...")
    idx = np.argwhere(retina)
    pad = 6
    crop = tuple((int(max(0, idx[:, k].min() - pad)),
                  int(min(labels.shape[k], idx[:, k].max() + pad)))
                 for k in range(3))
    sc = viz3d.Scene(dx=DX, crop=crop, theme="dark",
                     title="Retina only — |J| on the retinal surface",
                     subtitle="per 1 A injected · other structures hidden")
    sc.add_field_on_surface(labels, [RETINA], Jmag, "retina — |J| (A/m²)",
                            log=True, cmin=np.log10(max(lo, 1e-30)),
                            cmax=np.log10(max(hi, 1e-30)),
                            colorbar_title="log₁₀ |J| (A/m²)")
    sc.add_materials(labels, [VITREOUS], "vitreous (context)", role="context",
                     opacity=0.10, visible="legendonly")
    sc.add_electrode(labels, [STIM], "stimulating electrode", role="source")
    sc.add_electrode(labels, [GND], "return electrode", role="ground")
    sc.add_field_points(Jmag, "hot spots", mask=retina, percentile=97,
                        log=True, colorbar_title="log₁₀ |J|")
    sc.add_note("retina isolated")
    sc.write_html(OUT / "03_retina_only.html")

    # ------------------------------------------------------------ snapshots
    print("rendering PNG snapshots ...")
    viz3d.snapshot_png(OUT / "model_snapshot.png", model,
                       focus_ids=[RETINA], electrode_ids=[STIM, GND],
                       centre=eye_ctr, title="model: materials, eye & electrodes")
    viz3d.snapshot_png(OUT / "current_density_snapshot.png", model, field=Jmag,
                       focus_ids=[RETINA], electrode_ids=[STIM, GND],
                       centre=eye_ctr, log=True,
                       title="|J| per 1 A — retina outlined in green, electrodes in red")

    for p in sorted(OUT.iterdir()):
        print("  ", p.name, f"{p.stat().st_size/1e6:.1f} MB")
    print(f"\nopen the .html files in a browser: {OUT}")


if __name__ == "__main__":
    main()
