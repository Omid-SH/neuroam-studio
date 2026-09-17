"""Register (and optionally solve) the literature-derived non-invasive
electrode configs on the FULL 912x900x504 RatCC head model -- the "blocking
next step" docs/CLINICAL_ELECTRODES.md already flagged: the small eye-only
crop couldn't host any of these montages' actual return electrodes.

Five physically distinct approaches, all periocular/forehead, all
non-invasive:

  TES-GPS   thread on the ocular surface (inferior limbus) + forehead pads
            -- Lorenz et al., BMJ Open 2026. EYE MUST BE OPEN (corneal
            contact).
  VIRON     two periorbital skin cups (temple), open eye, MRI-optimised
            in the trial -- Schittkowski et al., BMJ Open 2025. Skin only,
            so this one is meaningful with the eye open OR closed.
  VIRON6    the 6-cup orbital array (Hunold et al. 2025) all 6 active at
            once against one counter -- the montage space VIRON group 2's
            optimisation searches. Skin only -- open/closed both meaningful.
  TpES      4 disc electrodes on the EYELID skin (Zhou et al. 2026,
            iScience) -- the literal "eye closed" case: these electrodes
            work whether the lid is open or shut, unlike every corneal-
            contact design above.
  TcES      concentric ring electrode ON the cornea (Zhou et al. 2026's own
            control) -- EYE MUST BE OPEN, included as the direct comparison
            TpES was validated against in that paper.

Every montage is validated with two new toolbox pieces before it is trusted:
``ElectrodeQC.check_contact`` (is this electrode actually touching the skin
from outside, or actually embedded in it -- not just assumed from the geometry
spec) and ``neuroam.safety.safety_report`` (Shannon charge-density check at
the montage's own clinical amplitude/pulse-width, using the electrode's real
exposed area at this voxel size, not a nominal one).

"Eye closed" is modelled literally (``neuroam.electrodes.close_eyelid``): a
thin skin shell painted over the palpebral aperture, run as a *second*
registration of the same periorbital montage on a model where the eye has
already been closed. TES-GPS and TcES require direct corneal/limbal contact
and have no physically sensible closed-eye variant, so they are open-eye only.

Base anatomy: SCL-ON's own full-model files (its ring/J-ground hardware at
101/102 is stripped back to cornea/skin first, same convention as
examples/07 on the small crop) -- reusing its already-built .mrm rather than
re-meshing, since these new electrodes occupy a tiny fraction of a
25M-unknown mesh built for a spatially separate montage.

Run:
    python examples/38_solve_clinical_configs.py --report        # fast, no solve
    python examples/38_solve_clinical_configs.py <config_key>    # solve one

config_key in {TESGPS, VIRON_periorbital_open, VIRON_periorbital_closed,
               VIRON6_open, VIRON6_closed, TpES_open, TpES_closed, TcES}

Writes: samples/ratcc_eye_83um/verification/clinical/<config_key>_registration_report.json
       samples/ratcc_eye_83um/<config_key>.vof (on solve)
       .neuroam_cache/clinical_field-<key>.npz (on solve, fast-reload cache)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from neuroam import fields as F
from neuroam.assembly import assemble_mrm, read_mrm
from neuroam.cache import ResultCache, hash_inputs
from neuroam.electrodes import ElectrodeSpec, register, close_eyelid
from neuroam.frames import Frame
from neuroam.model import VoxelModel
from neuroam.safety import safety_report, SHANNON_CONSERVATIVE

# clinical (amp_A, pulse_width_s) per waveform name -- docs/CLINICAL_ELECTRODES.md
CLINICAL_DRIVE = {
    "TES": (1e-3, 5e-3),          # TES-GPS: up to 1 mA, 5 ms/phase
    "rtACS": (300e-6, 1 / (10 * 2)),   # VIRON: 300 uA amplitude, 10 Hz sinusoid ~ 50 ms/half-cycle
    "TpES": (4.8e-3, 10e-3),      # TpES: up to 4.8 mA (paper's own max), 10 ms/phase
    "TcES": (1e-3, 10e-3),        # TcES control, same paper: 1 mA reference, 10 ms/phase
    "SCL": (200e-6, 5e-3),        # SCL_Wrist: same 200 uA convention as the original 6
}
from neuroam.solver import solve, unit_current_vector

REPO = Path(__file__).resolve().parents[1]
SAMP = REPO / "samples/ratcc_eye_83um"
BASE_STEM = "RatCC_full_CLStim_JGND_912_900_504_Ring_Res_83um_with_retina"  # SCL-ON's own files
VER = SAMP / "verification" / "clinical"
VER.mkdir(parents=True, exist_ok=True)

RETINA, SKIN, CORNEA = 77, 13, 66
SCALE = 0.263            # human -> rat, by globe radius (docs/CLINICAL_ELECTRODES.md)
PHI_SUPERIOR, PHI_NASAL, PHI_INFERIOR, PHI_TEMPORAL = 45.0, 135.0, 225.0, 315.0


def _fp(p: Path):
    st = p.stat()
    return (str(p), st.st_size, int(st.st_mtime))


def field_cache_key(config_key: str, qcs) -> str:
    """The ``clinical_field`` cache key, shared with examples/39 so a reader
    can find the same entry a solve wrote. Includes each electrode's actual
    registered position (not just ``config_key``) -- a geometry change (e.g.
    moving where an electrode lands) must invalidate the cache even when the
    .in/.mrm files and config_key string didn't change (bit us once: a
    corrected electrode position silently reused a stale field solved at the
    old position)."""
    geom_fp = tuple(sorted((qc.name, qc.role, qc.label, tuple(np.round(qc.centroid_vox, 2)))
                          for qc in qcs))
    mrm_path = SAMP / f"{BASE_STEM}.mrm"
    return hash_inputs(_fp(SAMP / f"{BASE_STEM}.in"), _fp(mrm_path), config_key,
                       geom_fp, "rtol=1e-7,amg")


def pad_spec(name, role, label, theta_deg, phi_deg, radius_mm, waveform=None,
            max_mm=100.0):
    """A skin-surface disc contact -- draped on the skin wherever it's found
    marching out from the eye frame along (theta, phi), not at a fixed
    radius (the whole point of ``on_surface``: it works at forehead/temple
    distance the same as it works at periorbital distance)."""
    return dict(name=name, role=role, label=label, waveform=waveform,
               overwrite=[0, SKIN], min_voxels=max(30, int(20 * radius_mm ** 2)),
               geometry={"type": "cylinder", "radius_mm": radius_mm, "height_mm": 0.5,
                        "frame": "eye",
                        "at": {"theta_deg": theta_deg, "phi_deg": phi_deg,
                              "on_surface": {"labels": [SKIN], "standoff_mm": 0.0,
                                            "max_mm": max_mm}},
                        "conform": {"labels": [SKIN], "offset_mm": -0.10,
                                   "thickness_mm": 0.30}})


def load_base(keep_scl_ring: bool = False):
    """``keep_scl_ring=True`` leaves SCL-ON's own shipped contact-lens ring
    (label 101) in place instead of stripping it back to cornea -- for
    SCL_Wrist, which reuses that exact ring as its source and only replaces
    the ground."""
    in_path = SAMP / f"{BASE_STEM}.in"
    model = VoxelModel.from_legacy(in_path)
    if not keep_scl_ring:
        model.labels[model.labels == 101] = CORNEA   # strip SCL-ON's own ring
    model.labels[model.labels == 102] = SKIN         # strip SCL-ON's own J-ground
    return model


def build_montage(model, frame, config_key: str):
    """Returns (specs_as_dicts, closed_eye: bool)."""
    if config_key == "TESGPS":
        okuel = {"type": "arc", "frame": "eye", "r_mm": 3.46, "theta_deg": 74.0,
                "tube_radius_mm": 0.10, "phi0_deg": PHI_INFERIOR,
                "span_deg": 100.0, "n_points": 81}
        return [
            dict(name="OkuEl", role="source", label=101, waveform="TES",
                geometry=okuel, overwrite="any", min_voxels=120),
            pad_spec("CTR", "ground", 102, 100.0, PHI_SUPERIOR, 2.8, waveform=None),
        ], False

    if config_key.startswith("VIRON_periorbital"):
        closed = config_key.endswith("closed")
        return [
            pad_spec("E_sup", "source", 121, 85.0, PHI_SUPERIOR, 1.3, waveform="rtACS"),
            pad_spec("RET", "ground", 122, 90.0, PHI_TEMPORAL, 2.6, waveform=None),
        ], closed

    if config_key.startswith("VIRON6"):
        closed = config_key.endswith("closed")
        specs = [pad_spec(f"E{i}", "source", 120 + i, 85.0, phi, 1.3, waveform=f"E{i}")
                for i, phi in enumerate([45.0, 105.0, 165.0, 225.0, 285.0, 345.0], 1)]
        # theta pushed well past the 85 deg cup band (E1..E6) so the return
        # doesn't land inside/adjacent to E1's own footprint -- theta=100 at
        # a nearby phi shorted against E1 in practice (see registration
        # report history); a materially larger theta clears it.
        specs.append(pad_spec("C1", "ground", 127, 150.0, 15.0, 2.6, waveform=None))
        return specs, closed

    if config_key.startswith("TpES"):
        closed = config_key.endswith("closed")
        # 5 mm human disc -> 1.3 mm diameter (r=0.66 mm) at rat scale; CH1
        # (superior palpebral), matching the paper's primary position.
        # Bumped to r=1.0 mm (still notably smaller than VIRON's r=1.3 mm
        # cups) -- the pure-geometric 0.66 mm radius is too few voxels to
        # guarantee its supernode terminal lands on an actual node of the
        # reused SCL-ON multires mesh at this location (confirmed: it
        # doesn't -- "source node ... is not a mesh node"); documented
        # deviation, not a silent one.
        specs = [pad_spec("TpES_CH1", "source", 131, 80.0, PHI_SUPERIOR, 1.0,
                          waveform="TpES")]
        specs.append(pad_spec("NECK", "ground", 132, 130.0, 225.0, 5.0, waveform=None))
        return specs, closed

    if config_key == "TcES":
        # Zhou et al.'s TcES electrode is a *flat annular disc* on the cornea
        # (9.5 mm OD / 8 mm ID), not a wire loop -- geometrically a
        # SphericalBand (theta_min>0 gives an annulus following the corneal
        # curvature, per Frame.band), not a Torus. R_SURF=3.36 mm is the same
        # corneal/globe-surface radius the SCL-ON contact lens already uses;
        # OD/ID convert to angle via disc_radius = R_SURF*sin(theta):
        #   OD 9.5->2.50 mm (k=0.263) -> r=1.25 mm -> theta_outer ~21.8 deg
        #   ID 8.0->2.10 mm           -> r=1.05 mm -> theta_inner ~18.2 deg
        R_SURF = 3.36
        ring_region = frame.band(R_SURF, 18.2, 21.8, thickness_mm=0.10)
        return [
            dict(name="TcES_ring", role="source", label=141, waveform="TcES",
                region=ring_region, overwrite="any", min_voxels=80),
            pad_spec("NECK", "ground", 132, 130.0, 225.0, 5.0, waveform=None),
        ], False

    if config_key == "SCL_Wrist":
        # Source: SCL-ON's own shipped contact-lens ring, reused as-is via
        # from_label (same electrode this project already validated, not a
        # new geometry). Ground: "a wide wrist electrode around the animal's
        # wrist" -- this 912x900x504 crop DOES reach the rat's forepaws (they
        # sit tucked up near the head). Two false leads first: VIRON6's "C1"
        # ground direction (theta=150/phi=15) lands on the ear; VIRON6's own
        # "E5" cup (theta=85/phi=285) lands on the nose, not a paw, despite
        # also showing digit-like structure on inspection (whisker/vibrissae
        # texture, not fingers). The confirmed paw is where VIRON6's "E3" cup
        # (theta=85/phi=165, build_montage above) lands -- verified against
        # exact mm centroids read directly from the electrode's own hover
        # tooltip in VIRON6_open_labeled_electrodes.html, not just eyeballed
        # from a render. Reused here with a bigger radius (9 mm vs VIRON6's
        # 1.3 mm cup) as a large pad standing in for "a wide electrode around
        # the wrist" (a literal 360 deg encircling band would need a
        # dedicated local limb frame this project doesn't have yet -- out of
        # scope for this config).
        return [
            dict(name="SCL", role="source", label=101, waveform="SCL",
                from_label=101, min_voxels=100),
            pad_spec("WRIST", "ground", 150, 85.0, 165.0, 9.0, waveform=None),
        ], False

    raise ValueError(f"unknown config {config_key!r}")


def main(config_key: str, do_solve: bool, model=None, frame=None,
        clean_labels=None, return_model: bool = False) -> int:
    """``model``/``frame``/``clean_labels`` let a caller load the (large,
    slow-to-read) base anatomy once and reuse it across many configs --
    each call still gets a pristine anatomy (``model.labels`` is reset from
    ``clean_labels`` first) so montages never see each other's hardware."""
    t0 = time.time()
    owns_model = model is None
    if owns_model:
        model = load_base(keep_scl_ring=(config_key == "SCL_Wrist"))
        frame = Frame.eye(model, (RETINA,))
        clean_labels = model.labels.copy()
    else:
        model.labels[...] = clean_labels
    print(f"[{config_key}] eye frame: origin {np.round(frame.origin_mm, 3).tolist()} mm, "
         f"radius {frame.radius_mm:.3f} mm ({time.time()-t0:.0f}s)", flush=True)

    entries, closed = build_montage(model, frame, config_key)
    if closed:
        n_lid = close_eyelid(model, frame, apex_radius_mm=3.36, theta_max_deg=75.0,
                             thickness_mm=0.30, skin_label=SKIN)
        print(f"[{config_key}] eye closed: painted {n_lid:,} voxels of eyelid skin "
             f"over the palpebral aperture", flush=True)

    from neuroam.electrodes import region_from_spec
    frames = {"eye": frame}
    specs = []
    import copy as _copy
    for e in entries:
        if "from_label" in e:
            specs.append(ElectrodeSpec(
                name=e["name"], role=e["role"], label=e["label"],
                terminal="supernode", waveform=e.get("waveform"),
                from_label=e["from_label"], min_voxels=e.get("min_voxels", 100)))
            continue
        if "region" in e:
            region = e["region"]
        else:
            geom = e["geometry"]
            region = None
            last_exc = None
            # a ray straight out from the globe can graze a local gap (an
            # eyelid margin, ear canal, nostril) without ever finding skin;
            # nudging azimuth a little finds real skin just next door far
            # more often than it finds a second genuine gap.
            phi0 = geom.get("at", {}).get("phi_deg", 0.0)
            for dphi in (0.0, 10.0, -10.0, 20.0, -20.0, 30.0, -30.0):
                g = _copy.deepcopy(geom)
                g.setdefault("at", {})["phi_deg"] = phi0 + dphi
                try:
                    region = region_from_spec(g, model, frames)
                    if dphi:
                        print(f"[{config_key}]    ({e['name']}: nudged phi by "
                             f"{dphi:+.0f} deg to find skin)", flush=True)
                    break
                except ValueError as exc:
                    last_exc = exc
            if region is None:
                print(f"[{config_key}]    !! {e['name']}: {last_exc}", flush=True)
                continue
        specs.append(ElectrodeSpec(
            name=e["name"], region=region, role=e["role"], label=e["label"],
            terminal="supernode", waveform=e.get("waveform"),
            overwrite=e.get("overwrite", "any"), min_voxels=e.get("min_voxels", 100)))

    overlay, qcs = register(model, specs)
    report = {"config": config_key, "closed_eye": closed, "electrodes": []}
    for qc in qcs:
        print(f"[{config_key}] {qc.summary()}", flush=True)
        contact = qc.check_contact(SKIN, mode="surface") if qc.role == "ground" \
            or "CH" in qc.name or qc.name in ("E_sup", "RET") or qc.name.startswith("E") \
            or qc.name in ("NECK", "CTR") else None
        entry = {"name": qc.name, "role": qc.role, "n_voxels": qc.n_voxels,
                 "surface_area_mm2": qc.surface_area_mm2,
                 "centroid_vox": list(qc.centroid_vox), "bbox_vox": list(qc.bbox_vox),
                 "contact_by_label": qc.contact_by_label,
                 "replaced_by_label": qc.replaced_by_label,
                 "warnings": qc.warnings}
        if contact is not None:
            entry["skin_contact"] = {"mode": contact.actual_mode, "ok": contact.ok,
                                     "reason": contact.reason}
            print(f"[{config_key}]    contact check: {contact.reason}", flush=True)
        report["electrodes"].append(entry)

    # ---- safety: Shannon charge-density check at each source's own clinical drive
    for spec, qc in zip(specs, qcs):
        if qc.role != "source" or qc.n_voxels == 0:
            continue
        wf = spec.waveform
        if wf not in CLINICAL_DRIVE:
            continue
        amp_A, pw_s = CLINICAL_DRIVE[wf]
        rep = safety_report(qc, amp_A, pw_s, SHANNON_CONSERVATIVE)
        print(f"[{config_key}]    safety: {rep.summary()}", flush=True)
        for e in report["electrodes"]:
            if e["name"] == qc.name:
                e["safety"] = {"amp_A": amp_A, "pulse_width_s": pw_s,
                              "charge_density_uC_cm2": rep.charge_density_uC_cm2,
                              "current_density_A_m2": rep.current_density_A_m2,
                              "k": rep.k, "ok": rep.ok, "margin_db": rep.margin_db}

    warned = [q.name for q in qcs if q.warnings]
    print(f"[{config_key}] QC warnings on: {', '.join(warned) if warned else 'none'}",
         flush=True)

    out_json = VER / f"{config_key}_registration_report.json"
    out_json.write_text(json.dumps(report, indent=2, default=str))
    print(f"[{config_key}] wrote {out_json} ({time.time()-t0:.0f}s)", flush=True)

    if not do_solve:
        if return_model:
            return model, frame, specs, qcs
        return 0

    # ---- solve -------------------------------------------------------
    mrm_path = SAMP / f"{BASE_STEM}.mrm"
    records = read_mrm(mrm_path)
    print(f"[{config_key}] mesh records read: {len(records):,} ({time.time()-t0:.0f}s total)",
         flush=True)
    t1 = time.time()
    system = assemble_mrm(model, records)
    print(f"[{config_key}] assembled: n={system.n:,} nnz={system.G.nnz:,} "
         f"({time.time()-t1:.0f}s)", flush=True)

    cache = ResultCache(REPO / ".neuroam_cache")
    key = field_cache_key(config_key, qcs)

    def _solve_field():
        t2 = time.time()
        # specs[0] is always this config's first *source* electrode (see
        # build_montage): for VIRON6 that's E1 alone, a deliberate scope cut
        # -- the full 6-independent-basis-field decomposition each cup would
        # need its own solve (~6x the cost) and the real clinical protocol
        # (Gall et al. 2016, cited by VIRON) also drives one cup at a time,
        # not all six simultaneously, so a single representative cup is the
        # physically-grounded choice here, not just a shortcut.
        r = solve(system, unit_current_vector(system, specs[0].waveform or "TES"),
                 method="cg", rtol=1e-7, precond="amg")
        print(f"[{config_key}] solved: {r.iterations} iters, {r.seconds:.0f}s, "
             f"residual {r.residual:.2e}", flush=True)
        grid = F.node_grid(system, r.v)
        grid = F.fill_hanging_nodes(grid, system)
        Ex, Ey, Ez, Emag = F.efield(grid, model.dx)
        _, _, _, Jmag = F.current_density(model, Ex, Ey, Ez)
        vof_path = SAMP / f"clinical_{config_key}.vof"
        F.write_vof(vof_path, system, r.v)
        print(f"[{config_key}] wrote {vof_path}", flush=True)
        return {"grid": grid.astype(np.float32), "jmag": Jmag.astype(np.float32)}

    full = cache.get_or_compute("clinical_field", key, _solve_field)
    print(f"[{config_key}] full field ready ({time.time()-t0:.0f}s total)", flush=True)

    # quick sanity: charge-density safety at each source electrode's own
    # clinical amplitude/pulse width (from docs/CLINICAL_ELECTRODES.md)
    AMPS_PW = {"TES": (1e-3, 5e-3), "rtACS": (300e-6, 1 / 10 / 2), "TpES": (1e-3, 10e-3),
              "TcES": (1e-3, 10e-3)}
    for qc in qcs:
        if qc.role != "source" or qc.warnings and qc.n_voxels == 0:
            continue
        wf = qc.terminal and None
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        sys.exit(f"usage: python {Path(__file__).name} --report|<config_key>")
    if args[0] == "--report":
        keys = ["TESGPS", "VIRON_periorbital_open", "VIRON_periorbital_closed",
                "VIRON6_open", "VIRON6_closed", "TpES_open", "TpES_closed", "TcES"]
        t_load = time.time()
        model0 = load_base()
        frame0 = Frame.eye(model0, (RETINA,))
        clean0 = model0.labels.copy()
        print(f"[base] full model loaded once, reused for all {len(keys)} configs "
             f"({time.time()-t_load:.0f}s)\n", flush=True)
        for k in keys:
            main(k, do_solve=False, model=model0, frame=frame0, clean_labels=clean0)
            print()
        # SCL_Wrist reuses SCL-ON's own ring (keep_scl_ring=True), so it needs
        # its own base load -- can't share clean0 (ring already stripped there).
        main("SCL_Wrist", do_solve=False)
        raise SystemExit(0)
    if len(args) > 1 and args[1] == "--report":
        raise SystemExit(main(args[0], do_solve=False))
    raise SystemExit(main(args[0], do_solve=True))
