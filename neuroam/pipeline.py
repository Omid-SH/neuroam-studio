"""ASCENT-style config-driven pipeline.

A single versioned JSON config describes model, materials, electrodes,
waveforms, solve options, outputs, and (optionally) NEURON coupling.  The
runner executes: build → assemble → basis-field solve (cached) → fields →
exports → coupling.  Every run writes a ``run_manifest.json`` with hashes,
solver diagnostics, and produced files, for reproducibility.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from . import __version__
from .assembly import assemble, System
from .cache import ResultCache, hash_inputs
from .coupling import (build_v_matrix, read_coordinates, sample_node_grid,
                       superpose_v_matrix, transform_coordinates,
                       write_unit_field, write_v_file, interpolation_report)
from .fields import (current_density, efield, fill_hanging_nodes,
                     mask_from_labels, mask_sphere_shell, node_grid,
                     roi_metrics, voxel_average, write_raw, write_vavg,
                     write_vof)
from .materials import Material, MaterialLibrary
from .model import VoxelModel
from .solver import kcl_report, solve, solve_basis, unit_current_vector
from .waveforms import Waveform, from_config as waveform_from_config
from . import viz

SCHEMA_VERSION = "0.1"


# ------------------------------------------------------------------ config
def load_config(path) -> Dict[str, Any]:
    cfg = json.loads(Path(path).read_text())
    if str(cfg.get("neuroam", SCHEMA_VERSION)) != SCHEMA_VERSION:
        raise ValueError(f"config schema {cfg.get('neuroam')} != {SCHEMA_VERSION}")
    cfg["_dir"] = str(Path(path).resolve().parent)
    return cfg


def _resolve(cfg: Dict[str, Any], p: Optional[str]) -> Optional[Path]:
    if p is None:
        return None
    q = Path(p)
    return q if q.is_absolute() else Path(cfg.get("_dir", ".")) / q


# ------------------------------------------------------------------ builders
def build_materials(cfg: Dict[str, Any]) -> MaterialLibrary:
    mcfg = cfg.get("materials", {})
    if "json" in mcfg:
        return MaterialLibrary.from_json(_resolve(cfg, mcfg["json"]))
    lib = MaterialLibrary()
    for d in mcfg.get("inline", []):
        d = dict(d)
        if "sigma" in d:
            sigma = d.pop("sigma")
            lib.add(Material.from_sigma(**d, sigma=sigma))
        else:
            rho = d.pop("rho")
            if isinstance(rho, (int, float)):
                lib.add(Material.isotropic_rho(rho=float(rho), **d))
            else:
                lib.add(Material(rho=tuple(rho), **d))
    if "default_material" in mcfg:
        lib.default_material = mcfg["default_material"]
    return lib


def build_model(cfg: Dict[str, Any]) -> VoxelModel:
    mc = cfg["model"]
    if "legacy_in" in mc:
        model = VoxelModel.from_legacy(_resolve(cfg, mc["legacy_in"]),
                                       model_path=_resolve(cfg, mc.get("legacy_model")))
        # allow config-side extra materials/overrides
        extra = build_materials(cfg)
        for m in extra:
            model.materials.add(m)
        return model

    lib = build_materials(cfg)
    model = VoxelModel.empty(mc["world"], float(mc["unit_voxel_m"]), lib,
                             background=int(mc.get("background", 0)),
                             name=cfg.get("name", "model"))
    for shp in mc.get("shapes", []):
        s = dict(shp)
        kind = s.pop("type")
        mat = int(s.pop("material"))
        getattr(model, f"add_{kind}")(mat, **s)
    for el in cfg.get("electrodes", []):
        e = dict(el)
        shape = e.pop("shape", {"type": "none"})
        skw = dict(shape)
        stype = skw.pop("type")
        model.add_electrode(shape=stype, role=e["role"],
                            material=int(e.get("material",
                                               101 if e["role"] == "source" else 100)),
                            waveform=e.get("waveform", e.get("name", "stim")),
                            node=e.get("node"), **skw)
    return model


def build_masks(cfg, model: VoxelModel) -> Dict[str, np.ndarray]:
    out = {}
    for r in cfg.get("outputs", {}).get("rois", []):
        name = r["name"]
        if "labels" in r:
            out[name] = mask_from_labels(model, r["labels"])
        elif r.get("type") == "sphere_shell":
            out[name] = mask_sphere_shell(model.world, r["center"], r["r_in"],
                                          r["r_out"], r.get("cone_axis"),
                                          r.get("cone_half_angle_deg"))
        else:
            raise ValueError(f"unknown roi spec: {r}")
    return out


# ------------------------------------------------------------------ runner
def run(config_path, out_dir: Optional[str] = None,
        progress=print) -> Dict[str, Any]:
    cfg = load_config(config_path)
    t_start = time.time()
    name = cfg.get("name", Path(config_path).stem)

    out = Path(out_dir) if out_dir else \
        _resolve(cfg, cfg.get("outputs", {}).get("dir", f"results_{name}"))
    out.mkdir(parents=True, exist_ok=True)
    produced: List[str] = []
    manifest: Dict[str, Any] = {
        "neuroam_version": __version__, "schema": SCHEMA_VERSION,
        "name": name, "python": sys.version.split()[0],
        "platform": platform.platform(),
        "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
    }

    progress(f"[neuroam] building model '{name}' ...")
    model = build_model(cfg)
    progress("  " + model.summary().replace("\n", "\n  "))

    solve_cfg = cfg.get("solve", {})
    cache = ResultCache(root=out / ".cache",
                        enabled=bool(solve_cfg.get("cache", True)))

    mrm = _resolve(cfg, cfg["model"].get("mrm")) if "mrm" in cfg.get("model", {}) \
        else None
    progress("[neuroam] assembling admittance system "
             + ("(multiresolution .mrm)" if mrm else "(uniform, direct)"))
    system = assemble(model, mrm_path=mrm)
    manifest["system"] = {"n_nodes": int(len(system.node_coords)),
                          "n_unknowns": int(system.n),
                          "nnz": int(system.G.nnz)}
    progress(f"  {system.n:,} unknowns, {system.G.nnz:,} nonzeros")

    # ---- basis-field solves (cached) ----
    model_key = hash_inputs(model.labels, model.dx,
                            [(m.id, m.rho, m.rho_im) for m in model.materials],
                            sorted(system.source_rows.items()),
                            model.ground_nodes,
                            str(mrm) if mrm else "uniform",
                            solve_cfg.get("rtol", 1e-8))
    basis: Dict[str, np.ndarray] = {}
    solver_stats = {}
    for src in system.source_rows:
        key = hash_inputs(model_key, src)

        def compute():
            progress(f"[neuroam] solving unit-current field for source '{src}' ...")
            r = solve(system, unit_current_vector(system, src),
                      method=solve_cfg.get("method", "auto"),
                      rtol=float(solve_cfg.get("rtol", 1e-8)),
                      precond=solve_cfg.get("precond", "diag"))
            solver_stats[src] = {"method": r.method, "iterations": r.iterations,
                                 "residual": r.residual, "seconds": r.seconds}
            return {"v": r.v}

        got = cache.get_or_compute("basis", key, compute)
        if src not in solver_stats:
            solver_stats[src] = {"cached": True}
            progress(f"[neuroam] reused cached field for source '{src}'")
        basis[src] = got["v"]
    manifest["solver"] = solver_stats

    # conservation check on the first basis field
    first = next(iter(basis))
    manifest["conservation"] = kcl_report(
        system, basis[first], unit_current_vector(system, first))

    # ---- waveforms ----
    waveforms: Dict[str, Waveform] = {}
    for wname, wcfg in cfg.get("waveforms", {}).items():
        wcfg = dict(wcfg)
        if wcfg.get("type") == "cur_file":
            wcfg["path"] = str(_resolve(cfg, wcfg["path"]))
        waveforms[wname] = waveform_from_config(wcfg)

    # ---- fields & exports ----
    ocfg = cfg.get("outputs", {})
    save = set(ocfg.get("save", ["npz", "png"]))
    grid = node_grid(system, basis[first])
    grid = fill_hanging_nodes(grid, system)
    vavg = voxel_average(grid)
    Ex, Ey, Ez, Emag = efield(grid, model.dx)
    Jx, Jy, Jz, Jmag = current_density(model, Ex, Ey, Ez)

    def emit(p: Path):
        produced.append(str(p))
        return p

    if "vof" in save:
        emit(write_vof(out / f"{name}_unit.vof", system, basis[first]))
    if "vavg" in save:
        emit(write_vavg(out / f"{name}_unit.vavg", vavg))
    if "raw" in save:
        emit(write_raw(out / f"{name}_V.raw", vavg))
        emit(write_raw(out / f"{name}_Jmag.raw", Jmag))
    if "npz" in save:
        np.savez_compressed(out / f"{name}_fields.npz", node_v=grid, vavg=vavg,
                            Ex=Ex, Ey=Ey, Ez=Ez, Emag=Emag, Jmag=Jmag,
                            labels=model.labels, dx=model.dx)
        produced.append(str(out / f"{name}_fields.npz"))
    if "legacy_model" in save:
        ip, mp = model.save_legacy(out, name)
        emit(ip); emit(mp)

    if "png" in save:
        for sl in ocfg.get("slices", [{"axis": "z", "index": None}]):
            ax_, ix = sl.get("axis", "z"), sl.get("index")
            tag = f"{ax_}{ix if ix is not None else 'mid'}"
            emit(viz.save_labels_slice(out / f"{name}_labels_{tag}.png",
                                       model.labels, ax_, ix))
            emit(viz.save_slice(out / f"{name}_V_{tag}.png", vavg, ax_, ix,
                                title=f"potential (unit current), {tag}",
                                units="V"))
            emit(viz.save_slice(out / f"{name}_Emag_{tag}.png", Emag, ax_, ix,
                                title=f"|E| (unit current), {tag}",
                                cmap="magma", log=True, units="V/m"))
        for wname, wf in waveforms.items():
            emit(viz.save_waveform(out / f"{name}_wave_{wname}.png", wf, wname))

    # ---- ROI dose metrics (per unit current) ----
    masks = build_masks(cfg, model)
    manifest["roi_metrics_per_unit_A"] = [
        roi_metrics(m, Emag, Jmag, name=nm) for nm, m in masks.items()]

    # ---- coupling ----
    ncfg = cfg.get("neuron")
    if ncfg:
        coords = read_coordinates(_resolve(cfg, ncfg["coordinates"]))
        coords = transform_coordinates(
            coords, translate=ncfg.get("translate", (0, 0, 0)),
            rotate_deg=ncfg.get("rotate_deg"), scale=float(ncfg.get("scale", 1.0)))
        manifest["coupling"] = interpolation_report(grid.shape, coords)
        unit_fields = {src: sample_node_grid(node_grid(system, b), coords)
                       for src, b in basis.items()}
        for src, uf in unit_fields.items():
            emit(write_unit_field(out / f"{name}_{src}_unitfield.v", uf))
        # waveform-resolved extracellular matrix (mV by default, like the hoc flow)
        wf_map = {}
        for s in model.sources:
            if s.name in waveforms:
                wf_map[s.name] = waveforms[s.name]
        if wf_map:
            vmat = superpose_v_matrix(
                {k: unit_fields[k] for k in wf_map}, wf_map,
                unit_scale=float(ncfg.get("unit_scale", 1e3)))  # V -> mV
            emit(write_v_file(out / f"{name}_vext.v", vmat))
            manifest["coupling"]["v_matrix_shape"] = list(vmat.shape)

    manifest["produced"] = produced
    manifest["seconds_total"] = time.time() - t_start
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2,
                                                      default=str))
    progress(f"[neuroam] done in {manifest['seconds_total']:.1f}s -> {out}")
    return manifest
