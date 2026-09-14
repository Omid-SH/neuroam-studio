"""Command line interface: ``python -m neuroam <cmd>``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="neuroam",
        description="NeuroAM Studio — config-driven AM + NEURON pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="run a pipeline config")
    pr.add_argument("config", help="path to JSON config")
    pr.add_argument("-o", "--out", default=None, help="output directory")

    pi = sub.add_parser("info", help="summarize a legacy .in/.model pair")
    pi.add_argument("in_file", help="path to legacy .in file")
    pi.add_argument("--view", action="store_true",
                    help="open the model in the interactive 3D viewer")
    pi.add_argument("--labels", nargs="+", type=int, default=None,
                    help="material labels to highlight in the 3D viewer")
    pi.add_argument("--opacity", type=float, default=0.10,
                    help="anatomy context opacity (default: 0.10)")
    pi.add_argument("--slice-axis", choices=("x", "y", "z"), default="z",
                    help="initial interactive tissue-slice axis (default: z)")
    pi.add_argument("--slice-index", type=int, default=None,
                    help="initial slice index (default: source node or midpoint)")
    pi.add_argument("--no-slice", action="store_true",
                    help="disable the interactive internal-tissue slice")
    pi.add_argument("--legend", action="store_true",
                    help="show the material-label legend inside the 3D window")

    pw = sub.add_parser("view", help="open an interactive 3D model viewer")
    pw.add_argument("input", help="legacy .in file or pipeline config .json")
    pw.add_argument("--model", default=None,
                    help="override the .model paired with a legacy .in file")
    pw.add_argument("--labels", nargs="+", type=int, default=None,
                    help="additional material labels to highlight")
    pw.add_argument("--opacity", type=float, default=0.10,
                    help="anatomy context opacity (default: 0.10)")
    pw.add_argument("--slice-axis", choices=("x", "y", "z"), default="z",
                    help="initial interactive tissue-slice axis (default: z)")
    pw.add_argument("--slice-index", type=int, default=None,
                    help="initial slice index (default: source node or midpoint)")
    pw.add_argument("--no-slice", action="store_true",
                    help="disable the interactive internal-tissue slice")
    pw.add_argument("--legend", action="store_true",
                    help="show the material-label legend inside the 3D window")
    pw.add_argument("--background", default="#20242B",
                    help="viewer background color")
    pw.add_argument("--screenshot", default=None,
                    help="save a PNG when the viewer closes")
    pw.add_argument("--off-screen", action="store_true",
                    help="render without opening a window (use with --screenshot)")

    pv = sub.add_parser("validate", help="check a config without solving")
    pv.add_argument("config")

    pe = sub.add_parser("electrodes",
                        help="design-check electrodes: register, QC, no solve")
    pe.add_argument("config")
    pe.add_argument("-o", "--out", default=None,
                    help="write <montage>.overlay.npz / .montage.json here")
    pe.add_argument("--legacy", metavar="DIR", default=None,
                    help="also bake anatomy+electrodes to .model/.in in DIR")

    args = p.parse_args(argv)

    if args.cmd == "run":
        from .pipeline import run
        run(args.config, out_dir=args.out)
        return 0

    if args.cmd == "info":
        from .model import VoxelModel
        m = VoxelModel.from_legacy(args.in_file)
        print(m.summary(), flush=True)
        if args.view:
            from .viewer import show_model
            try:
                show_model(m, labels=args.labels, opacity=args.opacity,
                           slice_axis=None if args.no_slice else args.slice_axis,
                           slice_index=args.slice_index,
                           show_legend=args.legend)
            except (RuntimeError, ValueError) as exc:
                print(f"neuroam: {exc}", file=sys.stderr)
                return 2
        return 0

    if args.cmd == "view":
        from .viewer import load_view_model, show_model
        try:
            m = load_view_model(args.input, model_path=args.model)
            print(m.summary(), flush=True)
            show_model(m, labels=args.labels, opacity=args.opacity,
                       background=args.background,
                       slice_axis=None if args.no_slice else args.slice_axis,
                       slice_index=args.slice_index, show_legend=args.legend,
                       screenshot=args.screenshot,
                       off_screen=args.off_screen)
        except (RuntimeError, ValueError, OSError) as exc:
            print(f"neuroam: {exc}", file=sys.stderr)
            return 2
        return 0

    if args.cmd == "electrodes":
        from .pipeline import load_config, build_model
        cfg = load_config(args.config)
        model = build_model(cfg)
        mont = getattr(model, "montage", None)
        if mont is None:
            print("config has no v0.2 'electrodes' block (geometry/from_label)")
            print(model.summary())
            return 1
        print(model.summary())
        print()
        print(mont.report())
        if args.out:
            print("wrote", mont.save(args.out))
        if args.legacy:
            in_p, model_p = mont.write_legacy(model, args.legacy)
            print("wrote", in_p, "and", model_p)
        return 2 if mont.warnings else 0

    if args.cmd == "validate":
        from .pipeline import load_config, build_model
        cfg = load_config(args.config)
        model = build_model(cfg)
        print("config OK")
        print(model.summary())
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
