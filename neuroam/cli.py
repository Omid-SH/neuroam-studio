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

    pv = sub.add_parser("validate", help="check a config without solving")
    pv.add_argument("config")

    args = p.parse_args(argv)

    if args.cmd == "run":
        from .pipeline import run
        run(args.config, out_dir=args.out)
        return 0

    if args.cmd == "info":
        from .model import VoxelModel
        m = VoxelModel.from_legacy(args.in_file)
        print(m.summary())
        return 0

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
