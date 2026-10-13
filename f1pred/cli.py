"""Ligne de commande : `python -m f1pred run`."""

import argparse
import os
from datetime import date
from pathlib import Path

from f1pred import pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="f1pred")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="Met à jour les données, les prédictions et le site")
    run.add_argument("--offline", action="store_true", help="N'appelle pas l'API")
    run.add_argument("--force", action="store_true", help="Recalcule même sans nouvelle manche")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--sims", type=int, default=10_000)
    run.add_argument("--data-dir", default="data")
    run.add_argument("--out", default="site")
    args = parser.parse_args(argv)

    outcome = pipeline.run(Path(args.data_dir), Path(args.out), offline=args.offline,
                           seed=args.seed, n_sims=args.sims, force=args.force,
                           today=date.today())
    print(outcome.message)
    gh_output = os.environ.get("GITHUB_OUTPUT")
    if gh_output:
        with open(gh_output, "a", encoding="utf-8") as fh:
            fh.write(f"updated={'true' if outcome.updated else 'false'}\n")
    return 0
