import argparse
from pathlib import Path

from .runner import main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run thesis diffusion analysis pipeline.")
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("Data"),
        help="Path to data root. If it contains rumor_detection_acl2017/ , that subdir is used automatically.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("thesis_outputs"),
        help="Directory for generated outputs (tables/figures/logs/captions).",
    )
    return parser.parse_args()


def run_from_cli() -> None:
    args = parse_args()
    main(data_dir=args.data_dir, out_dir=args.out_dir)
