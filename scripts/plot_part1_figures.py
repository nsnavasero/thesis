from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from bosonic_dissipation.part1_runner import DEFAULT_BUNDLE_NAME, get_bundle_directories, prepare_output_directories
from bosonic_dissipation.plotting import generate_figures_from_csvs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate only figures from existing Part 1 CSV files.")
    parser.add_argument("--bundle-name", default=DEFAULT_BUNDLE_NAME)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dirs = get_bundle_directories(PROJECT_ROOT, args.bundle_name)
    prepare_output_directories(
        output_dirs=output_dirs,
        clear_csv=False,
        clear_figures=True,
        clear_benchmark=False,
    )
    generate_figures_from_csvs(csv_dir=output_dirs["csv"], figure_dir=output_dirs["figures"])


if __name__ == "__main__":
    main()
