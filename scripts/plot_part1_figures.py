from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from bosonic_dissipation.plotting import generate_figures_from_csvs


DEFAULT_BUNDLE_NAME = "01_baseline_u0_method_comparison"


def get_bundle_directories(project_root: Path, bundle_name: str) -> dict[str, Path]:
    part_output_dir = project_root / "outputs" / bundle_name
    return {
        "root": part_output_dir,
        "csv": part_output_dir / "csv",
        "figures": part_output_dir / "figures",
        "benchmark": part_output_dir / "benchmark",
    }


def clear_directory(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for item in directory.iterdir():
        if item.is_dir():
            import shutil

            shutil.rmtree(item)
        else:
            item.unlink()


def prepare_output_directories(*, output_dirs: dict[str, Path], clear_csv: bool, clear_figures: bool, clear_benchmark: bool) -> None:
    output_dirs["root"].mkdir(parents=True, exist_ok=True)
    for key in ("csv", "figures", "benchmark"):
        output_dirs[key].mkdir(parents=True, exist_ok=True)
    if clear_csv:
        clear_directory(output_dirs["csv"])
    if clear_figures:
        clear_directory(output_dirs["figures"])
    if clear_benchmark:
        clear_directory(output_dirs["benchmark"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate only figures from existing Part 1 CSV files.")
    parser.add_argument("--bundle-name", default=DEFAULT_BUNDLE_NAME)
    parser.add_argument("--plot-profile", default="time", choices=["time", "t_over_gamma"])
    parser.add_argument(
        "--figure-subdir",
        default=None,
        help="Optional subfolder inside the bundle figures directory. Defaults to writing directly to figures/.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dirs = get_bundle_directories(PROJECT_ROOT, args.bundle_name)
    figure_dir = output_dirs["figures"] if args.figure_subdir is None else output_dirs["figures"] / args.figure_subdir
    prepare_output_directories(
        output_dirs=output_dirs,
        clear_csv=False,
        clear_figures=args.figure_subdir is None,
        clear_benchmark=False,
    )
    generate_figures_from_csvs(
        csv_dir=output_dirs["csv"],
        figure_dir=figure_dir,
        plot_profile=args.plot_profile,
    )


if __name__ == "__main__":
    main()
