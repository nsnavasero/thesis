from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from bosonic_dissipation.plotting import generate_figures_from_csvs  # noqa: E402


def get_bundle_directories(project_root: Path, bundle_name: str) -> dict[str, Path]:
    part_output_dir = project_root / "outputs" / bundle_name
    return {
        "root": part_output_dir,
        "csv": part_output_dir / "csv",
        "figures": part_output_dir / "figures",
        "benchmark": part_output_dir / "benchmark",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replot an existing Part 1 bundle from CSV files only.")
    parser.add_argument("--bundle-name", required=True)
    parser.add_argument("--plot-profile", default="time", choices=["time", "t_over_gamma"])
    parser.add_argument(
        "--figure-subdir",
        default=None,
        help="Optional subfolder inside the bundle figures directory. Defaults to the plot profile name.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dirs = get_bundle_directories(PROJECT_ROOT, args.bundle_name)
    if not output_dirs["csv"].exists():
        raise FileNotFoundError(f"CSV directory not found: {output_dirs['csv']}")

    figure_dir = output_dirs["figures"] / (args.figure_subdir or args.plot_profile)
    generated = generate_figures_from_csvs(
        csv_dir=output_dirs["csv"],
        figure_dir=figure_dir,
        plot_profile=args.plot_profile,
    )
    print(f"Generated {len(generated)} figures in {figure_dir}")


if __name__ == "__main__":
    main()
