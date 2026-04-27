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


def bundle_prefix(bundle_name: str) -> int | None:
    prefix = bundle_name.split("_", 1)[0]
    if prefix.isdigit():
        return int(prefix)
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replot a numbered range of Part 1 bundles.")
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--plot-profile", default="time", choices=["time", "t_over_gamma"])
    parser.add_argument(
        "--figure-subdir",
        default=None,
        help="Optional subfolder inside each bundle figures directory. Defaults to the plot profile name.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs_root = PROJECT_ROOT / "outputs"
    selected_bundles = []
    for bundle_dir in sorted(outputs_root.iterdir()):
        if not bundle_dir.is_dir():
            continue
        prefix = bundle_prefix(bundle_dir.name)
        if prefix is None or prefix < args.start or prefix > args.end:
            continue
        csv_dir = bundle_dir / "csv"
        if not csv_dir.exists():
            continue
        selected_bundles.append(bundle_dir.name)

    if not selected_bundles:
        raise FileNotFoundError(f"No bundle folders with CSV data found from {args.start:02d} to {args.end:02d}.")

    total_generated = 0
    for bundle_name in selected_bundles:
        output_dirs = get_bundle_directories(PROJECT_ROOT, bundle_name)
        figure_dir = output_dirs["figures"] / (args.figure_subdir or args.plot_profile)
        generated = generate_figures_from_csvs(
            csv_dir=output_dirs["csv"],
            figure_dir=figure_dir,
            plot_profile=args.plot_profile,
        )
        total_generated += len(generated)
        print(f"{bundle_name}: generated {len(generated)} figures in {figure_dir}")

    print(f"Finished replotting {len(selected_bundles)} bundle(s); generated {total_generated} total figures.")


if __name__ == "__main__":
    main()
