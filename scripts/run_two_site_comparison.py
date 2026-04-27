from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from bosonic_dissipation.two_site_comparison import run_two_site_comparison  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Permanent two-site Density-Matrix vs Positive-P comparison runner.")
    parser.add_argument("--interaction-strength", type=float, required=True)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--hopping", type=float, default=1.0)
    parser.add_argument("--time", type=float, default=5.0)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--num-samples", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=123456)
    parser.add_argument("--plot-profile", default="time", choices=["time", "t_over_gamma"])
    parser.add_argument(
        "--figure-subdir",
        default=None,
        help="Optional subfolder inside the permanent case figures directory. Defaults to the plot profile name.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_two_site_comparison(
        project_root=PROJECT_ROOT,
        interaction_strength=args.interaction_strength,
        gamma=args.gamma,
        hopping=args.hopping,
        total_time=args.time,
        dt=args.dt,
        num_samples=args.num_samples,
        seed=args.seed,
        plot_profile=args.plot_profile,
        figure_subdir=args.figure_subdir or args.plot_profile,
    )
    print(f"Case: {result['case_name']}")
    print(f"CSV: {result['csv_path']}")
    print(f"Figures: {len(result['figure_paths'])}")
    print(f"Positive-P stable through: {result['positive_p_stop_time']:.6f}")
    print(f"Max absolute mean error: {result['max_abs_error']:.6f}")


if __name__ == "__main__":
    main()
