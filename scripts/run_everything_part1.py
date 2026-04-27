from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from bosonic_dissipation.part1_runner import (
    DEFAULT_BUNDLE_NAME,
    DEFAULT_METHOD_NAMES,
    generate_csvs_and_benchmarks,
    get_bundle_directories,
    get_method_spec_map,
    get_shared_settings,
    get_state_configs,
    prepare_output_directories,
)
from bosonic_dissipation.plotting import generate_figures_from_csvs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Part 1 CSVs, benchmark files, and figures.")
    parser.add_argument("--bundle-name", default=DEFAULT_BUNDLE_NAME)
    parser.add_argument("--num-particles", type=float, default=1.0)
    parser.add_argument("--interaction-strength", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--time", type=float, default=5.0)
    parser.add_argument("--dt", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHOD_NAMES))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dirs = get_bundle_directories(PROJECT_ROOT, args.bundle_name)
    prepare_output_directories(
        output_dirs=output_dirs,
        clear_csv=True,
        clear_figures=True,
        clear_benchmark=True,
    )

    shared_settings = get_shared_settings(
        interaction_strength=args.interaction_strength,
        gamma=args.gamma,
        time=args.time,
        dt=args.dt,
    )
    state_configs = get_state_configs(num_of_particles=args.num_particles)
    method_spec_map = get_method_spec_map(seed=args.seed)

    generate_csvs_and_benchmarks(
        csv_dir=output_dirs["csv"],
        benchmark_dir=output_dirs["benchmark"],
        shared_settings=shared_settings,
        state_configs=state_configs,
        method_names=args.methods,
        method_spec_map=method_spec_map,
    )
    generate_figures_from_csvs(csv_dir=output_dirs["csv"], figure_dir=output_dirs["figures"])


if __name__ == "__main__":
    main()
