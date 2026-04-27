from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scienceplots


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

METHOD_ORDER = ["exact", "densityMatrix", "monteCarlo", "positiveP"]
STATE_ORDER = ["fock", "coherent"]
METHOD_LABELS = {
    ("exact", "fock"): "Exact Fock",
    ("exact", "coherent"): "Exact Coherent",
    ("densityMatrix", "fock"): "Density Matrix Fock",
    ("densityMatrix", "coherent"): "Density Matrix Coherent",
    ("monteCarlo", "fock"): "Monte Carlo Fock",
    ("monteCarlo", "coherent"): "Monte Carlo Coherent",
    ("positiveP", "fock"): "Positive-P Fock",
    ("positiveP", "coherent"): "Positive-P Coherent",
}
COLOR_MAP = {
    ("exact", "fock"): "#111111",
    ("exact", "coherent"): "#555555",
    ("densityMatrix", "fock"): "#1f77b4",
    ("densityMatrix", "coherent"): "#6baed6",
    ("monteCarlo", "fock"): "#ff7f0e",
    ("monteCarlo", "coherent"): "#fdae6b",
    ("positiveP", "fock"): "#2ca02c",
    ("positiveP", "coherent"): "#74c476",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate benchmark runtime and memory bar graphs from Part 1 benchmark text files."
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/benchmark bar graphs no 1000",
        help="Directory where the new benchmark figures will be written.",
    )
    parser.add_argument(
        "--particle-counts",
        nargs="+",
        type=int,
        default=[1, 10, 100],
        help="Particle-count groups to include in the charts.",
    )
    return parser.parse_args()


def parse_benchmark_file(path: Path) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("Method:"):
            parsed["method"] = line.split(":", 1)[1].strip()
            continue
        if not line.startswith("- ") or "=" not in line:
            continue
        key, value = line[2:].split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def parse_memory_mib(value: str | None) -> float | None:
    if value is None or value == "not tracked":
        return None
    number_text = value.replace("MiB", "").strip()
    return float(number_text)


def collect_benchmarks(outputs_dir: Path) -> dict[tuple[int, int, int], dict[tuple[str, str], dict[str, float | None]]]:
    bundles = sorted(outputs_dir.glob("*part1*method_comparison"))
    benchmark_data: dict[tuple[int, int, int], dict[tuple[str, str], dict[str, float | None]]] = {}
    for bundle_dir in bundles:
        benchmark_dir = bundle_dir / "benchmark"
        if not benchmark_dir.is_dir():
            continue
        for benchmark_file in sorted(benchmark_dir.glob("*.txt")):
            parsed = parse_benchmark_file(benchmark_file)
            try:
                gamma = int(float(parsed["gamma"]))
                interaction_strength = int(float(parsed["interaction_strength"]))
                num_particles = int(float(parsed["num_of_particles"]))
                method = str(parsed["method"])
                initial_state = str(parsed["initial_state_type"])
            except KeyError:
                continue

            benchmark_data.setdefault((gamma, interaction_strength, num_particles), {})[(method, initial_state)] = {
                "runtime_seconds": float(parsed["solve_runtime_seconds"]),
                "memory_mib": parse_memory_mib(parsed.get("peak_process_memory")),
            }
    return benchmark_data


def write_checked_values_table(
    *,
    benchmark_data: dict[tuple[int, int, int], dict[tuple[str, str], dict[str, float | None]]],
    output_dir: Path,
    particle_counts: list[int],
) -> Path:
    output_path = output_dir / "checked_benchmark_values_no_1000.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "gamma",
                "interaction_strength",
                "num_particles",
                "method",
                "initial_state",
                "solve_runtime_seconds",
                "peak_process_memory_mib",
            ]
        )
        for gamma in [0, 1]:
            for interaction_strength in [0, 1]:
                for num_particles in particle_counts:
                    for method in METHOD_ORDER:
                        for state in STATE_ORDER:
                            entry = benchmark_data.get((gamma, interaction_strength, num_particles), {}).get(
                                (method, state),
                                {},
                            )
                            writer.writerow(
                                [
                                    gamma,
                                    interaction_strength,
                                    num_particles,
                                    method,
                                    state,
                                    entry.get("runtime_seconds"),
                                    entry.get("memory_mib"),
                                ]
                            )
    return output_path


def format_runtime_label(value: float) -> str:
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}"
    if value >= 1:
        return f"{value:.2f}"
    if value >= 1e-2:
        return f"{value:.3f}"
    return f"{value:.1e}"


def format_memory_label(value: float) -> str:
    return f"{value:.0f}"


def configure_benchmark_plot_style() -> None:
    plt.style.use(["science", "no-latex"])
    plt.rcParams["text.usetex"] = False
    plt.rcParams.update(
        {
            "font.size": 20,
            "axes.titlesize": 42,
            "axes.labelsize": 38,
            "xtick.labelsize": 34,
            "ytick.labelsize": 34,
            "legend.fontsize": 28,
        }
    )


def plot_metric(
    *,
    benchmark_data: dict[tuple[int, int, int], dict[tuple[str, str], dict[str, float | None]]],
    output_dir: Path,
    gamma: int,
    interaction_strength: int,
    particle_counts: list[int],
    metric_key: str,
    y_label: str,
    title_prefix: str,
    output_prefix: str,
    log_scale: bool,
    formatter,
) -> Path:
    configure_benchmark_plot_style()
    fig, ax = plt.subplots(figsize=(30.8, 16.8))
    ax.tick_params(axis="both", labelsize=34)

    group_centers = np.arange(len(particle_counts), dtype=float) * 3.1
    bar_width = 0.30
    series_keys = [(method, state) for method in METHOD_ORDER for state in STATE_ORDER]
    offsets = (np.arange(len(series_keys)) - (len(series_keys) - 1) / 2.0) * bar_width

    for offset, series_key in zip(offsets, series_keys):
        values = []
        for num_particles in particle_counts:
            entry = benchmark_data.get((gamma, interaction_strength, num_particles), {}).get(series_key, {})
            metric_value = entry.get(metric_key)
            values.append(np.nan if metric_value is None else float(metric_value))
        rects = ax.bar(
            group_centers + offset,
            values,
            width=bar_width * 0.99,
            label=METHOD_LABELS[series_key],
            color=COLOR_MAP[series_key],
            edgecolor="none",
        )

    ax.set_xticks(group_centers)
    ax.set_xticklabels([str(value) for value in particle_counts])
    ax.set_xlabel("Number of Particles", fontsize=38)
    ax.set_ylabel(y_label, fontsize=38)
    ax.set_title(f"{title_prefix} ($\\gamma={gamma}$, $U={interaction_strength}$)", fontsize=42)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    if log_scale:
        ax.set_yscale("log")
        finite_values = [
            float(entry[metric_key])
            for bundle in benchmark_data.values()
            for entry in bundle.values()
            if entry.get(metric_key) is not None and float(entry[metric_key]) > 0.0
        ]
        if finite_values:
            lower = min(finite_values) / 2.5
            upper = max(finite_values) * 3.0
            ax.set_ylim(lower, upper)
    else:
        visible_values = []
        for num_particles in particle_counts:
            for series_key in series_keys:
                entry = benchmark_data.get((gamma, interaction_strength, num_particles), {}).get(series_key, {})
                metric_value = entry.get(metric_key)
                if metric_value is not None:
                    visible_values.append(float(metric_value))
        upper = max(visible_values) * 1.18 if visible_values else 1.0
        ax.set_ylim(0.0, upper)

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True, fontsize=28)
    fig.tight_layout()

    output_path = output_dir / f"{output_prefix}_gamma{gamma}_u{interaction_strength}.svg"
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_data = collect_benchmarks(PROJECT_ROOT / "outputs")
    particle_counts = sorted(set(args.particle_counts))
    write_checked_values_table(
        benchmark_data=benchmark_data,
        output_dir=output_dir,
        particle_counts=particle_counts,
    )

    for gamma in [0, 1]:
        for interaction_strength in [0, 1]:
            plot_metric(
                benchmark_data=benchmark_data,
                output_dir=output_dir,
                gamma=gamma,
                interaction_strength=interaction_strength,
                particle_counts=particle_counts,
                metric_key="runtime_seconds",
                y_label="Solve Runtime (s)",
                title_prefix="Benchmark Runtime by Particle Number",
                output_prefix="runtime",
                log_scale=True,
                formatter=format_runtime_label,
            )
            plot_metric(
                benchmark_data=benchmark_data,
                output_dir=output_dir,
                gamma=gamma,
                interaction_strength=interaction_strength,
                particle_counts=particle_counts,
                metric_key="memory_mib",
                y_label="Peak Process Memory (MiB)",
                title_prefix="Benchmark Memory by Particle Number",
                output_prefix="memory",
                log_scale=False,
                formatter=format_memory_label,
            )


if __name__ == "__main__":
    main()
