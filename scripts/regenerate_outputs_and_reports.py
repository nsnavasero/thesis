from __future__ import annotations

import shutil
import sys
import tracemalloc
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from bosonic_dissipation.density_matrix_method import run_density_matrix_and_save
from bosonic_dissipation.exact_method import run_exact_and_save
from bosonic_dissipation.monte_carlo_method import run_monte_carlo_and_save
from bosonic_dissipation.positive_p_method import run_positive_p_and_save


OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = PROJECT_ROOT / "figures"
WARMUP_METHODS = {"exact", "densityMatrix", "positiveP"}


def sanitize_number(value: float | int) -> str:
    numeric_value = float(value)
    if abs(numeric_value - round(numeric_value)) <= 1e-12:
        return str(int(round(numeric_value)))
    return f"{numeric_value:.15g}".replace(".", "p")


def prepare_output_directories() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    FIGURE_DIR.mkdir(exist_ok=True)
    for directory in (OUTPUT_DIR, FIGURE_DIR):
        for item in directory.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()


def run_method(method_name: str, kwargs: dict):
    if method_name == "exact":
        return run_exact_and_save(OUTPUT_DIR, **kwargs)
    if method_name == "densityMatrix":
        return run_density_matrix_and_save(OUTPUT_DIR, **kwargs)
    if method_name == "monteCarlo":
        return run_monte_carlo_and_save(OUTPUT_DIR, **kwargs)
    if method_name == "positiveP":
        return run_positive_p_and_save(OUTPUT_DIR, **kwargs)
    raise ValueError(f"Unknown method: {method_name}")


def should_warm_up(method_name: str) -> bool:
    return method_name in WARMUP_METHODS


def generate_results():
    shared = {
        "gamma": 1.0,
        "time": 5.0,
        "dt": 1e-3,
    }

    state_configs = {
        "fock": {"initial_state_type": "fock", "num_of_particles": 1},
        "coherent": {"initial_state_type": "coherent", "num_of_particles": 1},
    }

    method_specs = [
        ("exact", {"num_of_samples": 1, "prefer_gpu": True}),
        ("densityMatrix", {"num_of_samples": 1}),
        ("monteCarlo", {"num_of_samples": 1000, "interaction_strength": 0.0, "seed": 123}),
        ("positiveP", {"num_of_samples": 1000, "interaction_strength": 0.0, "prefer_gpu": True, "seed": 123}),
    ]

    results_by_state = {}
    benchmark_entries = {}

    for state_name, state_cfg in state_configs.items():
        results_by_state[state_name] = {}
        benchmark_entries[state_name] = []
        for method_name, method_cfg in method_specs:
            kwargs = {**state_cfg, **shared, **method_cfg}
            if should_warm_up(method_name):
                warmup_kwargs = kwargs.copy()
                if method_name == "positiveP":
                    warmup_kwargs["prefer_gpu"] = False
                print(f"Warming up {method_name} ({state_name})")
                run_method(method_name, warmup_kwargs)
            tracemalloc.start()
            wall_start = perf_counter()
            result, output_path = run_method(method_name, kwargs)
            _, peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            wall_runtime = perf_counter() - wall_start
            results_by_state[state_name][method_name] = (result, output_path)
            benchmark_entries[state_name].append(
                {
                    "method_name": method_name,
                    "result": result,
                    "output_path": output_path,
                    "wall_runtime": wall_runtime,
                    "solver_runtime": result.runtime_seconds,
                    "peak_mem_mib": peak_mem / (1024 * 1024),
                    "csv_size_kib": output_path.stat().st_size / 1024,
                    "output_points": int(len(result.time_values)),
                    "params": kwargs,
                }
            )
            print(f"Generated {output_path.name}")

    return shared, state_configs, results_by_state, benchmark_entries


def configure_plot_style() -> None:
    try:
        import scienceplots  # noqa: F401

        plt.style.use(["science", "no-latex"])
    except ImportError:
        pass

    plt.rcParams["text.usetex"] = False
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
        }
    )


def generate_figures(shared: dict, state_configs: dict, results_by_state: dict) -> None:
    configure_plot_style()

    style_map = {
        "exact": {"color": "black", "linestyle": "-", "linewidth": 2.3, "marker": None, "zorder": 5},
        "densityMatrix": {
            "color": "tab:blue",
            "linestyle": "--",
            "linewidth": 2.0,
            "marker": "^",
            "markevery": 400,
            "markersize": 3.6,
            "zorder": 4,
        },
        "monteCarlo": {
            "color": "tab:orange",
            "linestyle": "-.",
            "linewidth": 1.9,
            "marker": "o",
            "markevery": 350,
            "markersize": 3.5,
            "zorder": 3,
        },
        "positiveP": {
            "color": "tab:green",
            "linestyle": ":",
            "linewidth": 2.2,
            "marker": "s",
            "markevery": 450,
            "markersize": 3.2,
            "zorder": 2,
        },
    }
    default_style = {"linewidth": 1.8, "linestyle": "-", "marker": None, "zorder": 1}
    display_name_map = {
        "exact": "Exact",
        "densityMatrix": "Density Matrix",
        "monteCarlo": "Monte Carlo",
        "positiveP": "Positive-P",
    }
    plot_order = ["exact", "densityMatrix", "monteCarlo", "positiveP"]

    def get_style(method_name: str) -> dict:
        style = default_style.copy()
        style.update(style_map.get(method_name, {}))
        return style

    def get_difference_style(method_name: str) -> dict:
        style = get_style(method_name)
        style["marker"] = None
        style["linestyle"] = "-"
        style.pop("markevery", None)
        style.pop("markersize", None)
        return style

    for state_name, method_map in results_by_state.items():
        suffix = "_".join(
            [
                f"numOfParticles{sanitize_number(state_configs[state_name]['num_of_particles'])}",
                f"gamma{sanitize_number(shared['gamma'])}",
                f"time{sanitize_number(shared['time'])}",
                f"dt{sanitize_number(shared['dt'])}",
            ]
        )
        if state_name == "fock":
            state_title_text = rf"Fock state $|{state_configs[state_name]['num_of_particles']}\rangle$"
        else:
            alpha_text = f"{state_configs[state_name]['num_of_particles'] ** 0.5:g}"
            state_title_text = rf"Coherent state $|\alpha={alpha_text}\rangle$"
        gamma_text = sanitize_number(shared["gamma"]).replace("p", ".")

        frames = {method_name: pd.read_csv(output_path) for method_name, (_, output_path) in method_map.items()}
        shared_end_time = min(frame["time"].max() for frame in frames.values())
        cropped_frames = {name: frame[frame["time"] <= shared_end_time].copy() for name, frame in frames.items()}

        mean_fig, mean_ax = plt.subplots(figsize=(7.5, 5.2))
        for method_name in plot_order:
            frame = cropped_frames[method_name]
            mean_ax.plot(
                frame["time"],
                frame["mean_particle_number"],
                label=display_name_map[method_name],
                **get_style(method_name),
            )
        mean_ax.set_xlabel("Time")
        mean_ax.set_ylabel("Mean Particle Number")
        mean_ax.set_title(rf"Mean vs Time ({state_title_text}, $\gamma={gamma_text}$)")
        mean_ax.legend()
        mean_ax.grid(True, alpha=0.35)
        mean_fig.savefig(FIGURE_DIR / f"{state_name}_mean_{suffix}.svg", format="svg", bbox_inches="tight")
        plt.close(mean_fig)

        variance_fig, variance_ax = plt.subplots(figsize=(7.5, 5.2))
        for method_name in plot_order:
            frame = cropped_frames[method_name]
            variance_ax.plot(frame["time"], frame["variance"], label=display_name_map[method_name], **get_style(method_name))
        variance_ax.set_xlabel("Time")
        variance_ax.set_ylabel("Variance")
        variance_ax.set_title(rf"Variance vs Time ({state_title_text}, $\gamma={gamma_text}$)")
        variance_ax.legend()
        variance_ax.grid(True, alpha=0.35)
        variance_fig.savefig(FIGURE_DIR / f"{state_name}_variance_{suffix}.svg", format="svg", bbox_inches="tight")
        plt.close(variance_fig)

        exact_frame = cropped_frames["exact"][["time", "mean_particle_number", "variance"]].reset_index(drop=True)

        mean_diff_fig, mean_diff_ax = plt.subplots(figsize=(7.5, 5.2))
        for method_name in plot_order:
            if method_name == "exact":
                continue
            frame = cropped_frames[method_name]
            mean_difference = frame["mean_particle_number"].to_numpy() - exact_frame["mean_particle_number"].to_numpy()
            mean_diff_ax.plot(frame["time"], mean_difference, label=display_name_map[method_name], **get_difference_style(method_name))
        mean_diff_ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
        mean_diff_ax.set_xlabel("Time")
        mean_diff_ax.set_ylabel("Mean difference from exact")
        mean_diff_ax.set_title(rf"Mean Difference vs Time ({state_title_text}, $\gamma={gamma_text}$)")
        mean_diff_ax.legend()
        mean_diff_ax.grid(True, alpha=0.35)
        mean_diff_fig.savefig(FIGURE_DIR / f"{state_name}_meanDifference_{suffix}.svg", format="svg", bbox_inches="tight")
        plt.close(mean_diff_fig)

        variance_diff_fig, variance_diff_ax = plt.subplots(figsize=(7.5, 5.2))
        for method_name in plot_order:
            if method_name == "exact":
                continue
            frame = cropped_frames[method_name]
            variance_difference = frame["variance"].to_numpy() - exact_frame["variance"].to_numpy()
            variance_diff_ax.plot(frame["time"], variance_difference, label=display_name_map[method_name], **get_difference_style(method_name))
        variance_diff_ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
        variance_diff_ax.set_xlabel("Time")
        variance_diff_ax.set_ylabel("Variance difference from exact")
        variance_diff_ax.set_title(rf"Variance Difference vs Time ({state_title_text}, $\gamma={gamma_text}$)")
        variance_diff_ax.legend()
        variance_diff_ax.grid(True, alpha=0.35)
        variance_diff_fig.savefig(FIGURE_DIR / f"{state_name}_varDifference_{suffix}.svg", format="svg", bbox_inches="tight")
        plt.close(variance_diff_fig)


def build_detailed_report(shared: dict, state_configs: dict, benchmark_entries: dict) -> str:
    lines: list[str] = []
    lines.append("Single-Site Bosonic Dissipation Benchmark Report")
    lines.append("")
    lines.append("Shared comparison settings:")
    lines.append(f"- gamma = {shared['gamma']}")
    lines.append(f"- time = {shared['time']}")
    lines.append(f"- dt = {shared['dt']}")
    lines.append(f"- number of time points = {int(shared['time'] / shared['dt']) + 1}")
    lines.append(f"- warm-up enabled for methods = {', '.join(sorted(WARMUP_METHODS))}")
    lines.append("")

    for state_name in ["fock", "coherent"]:
        entries = benchmark_entries[state_name]
        cfg = state_configs[state_name]
        lines.append(f"=== {state_name.upper()} STATE ===")
        lines.append(f"- initial_state_type = {cfg['initial_state_type']}")
        lines.append(f"- num_of_particles = {cfg['num_of_particles']}")
        lines.append("- default num_of_samples for stochastic methods = 1000")
        lines.append("")

        for entry in entries:
            result = entry["result"]
            params = entry["params"]
            lines.append(f"Method: {entry['method_name']}")
            lines.append(
                "- parameters used: "
                f"initial_state_type={params['initial_state_type']}, "
                f"num_of_particles={params['num_of_particles']}, "
                f"gamma={params['gamma']}, "
                f"time={params['time']}, "
                f"dt={params['dt']}, "
                f"num_of_samples={params['num_of_samples']}"
            )
            if "interaction_strength" in params:
                lines.append(f"- interaction_strength = {params['interaction_strength']}")
            if hasattr(result, "hilbert_size"):
                lines.append(f"- hilbert_size = {result.hilbert_size}")
            if getattr(result, "coherent_alpha", None) is not None:
                lines.append(f"- coherent_alpha = {result.coherent_alpha}")
            lines.append(f"- backend used = {result.backend}")
            lines.append(f"- runtime reported by method = {result.runtime_seconds:.6f} s")
            lines.append(f"- wall-clock runtime = {entry['wall_runtime']:.6f} s")
            lines.append(f"- peak Python-allocated memory tracked = {entry['peak_mem_mib']:.3f} MiB")
            lines.append(f"- output points = {entry['output_points']}")
            lines.append(f"- CSV size = {entry['csv_size_kib']:.3f} KiB")
            lines.append(f"- CSV path = {entry['output_path']}")
            if hasattr(result, "fast_path_used"):
                lines.append(f"- fast_path_used = {result.fast_path_used}")
            if hasattr(result, "notes"):
                lines.append(f"- notes = {result.notes}")
            lines.append("")

        fastest = min(entries, key=lambda item: item["solver_runtime"])
        slowest = max(entries, key=lambda item: item["solver_runtime"])
        lowest_memory = min(entries, key=lambda item: item["peak_mem_mib"])
        highest_memory = max(entries, key=lambda item: item["peak_mem_mib"])
        lines.append("Quick comparison notes:")
        lines.append(f"- fastest solver here = {fastest['method_name']} ({fastest['solver_runtime']:.6f} s)")
        lines.append(f"- slowest solver here = {slowest['method_name']} ({slowest['solver_runtime']:.6f} s)")
        lines.append(
            f"- lowest tracked Python memory = {lowest_memory['method_name']} ({lowest_memory['peak_mem_mib']:.3f} MiB)"
        )
        lines.append(
            f"- highest tracked Python memory = {highest_memory['method_name']} ({highest_memory['peak_mem_mib']:.3f} MiB)"
        )
        lines.append("")

    return "\n".join(lines)

def method_pro_and_con(method_name: str, entry: dict) -> tuple[str, str]:
    result = entry["result"]
    if method_name == "exact":
        return ("reference baseline", "special-case analytic benchmark")
    if method_name == "densityMatrix":
        return ("deterministic, no sampling noise", "memory grows with Hilbert space")
    if method_name == "monteCarlo":
        return ("trajectory-based dissipation", "slow because many trajectories are needed")
    if method_name == "positiveP":
        if getattr(result, "fast_path_used", False):
            return ("very fast here", "performance is regime-dependent")
        return ("fast stochastic method", "interpretation needs care")
    raise ValueError(f"Unknown method: {method_name}")


def build_thesis_summary_report(shared: dict, state_configs: dict, benchmark_entries: dict) -> str:
    lines: list[str] = []
    lines.append("Thesis Benchmark Summary")
    lines.append("")
    lines.append("Scope")
    lines.append("- single site")
    lines.append("- dissipation only")
    lines.append(f"- gamma = {shared['gamma']}")
    lines.append(f"- time = {shared['time']}")
    lines.append(f"- dt = {shared['dt']}")
    lines.append("- stochastic methods: 1000 samples")
    lines.append(f"- warm-up used for: {', '.join(sorted(WARMUP_METHODS))}")
    lines.append("")

    overall_entries = [entry for entries in benchmark_entries.values() for entry in entries]
    slowest_overall = max(overall_entries, key=lambda item: item["solver_runtime"])
    fastest_overall = min(overall_entries, key=lambda item: item["solver_runtime"])
    lines.append("Solver Ranking")
    lines.append(f"- fastest solver: {fastest_overall['method_name']} ({fastest_overall['solver_runtime']:.6f} s)")
    lines.append(f"- slowest solver: {slowest_overall['method_name']} ({slowest_overall['solver_runtime']:.6f} s)")
    lines.append("")

    for method_name in ["exact", "densityMatrix", "monteCarlo", "positiveP"]:
        fock_entry = next(entry for entry in benchmark_entries["fock"] if entry["method_name"] == method_name)
        coherent_entry = next(entry for entry in benchmark_entries["coherent"] if entry["method_name"] == method_name)
        pro_label, con_label = method_pro_and_con(method_name, coherent_entry)
        lines.append(f"Method: {method_name}")
        lines.append(f"- pro: {pro_label}")
        lines.append(f"- con: {con_label}")
        lines.append(
            f"- fock: solve {fock_entry['solver_runtime']:.6f} s | wall {fock_entry['wall_runtime']:.3f} s | mem {fock_entry['peak_mem_mib']:.3f} MiB"
        )
        lines.append(
            f"- coherent: solve {coherent_entry['solver_runtime']:.6f} s | wall {coherent_entry['wall_runtime']:.3f} s | mem {coherent_entry['peak_mem_mib']:.3f} MiB"
        )
        if hasattr(fock_entry["result"], "hilbert_size"):
            lines.append(f"- Hilbert size: fock {fock_entry['result'].hilbert_size} | coherent {coherent_entry['result'].hilbert_size}")
        if hasattr(coherent_entry["result"], "hilbert_size"):
            pass
        if method_name == "positiveP" and getattr(coherent_entry["result"], "fast_path_used", False):
            lines.append("- note: coherent run used deterministic fast path")
        lines.append("")

    return "\n".join(lines)


def write_reports(shared: dict, state_configs: dict, benchmark_entries: dict) -> None:
    detailed_report = build_detailed_report(shared, state_configs, benchmark_entries)
    thesis_summary_report = build_thesis_summary_report(shared, state_configs, benchmark_entries)
    (OUTPUT_DIR / "benchmark_report.txt").write_text(detailed_report, encoding="utf-8")
    (OUTPUT_DIR / "benchmark_thesis_summary.txt").write_text(thesis_summary_report, encoding="utf-8")
    print("Wrote benchmark_report.txt")
    print("Wrote benchmark_thesis_summary.txt")


def main() -> None:
    prepare_output_directories()
    shared, state_configs, results_by_state, benchmark_entries = generate_results()
    generate_figures(shared, state_configs, results_by_state)
    write_reports(shared, state_configs, benchmark_entries)


if __name__ == "__main__":
    main()
