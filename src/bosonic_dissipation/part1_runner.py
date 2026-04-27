from __future__ import annotations

import shutil
import threading
from time import sleep
from pathlib import Path

import psutil

from .density_matrix_method import run_density_matrix_and_save
from .exact_method import run_exact_and_save
from .io_utils import build_output_stem
from .monte_carlo_method import run_monte_carlo_and_save
from .positive_p_method import run_positive_p_and_save


DEFAULT_BUNDLE_NAME = "01_baseline_u0_method_comparison"
DEFAULT_METHOD_NAMES = ("exact", "densityMatrix", "monteCarlo", "positiveP")
PROCESS_MEMORY_POLL_INTERVAL_SECONDS = 0.01


def sanitize_number(value: float | int) -> str:
    numeric_value = float(value)
    if abs(numeric_value - round(numeric_value)) <= 1e-12:
        return str(int(round(numeric_value)))
    return f"{numeric_value:.15g}".replace(".", "p")


def format_memory_value(memory_mib: float | None) -> str:
    if memory_mib is None:
        return "not tracked"
    return f"{memory_mib:.3f} MiB"


def get_bundle_directories(project_root: Path, bundle_name: str) -> dict[str, Path]:
    part_output_dir = project_root / "outputs" / bundle_name
    return {
        "root": part_output_dir,
        "csv": part_output_dir / "csv",
        "figures": part_output_dir / "figures",
        "benchmark": part_output_dir / "benchmark",
    }


def get_shared_settings(
    *,
    interaction_strength: float = 0.0,
    gamma: float = 1.0,
    time: float = 5.0,
    dt: float = 1e-3,
) -> dict:
    return {
        "interaction_strength": interaction_strength,
        "gamma": gamma,
        "time": time,
        "dt": dt,
    }


def get_state_configs(*, num_of_particles: float = 1.0) -> dict:
    return {
        "fock": {"initial_state_type": "fock", "num_of_particles": num_of_particles},
        "coherent": {"initial_state_type": "coherent", "num_of_particles": num_of_particles},
    }


def get_method_specs(*, monte_carlo_samples: int = 1000, positive_p_samples: int = 1000, seed: int = 0) -> list[tuple[str, dict]]:
    return [
        ("exact", {"num_of_samples": 1}),
        ("densityMatrix", {"num_of_samples": 1}),
        ("monteCarlo", {"num_of_samples": monte_carlo_samples, "seed": seed}),
        ("positiveP", {"num_of_samples": positive_p_samples, "seed": seed}),
    ]


def get_method_spec_map(*, monte_carlo_samples: int = 1000, positive_p_samples: int = 1000, seed: int = 0) -> dict[str, dict]:
    return dict(
        get_method_specs(
            monte_carlo_samples=monte_carlo_samples,
            positive_p_samples=positive_p_samples,
            seed=seed,
        )
    )


def validate_method_names(method_names: list[str] | tuple[str, ...], *, method_spec_map: dict[str, dict]) -> list[str]:
    unknown_methods = sorted(set(method_names) - set(method_spec_map))
    if unknown_methods:
        raise ValueError(f"Unknown methods requested: {', '.join(unknown_methods)}")
    return list(method_names)


def clear_directory(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for item in directory.iterdir():
        if item.is_dir():
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


def run_method(method_name: str, kwargs: dict, *, csv_dir: Path):
    if method_name == "exact":
        return run_exact_and_save(csv_dir, **kwargs)
    if method_name == "densityMatrix":
        return run_density_matrix_and_save(csv_dir, **kwargs)
    if method_name == "monteCarlo":
        return run_monte_carlo_and_save(csv_dir, **kwargs)
    if method_name == "positiveP":
        return run_positive_p_and_save(csv_dir, **kwargs)
    raise ValueError(f"Unknown method: {method_name}")


def measure_peak_process_memory_for_call(func, *args, **kwargs):
    process = psutil.Process()
    peak_rss_bytes = process.memory_info().rss
    stop_event = threading.Event()

    def sample_memory() -> None:
        nonlocal peak_rss_bytes
        while not stop_event.is_set():
            peak_rss_bytes = max(peak_rss_bytes, process.memory_info().rss)
            sleep(PROCESS_MEMORY_POLL_INTERVAL_SECONDS)
        peak_rss_bytes = max(peak_rss_bytes, process.memory_info().rss)

    sampler_thread = threading.Thread(target=sample_memory, daemon=True)
    sampler_thread.start()
    try:
        result = func(*args, **kwargs)
    finally:
        stop_event.set()
        sampler_thread.join()
    return result, peak_rss_bytes / (1024 * 1024)


def generate_csvs_and_benchmarks(
    *,
    csv_dir: Path,
    benchmark_dir: Path,
    shared_settings: dict,
    state_configs: dict,
    method_names: list[str] | tuple[str, ...] = DEFAULT_METHOD_NAMES,
    method_spec_map: dict[str, dict] | None = None,
) -> tuple[dict[str, dict[str, tuple[object, Path]]], list[dict]]:
    if method_spec_map is None:
        method_spec_map = get_method_spec_map()
    method_names = validate_method_names(method_names, method_spec_map=method_spec_map)

    results_by_state: dict[str, dict[str, tuple[object, Path]]] = {}
    benchmark_entries: list[dict] = []

    for state_name, state_cfg in state_configs.items():
        results_by_state[state_name] = {}
        for method_name in method_names:
            kwargs = {**state_cfg, **shared_settings, **method_spec_map[method_name]}
            (result, output_path), peak_process_memory_mib = measure_peak_process_memory_for_call(
                run_method,
                method_name,
                kwargs,
                csv_dir=csv_dir,
            )
            results_by_state[state_name][method_name] = (result, output_path)
            benchmark_entries.append(
                {
                    "method_name": method_name,
                    "state_name": state_name,
                    "result": result,
                    "output_path": output_path,
                    "csv_size_kib": output_path.stat().st_size / 1024,
                    "output_points": int(len(result.time_values)),
                    "peak_process_memory_mib": peak_process_memory_mib,
                    "params": kwargs,
                }
            )
            print(f"Generated {output_path.name}")
            write_benchmark_files([benchmark_entries[-1]], benchmark_dir=benchmark_dir)
    return results_by_state, benchmark_entries


def build_method_benchmark_text(entry: dict) -> str:
    result = entry["result"]
    params = entry["params"]

    lines: list[str] = []
    lines.append(f"Method: {entry['method_name']}")
    lines.append(f"- initial_state_type = {params['initial_state_type']}")
    lines.append(f"- num_of_particles = {params['num_of_particles']}")
    lines.append(f"- gamma = {params['gamma']}")
    lines.append(f"- time = {params['time']}")
    lines.append(f"- dt = {params['dt']}")
    lines.append(f"- num_of_samples = {params['num_of_samples']}")
    if "interaction_strength" in params:
        lines.append(f"- interaction_strength = {params['interaction_strength']}")
    lines.append(f"- seed = {result.seed if result.seed is not None else 'not used'}")
    if hasattr(result, "hilbert_size"):
        lines.append(f"- hilbert_size = {result.hilbert_size}")
    if getattr(result, "coherent_alpha", None) is not None:
        lines.append(f"- coherent_alpha = {result.coherent_alpha}")
    lines.append(f"- backend used = {result.backend}")
    lines.append(f"- setup_runtime_seconds = {result.setup_runtime_seconds:.6f}")
    lines.append(f"- solve_runtime_seconds = {result.solve_runtime_seconds:.6f}")
    lines.append(f"- postprocess_runtime_seconds = {result.postprocess_runtime_seconds:.6f}")
    lines.append(f"- total_runtime_seconds = {result.total_runtime_seconds:.6f}")
    lines.append(f"- solver_peak_python_memory = {format_memory_value(result.solver_peak_python_memory_mib)}")
    lines.append(f"- peak_process_memory = {format_memory_value(entry.get('peak_process_memory_mib'))}")
    lines.append(f"- output points = {entry['output_points']}")
    lines.append(f"- CSV size = {entry['csv_size_kib']:.3f} KiB")
    lines.append(f"- CSV path = {entry['output_path']}")
    if hasattr(result, "fast_path_used"):
        lines.append(f"- fast_path_used = {result.fast_path_used}")
    if hasattr(result, "notes"):
        lines.append(f"- notes = {result.notes}")
    return "\n".join(lines)


def write_benchmark_files(benchmark_entries: list[dict], *, benchmark_dir: Path) -> None:
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    for entry in benchmark_entries:
        result = entry["result"]
        params = entry["params"]
        stem = build_output_stem(
            method_name=entry["method_name"],
            initial_state_type=params["initial_state_type"],
            num_of_particles=params["num_of_particles"],
            interaction_strength=params.get("interaction_strength"),
            gamma=params["gamma"],
            time=params["time"],
            dt=params["dt"],
            num_of_samples=params["num_of_samples"],
            hilbert_size=getattr(result, "hilbert_size", None),
            seed=result.seed,
        )
        benchmark_path = benchmark_dir / f"{stem}.txt"
        benchmark_path.write_text(build_method_benchmark_text(entry), encoding="utf-8")
        print(f"Wrote {benchmark_path.name}")
