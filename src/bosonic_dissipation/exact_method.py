from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .io_utils import save_method_output_csv


@dataclass(slots=True)
class ExactMethodResult:
    method_name: str
    initial_state_type: str
    num_of_particles: float
    gamma: float
    total_time: float
    dt: float
    num_of_samples: int
    backend: str
    seed: int | None
    setup_runtime_seconds: float
    solve_runtime_seconds: float
    postprocess_runtime_seconds: float
    total_runtime_seconds: float
    solver_peak_python_memory_mib: float | None
    time_values: np.ndarray
    mean_particle_number: np.ndarray
    variance: np.ndarray


def _validate_initial_state(initial_state_type: str) -> str:
    normalized = initial_state_type.lower()
    if normalized not in {"fock", "coherent"}:
        raise ValueError("initial_state_type must be either 'fock' or 'coherent'.")
    return normalized


def simulate_exact_method(
    *,
    initial_state_type: str,
    num_of_particles: float,
    gamma: float,
    time: float,
    dt: float,
    num_of_samples: int = 1,
) -> ExactMethodResult:
    initial_state_type = _validate_initial_state(initial_state_type)

    if dt <= 0:
        raise ValueError("dt must be positive.")
    if time <= 0:
        raise ValueError("time must be positive.")
    if gamma < 0:
        raise ValueError("gamma must be non-negative.")

    total_start = perf_counter()
    setup_start = perf_counter()
    time_values = np.arange(0.0, time + dt, dt, dtype=np.float64)
    setup_runtime_seconds = perf_counter() - setup_start

    solve_start = perf_counter()
    mean_values = num_of_particles * np.exp(-gamma * time_values)
    if initial_state_type == "fock":
        variance = num_of_particles * (np.exp(-gamma * time_values) - np.exp(-2.0 * gamma * time_values))
    else:
        variance = mean_values.copy()
    solve_runtime_seconds = perf_counter() - solve_start

    postprocess_start = perf_counter()
    mean_values = np.asarray(mean_values, dtype=np.float64)
    variance = np.asarray(variance, dtype=np.float64)
    postprocess_runtime_seconds = perf_counter() - postprocess_start
    total_runtime_seconds = perf_counter() - total_start
    return ExactMethodResult(
        method_name="exact",
        initial_state_type=initial_state_type,
        num_of_particles=num_of_particles,
        gamma=gamma,
        total_time=time,
        dt=dt,
        num_of_samples=num_of_samples,
        backend="cpu",
        seed=None,
        setup_runtime_seconds=setup_runtime_seconds,
        solve_runtime_seconds=solve_runtime_seconds,
        postprocess_runtime_seconds=postprocess_runtime_seconds,
        total_runtime_seconds=total_runtime_seconds,
        solver_peak_python_memory_mib=None,
        time_values=time_values,
        mean_particle_number=mean_values,
        variance=variance,
    )


def run_exact_and_save(
    output_dir: str,
    *,
    initial_state_type: str,
    num_of_particles: float,
    gamma: float,
    time: float,
    dt: float,
    num_of_samples: int = 1,
):
    result = simulate_exact_method(
        initial_state_type=initial_state_type,
        num_of_particles=num_of_particles,
        gamma=gamma,
        time=time,
        dt=dt,
        num_of_samples=num_of_samples,
    )
    output_path = save_method_output_csv(
        output_dir,
        method_name=result.method_name,
        initial_state_type=result.initial_state_type,
        num_of_particles=result.num_of_particles,
        gamma=result.gamma,
        time=result.total_time,
        dt=result.dt,
        num_of_samples=result.num_of_samples,
        time_values=result.time_values,
        mean_values=result.mean_particle_number,
        variance_values=result.variance,
    )
    return result, output_path
