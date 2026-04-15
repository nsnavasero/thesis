from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import numpy as np

from .io_utils import save_method_output_csv


ArrayBackend = Literal["cpu", "gpu"]


def _get_array_module(prefer_gpu: bool):
    if prefer_gpu:
        try:
            import cupy as cp  # type: ignore

            return cp, "gpu"
        except ImportError:
            pass
    return np, "cpu"


@dataclass(slots=True)
class ExactMethodResult:
    method_name: str
    initial_state_type: str
    num_of_particles: float
    gamma: float
    total_time: float
    dt: float
    num_of_samples: int
    backend: ArrayBackend
    runtime_seconds: float
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
    prefer_gpu: bool = True,
) -> ExactMethodResult:
    initial_state_type = _validate_initial_state(initial_state_type)
    xp, backend = _get_array_module(prefer_gpu=prefer_gpu)

    if dt <= 0:
        raise ValueError("dt must be positive.")
    if time <= 0:
        raise ValueError("time must be positive.")
    if gamma < 0:
        raise ValueError("gamma must be non-negative.")

    start = perf_counter()
    time_values_backend = xp.arange(0.0, time + dt, dt, dtype=xp.float64)
    mean_values_backend = num_of_particles * xp.exp(-gamma * time_values_backend)

    if initial_state_type == "fock":
        variance_backend = num_of_particles * (
            xp.exp(-gamma * time_values_backend) - xp.exp(-2.0 * gamma * time_values_backend)
        )
    else:
        variance_backend = mean_values_backend.copy()

    if backend == "gpu":
        time_values = xp.asnumpy(time_values_backend)
        mean_values = xp.asnumpy(mean_values_backend)
        variance = xp.asnumpy(variance_backend)
    else:
        time_values = np.asarray(time_values_backend)
        mean_values = np.asarray(mean_values_backend)
        variance = np.asarray(variance_backend)

    runtime_seconds = perf_counter() - start
    return ExactMethodResult(
        method_name="exact",
        initial_state_type=initial_state_type,
        num_of_particles=num_of_particles,
        gamma=gamma,
        total_time=time,
        dt=dt,
        num_of_samples=num_of_samples,
        backend=backend,
        runtime_seconds=runtime_seconds,
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
    prefer_gpu: bool = True,
):
    result = simulate_exact_method(
        initial_state_type=initial_state_type,
        num_of_particles=num_of_particles,
        gamma=gamma,
        time=time,
        dt=dt,
        num_of_samples=num_of_samples,
        prefer_gpu=prefer_gpu,
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
