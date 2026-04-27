from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from time import perf_counter

import numpy as np
import tracemalloc

from .config import resolve_hilbert_size
from .exact_method import _validate_initial_state
from .io_utils import compute_g2_from_mean_and_factorial_second_moment, save_method_output_csv


@dataclass(slots=True)
class DensityMatrixMethodResult:
    method_name: str
    initial_state_type: str
    num_of_particles: float
    interaction_strength: float
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
    hilbert_size: int
    coherent_alpha: complex | None
    time_values: np.ndarray
    mean_particle_number: np.ndarray
    variance: np.ndarray
    factorial_second_moment: np.ndarray
    g1: np.ndarray
    g2: np.ndarray

def _build_initial_density_matrix(initial_state_type: str, hilbert_size: int, num_of_particles: float):
    import qutip as qt

    if initial_state_type == "fock":
        fock_index = int(round(num_of_particles))
        if not np.isclose(num_of_particles, fock_index):
            raise ValueError("For a fock state, num_of_particles should be an integer.")
        if fock_index >= hilbert_size:
            raise ValueError("hilbert_size must be larger than the requested Fock occupation.")
        return qt.fock_dm(hilbert_size, fock_index), None

    alpha = sqrt(num_of_particles)
    return qt.coherent_dm(hilbert_size, alpha), alpha


def simulate_density_matrix_method(
    *,
    initial_state_type: str,
    num_of_particles: float,
    interaction_strength: float = 0.0,
    gamma: float,
    time: float,
    dt: float,
    num_of_samples: int = 1,
    hilbert_size: int | None = None,
):
    import qutip as qt

    initial_state_type = _validate_initial_state(initial_state_type)

    if dt <= 0:
        raise ValueError("dt must be positive.")
    if time <= 0:
        raise ValueError("time must be positive.")
    if gamma < 0:
        raise ValueError("gamma must be non-negative.")

    hilbert_size = resolve_hilbert_size(
        initial_state_type=initial_state_type,
        num_of_particles=num_of_particles,
        hilbert_size=hilbert_size,
    )

    total_start = perf_counter()

    setup_start = perf_counter()
    time_values = np.arange(0.0, time + dt, dt, dtype=float)
    a = qt.destroy(hilbert_size)
    n_op = a.dag() * a
    factorial_second_moment_op = (a.dag() ** 2) * (a ** 2)
    hamiltonian = 0.5 * interaction_strength * (a.dag() ** 2) * (a ** 2)
    collapse_operators = [np.sqrt(gamma) * a]
    rho0, coherent_alpha = _build_initial_density_matrix(
        initial_state_type=initial_state_type,
        hilbert_size=hilbert_size,
        num_of_particles=num_of_particles,
    )
    setup_runtime_seconds = perf_counter() - setup_start

    tracemalloc.start()
    solve_start = perf_counter()
    result = qt.mesolve(hamiltonian, rho0, time_values, collapse_operators, [a, n_op, factorial_second_moment_op])
    _, solver_peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    solve_runtime_seconds = perf_counter() - solve_start

    postprocess_start = perf_counter()
    g1 = np.asarray(result.expect[0], dtype=np.complex128)
    mean_particle_number = np.real_if_close(np.asarray(result.expect[1], dtype=np.complex128)).astype(float)
    factorial_second_moment = np.real_if_close(np.asarray(result.expect[2], dtype=np.complex128)).astype(float)
    variance = factorial_second_moment + mean_particle_number - mean_particle_number**2
    g2 = compute_g2_from_mean_and_factorial_second_moment(mean_particle_number, factorial_second_moment)
    postprocess_runtime_seconds = perf_counter() - postprocess_start

    total_runtime_seconds = perf_counter() - total_start
    return DensityMatrixMethodResult(
        method_name="densityMatrix",
        initial_state_type=initial_state_type,
        num_of_particles=num_of_particles,
        interaction_strength=interaction_strength,
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
        solver_peak_python_memory_mib=solver_peak_bytes / (1024 * 1024),
        hilbert_size=hilbert_size,
        coherent_alpha=coherent_alpha,
        time_values=time_values,
        mean_particle_number=mean_particle_number,
        variance=variance,
        factorial_second_moment=factorial_second_moment,
        g1=g1,
        g2=g2,
    )


def run_density_matrix_and_save(
    output_dir: str,
    *,
    initial_state_type: str,
    num_of_particles: float,
    interaction_strength: float = 0.0,
    gamma: float,
    time: float,
    dt: float,
    num_of_samples: int = 1,
    hilbert_size: int | None = None,
):
    result = simulate_density_matrix_method(
        initial_state_type=initial_state_type,
        num_of_particles=num_of_particles,
        interaction_strength=interaction_strength,
        gamma=gamma,
        time=time,
        dt=dt,
        num_of_samples=num_of_samples,
        hilbert_size=hilbert_size,
    )
    output_path = save_method_output_csv(
        output_dir,
        method_name=result.method_name,
        initial_state_type=result.initial_state_type,
        num_of_particles=result.num_of_particles,
        interaction_strength=result.interaction_strength,
        gamma=result.gamma,
        time=result.total_time,
        dt=result.dt,
        num_of_samples=result.num_of_samples,
        hilbert_size=result.hilbert_size,
        seed=result.seed,
        time_values=result.time_values,
        mean_values=result.mean_particle_number,
        variance_values=result.variance,
        extra_columns={
            "factorial_second_moment": result.factorial_second_moment,
            "g1_real": np.real(result.g1),
            "g1_imag": np.imag(result.g1),
            "g1_magnitude": np.abs(result.g1),
            "g2": result.g2,
        },
    )
    return result, output_path
