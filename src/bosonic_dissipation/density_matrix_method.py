from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from time import perf_counter

import numpy as np

from .config import resolve_hilbert_size
from .exact_method import _validate_initial_state
from .io_utils import save_method_output_csv


@dataclass(slots=True)
class DensityMatrixMethodResult:
    method_name: str
    initial_state_type: str
    num_of_particles: float
    gamma: float
    total_time: float
    dt: float
    num_of_samples: int
    backend: str
    runtime_seconds: float
    hilbert_size: int
    coherent_alpha: complex | None
    time_values: np.ndarray
    mean_particle_number: np.ndarray
    variance: np.ndarray

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

    start = perf_counter()

    time_values = np.arange(0.0, time + dt, dt, dtype=float)
    a = qt.destroy(hilbert_size)
    n_op = a.dag() * a
    n_sq_op = n_op * n_op
    hamiltonian = 0 * a
    collapse_operators = [np.sqrt(gamma) * a]
    rho0, coherent_alpha = _build_initial_density_matrix(
        initial_state_type=initial_state_type,
        hilbert_size=hilbert_size,
        num_of_particles=num_of_particles,
    )

    result = qt.mesolve(hamiltonian, rho0, time_values, collapse_operators, [n_op, n_sq_op])

    mean_particle_number = np.real_if_close(np.asarray(result.expect[0], dtype=np.complex128)).astype(float)
    n_sq_values = np.real_if_close(np.asarray(result.expect[1], dtype=np.complex128)).astype(float)
    variance = np.maximum(n_sq_values - mean_particle_number**2, 0.0)

    runtime_seconds = perf_counter() - start
    return DensityMatrixMethodResult(
        method_name="densityMatrix",
        initial_state_type=initial_state_type,
        num_of_particles=num_of_particles,
        gamma=gamma,
        total_time=time,
        dt=dt,
        num_of_samples=num_of_samples,
        backend="cpu",
        runtime_seconds=runtime_seconds,
        hilbert_size=hilbert_size,
        coherent_alpha=coherent_alpha,
        time_values=time_values,
        mean_particle_number=mean_particle_number,
        variance=variance,
    )


def run_density_matrix_and_save(
    output_dir: str,
    *,
    initial_state_type: str,
    num_of_particles: float,
    gamma: float,
    time: float,
    dt: float,
    num_of_samples: int = 1,
    hilbert_size: int | None = None,
):
    result = simulate_density_matrix_method(
        initial_state_type=initial_state_type,
        num_of_particles=num_of_particles,
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
        gamma=result.gamma,
        time=result.total_time,
        dt=result.dt,
        num_of_samples=result.num_of_samples,
        hilbert_size=result.hilbert_size,
        time_values=result.time_values,
        mean_values=result.mean_particle_number,
        variance_values=result.variance,
    )
    return result, output_path
