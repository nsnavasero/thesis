from __future__ import annotations

from dataclasses import dataclass
import secrets
from time import perf_counter

import numpy as np
import tracemalloc

from .config import resolve_hilbert_size
from .exact_method import _validate_initial_state
from .io_utils import save_method_output_csv


@dataclass(slots=True)
class MonteCarloMethodResult:
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
    fast_path_used: bool
    notes: str
    hilbert_size: int
    coherent_alpha: complex | None
    time_values: np.ndarray
    mean_particle_number: np.ndarray
    variance: np.ndarray


def _build_initial_state(initial_state_type: str, hilbert_size: int, num_of_particles: float):
    import qutip as qt

    if initial_state_type == "fock":
        fock_index = int(round(num_of_particles))
        if not np.isclose(num_of_particles, fock_index):
            raise ValueError("For a fock state, num_of_particles should be an integer.")
        if fock_index >= hilbert_size:
            raise ValueError("hilbert_size must be larger than the requested Fock occupation.")
        return qt.basis(hilbert_size, fock_index), None

    alpha = np.sqrt(num_of_particles)
    return qt.coherent(hilbert_size, alpha), alpha


def simulate_monte_carlo_method(
    *,
    initial_state_type: str,
    num_of_particles: float,
    interaction_strength: float = 0.0,
    gamma: float,
    time: float,
    dt: float,
    num_of_samples: int,
    hilbert_size: int | None = None,
    seed: int | None = None,
):
    import qutip as qt

    initial_state_type = _validate_initial_state(initial_state_type)

    if dt <= 0:
        raise ValueError("dt must be positive.")
    if time <= 0:
        raise ValueError("time must be positive.")
    if gamma < 0:
        raise ValueError("gamma must be non-negative.")
    if num_of_samples <= 0:
        raise ValueError("num_of_samples must be positive.")

    hilbert_size = resolve_hilbert_size(
        initial_state_type=initial_state_type,
        num_of_particles=num_of_particles,
        hilbert_size=hilbert_size,
    )

    time_values = np.arange(0.0, time + dt, dt, dtype=float)
    if seed is None:
        seed = secrets.randbits(63)

    coherent_alpha = None if initial_state_type == "fock" else np.sqrt(num_of_particles)
    total_start = perf_counter()

    setup_start = perf_counter()
    a = qt.destroy(hilbert_size)
    n_op = a.dag() * a
    n_sq_op = n_op * n_op
    hamiltonian = 0.5 * interaction_strength * (a.dag() ** 2) * (a ** 2)
    collapse_operators = [np.sqrt(gamma) * a]
    psi0, coherent_alpha = _build_initial_state(
        initial_state_type=initial_state_type,
        hilbert_size=hilbert_size,
        num_of_particles=num_of_particles,
    )

    # Monte Carlo is kept as a true trajectory-based method even when
    # interaction_strength = 0. In the pure-loss case the random jump process
    # still matters for Fock states, and keeping the sampled solver path helps
    # expose the method's convergence limitations in comparisons.
    options = {"progress_bar": False}
    setup_runtime_seconds = perf_counter() - setup_start

    tracemalloc.start()
    solve_start = perf_counter()
    result = qt.mcsolve(
        hamiltonian,
        psi0,
        time_values,
        collapse_operators,
        [n_op, n_sq_op],
        ntraj=num_of_samples,
        seeds=seed,
        options=options,
    )
    _, solver_peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    solve_runtime_seconds = perf_counter() - solve_start

    postprocess_start = perf_counter()
    mean_particle_number = np.real_if_close(np.asarray(result.expect[0], dtype=np.complex128)).astype(float)
    n_sq_values = np.real_if_close(np.asarray(result.expect[1], dtype=np.complex128)).astype(float)
    variance = np.maximum(n_sq_values - mean_particle_number**2, 0.0)
    postprocess_runtime_seconds = perf_counter() - postprocess_start

    fast_path_used = False
    if interaction_strength == 0.0:
        notes = (
            "interaction_strength = 0, but Monte Carlo was still run as a true trajectory-based dissipation method. "
            "Trajectory averaging remains relevant here for method comparison. QuTiP handles the trajectory "
            "loop internally, and this path is CPU-based."
        )
    else:
        notes = (
            "interaction_strength != 0, so the interacting Monte Carlo template was used through QuTiP mcsolve. "
            "Trajectory averaging matters in this path. QuTiP handles the trajectory loop internally, but this path "
            "is still CPU-based."
        )

    total_runtime_seconds = perf_counter() - total_start
    return MonteCarloMethodResult(
        method_name="monteCarlo",
        initial_state_type=initial_state_type,
        num_of_particles=num_of_particles,
        interaction_strength=interaction_strength,
        gamma=gamma,
        total_time=time,
        dt=dt,
        num_of_samples=num_of_samples,
        backend="cpu",
        seed=seed,
        setup_runtime_seconds=setup_runtime_seconds,
        solve_runtime_seconds=solve_runtime_seconds,
        postprocess_runtime_seconds=postprocess_runtime_seconds,
        total_runtime_seconds=total_runtime_seconds,
        solver_peak_python_memory_mib=solver_peak_bytes / (1024 * 1024),
        fast_path_used=fast_path_used,
        notes=notes,
        hilbert_size=hilbert_size,
        coherent_alpha=coherent_alpha,
        time_values=time_values,
        mean_particle_number=mean_particle_number,
        variance=variance,
    )


def run_monte_carlo_and_save(
    output_dir: str,
    *,
    initial_state_type: str,
    num_of_particles: float,
    interaction_strength: float = 0.0,
    gamma: float,
    time: float,
    dt: float,
    num_of_samples: int,
    hilbert_size: int | None = None,
    seed: int | None = None,
):
    result = simulate_monte_carlo_method(
        initial_state_type=initial_state_type,
        num_of_particles=num_of_particles,
        interaction_strength=interaction_strength,
        gamma=gamma,
        time=time,
        dt=dt,
        num_of_samples=num_of_samples,
        hilbert_size=hilbert_size,
        seed=seed,
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
        seed=result.seed,
        time_values=result.time_values,
        mean_values=result.mean_particle_number,
        variance_values=result.variance,
    )
    return result, output_path
