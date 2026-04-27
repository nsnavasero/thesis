from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import sqrt
from time import perf_counter

import numpy as np
import tracemalloc

from .exact_method import _validate_initial_state


@dataclass(slots=True)
class MultiSiteDensityMatrixMethodResult:
    method_name: str
    initial_state_type: str
    site_occupations: tuple[float, ...]
    interaction_strength: float
    gamma: float
    hopping: float
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
    num_sites: int
    local_hilbert_size: int
    time_values: np.ndarray
    g1: np.ndarray
    mean_particle_numbers: np.ndarray
    variance: np.ndarray
    factorial_second_moment: np.ndarray
    total_mean_particle_number: np.ndarray


def _normalize_site_occupations(
    *,
    initial_state_type: str,
    site_occupations: Sequence[float] | float,
    num_sites: int | None,
) -> tuple[float, ...]:
    initial_state_type = _validate_initial_state(initial_state_type)

    if isinstance(site_occupations, (int, float)):
        if num_sites is None:
            raise ValueError("num_sites must be provided when site_occupations is a scalar.")
        normalized = tuple(float(site_occupations) for _ in range(num_sites))
    else:
        normalized = tuple(float(value) for value in site_occupations)
        if num_sites is not None and len(normalized) != num_sites:
            raise ValueError("num_sites must match the length of site_occupations.")

    if len(normalized) == 0:
        raise ValueError("site_occupations must contain at least one site.")
    if any(value < 0 for value in normalized):
        raise ValueError("site_occupations must be non-negative.")
    if initial_state_type == "fock":
        for value in normalized:
            if not float(value).is_integer():
                raise ValueError("For a fock state, each site occupation must be an integer.")

    return normalized


def _build_multisite_initial_density_matrix(
    *,
    initial_state_type: str,
    local_hilbert_size: int,
    site_occupations: Sequence[float],
):
    import qutip as qt

    if initial_state_type == "fock":
        density_matrices = []
        for occupation in site_occupations:
            fock_index = int(round(occupation))
            if fock_index >= local_hilbert_size:
                raise ValueError("local_hilbert_size must exceed every requested Fock occupation.")
            density_matrices.append(qt.fock_dm(local_hilbert_size, fock_index))
        return qt.tensor(density_matrices)

    density_matrices = []
    for occupation in site_occupations:
        alpha = sqrt(occupation)
        density_matrices.append(qt.coherent_dm(local_hilbert_size, alpha))
    return qt.tensor(density_matrices)


def simulate_multisite_density_matrix_method(
    *,
    initial_state_type: str,
    site_occupations: Sequence[float] | float,
    interaction_strength: float = 0.0,
    gamma: float,
    hopping: float,
    time: float,
    dt: float,
    local_hilbert_size: int,
    num_sites: int | None = None,
):
    import qutip as qt

    initial_state_type = _validate_initial_state(initial_state_type)
    normalized_site_occupations = _normalize_site_occupations(
        initial_state_type=initial_state_type,
        site_occupations=site_occupations,
        num_sites=num_sites,
    )
    num_sites = len(normalized_site_occupations)

    if dt <= 0:
        raise ValueError("dt must be positive.")
    if time <= 0:
        raise ValueError("time must be positive.")
    if gamma < 0:
        raise ValueError("gamma must be non-negative.")
    if local_hilbert_size <= 0:
        raise ValueError("local_hilbert_size must be positive.")

    total_start = perf_counter()

    setup_start = perf_counter()
    time_values = np.arange(0.0, time + dt, dt, dtype=float)
    identities = [qt.qeye(local_hilbert_size) for _ in range(num_sites)]
    annihilators: list[qt.Qobj] = []
    number_ops: list[qt.Qobj] = []
    factorial_second_moment_ops: list[qt.Qobj] = []

    for site_index in range(num_sites):
        factors = identities.copy()
        factors[site_index] = qt.destroy(local_hilbert_size)
        a_site = qt.tensor(factors)
        annihilators.append(a_site)
        number_ops.append(a_site.dag() * a_site)
        factorial_second_moment_ops.append((a_site.dag() ** 2) * (a_site ** 2))

    hamiltonian = 0
    for site_index in range(num_sites):
        hamiltonian += (
            0.5
            * interaction_strength
            * (annihilators[site_index].dag() ** 2)
            * (annihilators[site_index] ** 2)
        )

    for site_index in range(num_sites - 1):
        hamiltonian += -hopping * (
            annihilators[site_index].dag() * annihilators[site_index + 1]
            + annihilators[site_index + 1].dag() * annihilators[site_index]
        )

    collapse_operators = [np.sqrt(gamma) * a_site for a_site in annihilators]
    rho0 = _build_multisite_initial_density_matrix(
        initial_state_type=initial_state_type,
        local_hilbert_size=local_hilbert_size,
        site_occupations=normalized_site_occupations,
    )
    setup_runtime_seconds = perf_counter() - setup_start

    tracemalloc.start()
    solve_start = perf_counter()
    observables = annihilators + number_ops + factorial_second_moment_ops
    result = qt.mesolve(hamiltonian, rho0, time_values, collapse_operators, observables)
    _, solver_peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    solve_runtime_seconds = perf_counter() - solve_start

    postprocess_start = perf_counter()
    g1 = np.column_stack(
        [np.asarray(result.expect[site_index], dtype=np.complex128) for site_index in range(num_sites)]
    )
    mean_particle_numbers = np.column_stack(
        [
            np.real(np.asarray(result.expect[num_sites + site_index], dtype=np.complex128))
            for site_index in range(num_sites)
        ]
    )
    factorial_second_moment = np.column_stack(
        [
            np.real(np.asarray(result.expect[2 * num_sites + site_index], dtype=np.complex128))
            for site_index in range(num_sites)
        ]
    )
    variance = factorial_second_moment + mean_particle_numbers - mean_particle_numbers**2
    total_mean_particle_number = np.sum(mean_particle_numbers, axis=1)
    postprocess_runtime_seconds = perf_counter() - postprocess_start

    total_runtime_seconds = perf_counter() - total_start
    return MultiSiteDensityMatrixMethodResult(
        method_name="densityMatrixMultiSite",
        initial_state_type=initial_state_type,
        site_occupations=normalized_site_occupations,
        interaction_strength=interaction_strength,
        gamma=gamma,
        hopping=hopping,
        total_time=time,
        dt=dt,
        num_of_samples=1,
        backend="cpu",
        seed=None,
        setup_runtime_seconds=setup_runtime_seconds,
        solve_runtime_seconds=solve_runtime_seconds,
        postprocess_runtime_seconds=postprocess_runtime_seconds,
        total_runtime_seconds=total_runtime_seconds,
        solver_peak_python_memory_mib=solver_peak_bytes / (1024 * 1024),
        num_sites=num_sites,
        local_hilbert_size=local_hilbert_size,
        time_values=time_values,
        g1=g1,
        mean_particle_numbers=mean_particle_numbers,
        variance=variance,
        factorial_second_moment=factorial_second_moment,
        total_mean_particle_number=total_mean_particle_number,
    )
