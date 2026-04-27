from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import secrets
from time import perf_counter

import numpy as np
import tracemalloc

from .exact_method import _validate_initial_state


MAX_TRAJECTORY_MAGNITUDE = 1e150


@dataclass(slots=True)
class MultiSitePositivePMethodResult:
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
    completed_full_time: bool
    stop_time: float | None
    notes: str
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


def _site_fock_positive_p_samples(
    *,
    occupation: int,
    num_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    eta1 = rng.standard_normal(num_samples)
    eta2 = rng.standard_normal(num_samples)
    mu = (eta1 + 1j * eta2) / np.sqrt(2.0)

    gamma_radius_sq = rng.gamma(shape=occupation + 1.0, scale=1.0, size=num_samples)
    gamma_phase = rng.uniform(0.0, 2.0 * np.pi, size=num_samples)
    gamma_sample = np.sqrt(gamma_radius_sq) * np.exp(1j * gamma_phase)

    alpha = gamma_sample + mu
    alpha_plus = np.conjugate(gamma_sample) - np.conjugate(mu)
    return alpha.astype(np.complex128), alpha_plus.astype(np.complex128)


def _sample_multisite_positive_p_initial_state(
    *,
    initial_state_type: str,
    site_occupations: Sequence[float],
    num_of_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    num_sites = len(site_occupations)
    alpha = np.zeros((num_of_samples, num_sites), dtype=np.complex128)
    alpha_plus = np.zeros((num_of_samples, num_sites), dtype=np.complex128)

    if initial_state_type == "coherent":
        coherent_amplitudes = np.sqrt(np.asarray(site_occupations, dtype=np.float64))
        alpha[:, :] = coherent_amplitudes
        alpha_plus[:, :] = coherent_amplitudes
        return alpha, alpha_plus

    for site_index, occupation in enumerate(site_occupations):
        alpha[:, site_index], alpha_plus[:, site_index] = _site_fock_positive_p_samples(
            occupation=int(round(occupation)),
            num_samples=num_of_samples,
            rng=rng,
        )

    return alpha, alpha_plus


def _build_neighbor_sum(values: np.ndarray) -> np.ndarray:
    num_sites = values.shape[1]
    neighbor_sum = np.zeros_like(values)
    for site_index in range(num_sites):
        if site_index > 0:
            neighbor_sum[:, site_index] += values[:, site_index - 1]
        if site_index + 1 < num_sites:
            neighbor_sum[:, site_index] += values[:, site_index + 1]
    return neighbor_sum


def simulate_multisite_positive_p_method(
    *,
    initial_state_type: str,
    site_occupations: Sequence[float] | float,
    interaction_strength: float = 0.0,
    gamma: float,
    hopping: float,
    time: float,
    dt: float,
    num_of_samples: int,
    num_sites: int | None = None,
    seed: int | None = None,
):
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
    if num_of_samples <= 0:
        raise ValueError("num_of_samples must be positive.")

    if seed is None:
        seed = secrets.randbits(63)
    rng = np.random.default_rng(seed)

    total_start = perf_counter()

    setup_start = perf_counter()
    time_values = np.arange(0.0, time + dt, dt, dtype=np.float64)
    alpha, alpha_plus = _sample_multisite_positive_p_initial_state(
        initial_state_type=initial_state_type,
        site_occupations=normalized_site_occupations,
        num_of_samples=num_of_samples,
        rng=rng,
    )
    setup_runtime_seconds = perf_counter() - setup_start

    tracemalloc.start()
    solve_start = perf_counter()
    occupation = alpha_plus * alpha
    g1_rows: list[np.ndarray] = [np.mean(alpha, axis=0)]
    mean_rows: list[np.ndarray] = [np.real(np.mean(occupation, axis=0))]
    factorial_rows: list[np.ndarray] = [np.real(np.mean(occupation**2, axis=0))]

    sqrt_minus_i_u = np.sqrt(-1j * interaction_strength)
    sqrt_plus_i_u = np.sqrt(1j * interaction_strength)
    sqrt_dt = np.sqrt(dt)
    completed_full_time = True

    for _time_index in range(1, time_values.size):
        alpha_neighbors = _build_neighbor_sum(alpha)
        alpha_plus_neighbors = _build_neighbor_sum(alpha_plus)

        drift_alpha = (
            -0.5 * gamma * alpha
            - 1j * interaction_strength * alpha_plus * alpha * alpha
            + 1j * hopping * alpha_neighbors
        )
        drift_alpha_plus = (
            -0.5 * gamma * alpha_plus
            + 1j * interaction_strength * alpha_plus * alpha * alpha_plus
            - 1j * hopping * alpha_plus_neighbors
        )

        noise_1 = rng.standard_normal((num_of_samples, num_sites)) * sqrt_dt
        noise_2 = rng.standard_normal((num_of_samples, num_sites)) * sqrt_dt

        alpha = alpha + drift_alpha * dt + sqrt_minus_i_u * alpha * noise_1
        alpha_plus = alpha_plus + drift_alpha_plus * dt + sqrt_plus_i_u * alpha_plus * noise_2

        if not np.isfinite(alpha).all() or not np.isfinite(alpha_plus).all():
            completed_full_time = False
            break

        if np.max(np.abs(alpha)) > MAX_TRAJECTORY_MAGNITUDE or np.max(np.abs(alpha_plus)) > MAX_TRAJECTORY_MAGNITUDE:
            completed_full_time = False
            break

        current_mean = np.real(np.mean(alpha_plus * alpha, axis=0))
        if not np.isfinite(current_mean).all():
            completed_full_time = False
            break

        occupation = alpha_plus * alpha
        g1_rows.append(np.mean(alpha, axis=0))
        mean_rows.append(current_mean)
        factorial_rows.append(np.real(np.mean(occupation**2, axis=0)))

    if not completed_full_time and len(mean_rows) > 1:
        g1_rows = g1_rows[:-1]
        mean_rows = mean_rows[:-1]
        factorial_rows = factorial_rows[:-1]

    _, solver_peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    solve_runtime_seconds = perf_counter() - solve_start

    postprocess_start = perf_counter()
    g1 = np.asarray(g1_rows, dtype=np.complex128)
    mean_particle_numbers = np.asarray(mean_rows, dtype=float)
    factorial_second_moment = np.asarray(factorial_rows, dtype=float)
    variance = factorial_second_moment + mean_particle_numbers - mean_particle_numbers**2
    stable_time_values = time_values[: mean_particle_numbers.shape[0]]
    total_mean_particle_number = np.sum(mean_particle_numbers, axis=1)
    stop_time = float(stable_time_values[-1]) if stable_time_values.size else None
    postprocess_runtime_seconds = perf_counter() - postprocess_start

    total_runtime_seconds = perf_counter() - total_start
    return MultiSitePositivePMethodResult(
        method_name="positivePMultiSite",
        initial_state_type=initial_state_type,
        site_occupations=normalized_site_occupations,
        interaction_strength=interaction_strength,
        gamma=gamma,
        hopping=hopping,
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
        num_sites=num_sites,
        completed_full_time=completed_full_time,
        stop_time=stop_time,
        notes="Open-boundary nearest-neighbor multisite Positive-P simulation.",
        time_values=stable_time_values,
        g1=g1,
        mean_particle_numbers=mean_particle_numbers,
        variance=variance,
        factorial_second_moment=factorial_second_moment,
        total_mean_particle_number=total_mean_particle_number,
    )
