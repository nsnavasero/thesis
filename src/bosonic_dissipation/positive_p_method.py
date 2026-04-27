from __future__ import annotations

from dataclasses import dataclass
import secrets
from time import perf_counter

import numpy as np
import tracemalloc

from .exact_method import _validate_initial_state
from .io_utils import compute_g2_from_mean_and_factorial_second_moment, save_method_output_csv


ALPHA_INDEX = 0
ALPHA_PLUS_INDEX = 1


@dataclass(slots=True)
class PositivePMethodResult:
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
    time_values: np.ndarray
    mean_particle_number: np.ndarray
    variance: np.ndarray
    factorial_second_moment: np.ndarray
    g1: np.ndarray
    g2: np.ndarray
    mean_particle_number_imag: np.ndarray
    variance_imag: np.ndarray
    factorial_second_moment_imag: np.ndarray


def _sample_positive_p_initial_state(
    *,
    initial_state_type: str,
    num_of_particles: float,
    num_of_samples: int,
    xp,
    rng,
):
    """Return samples with shape (num_samples, num_sites=1, components=2).

    Internal component convention for this project:
    - `state[..., ALPHA_INDEX]` is Olsen-style `alpha`
    - `state[..., ALPHA_PLUS_INDEX]` is Olsen-style `alphaPlus`

    Mapping to the Deuar-style notation:
    - Deuar et al. (2012): `alpha`, `alphaTilde`
    - Olsen: `alpha`, `alphaPlus`
    - relation used here: `alphaTilde = conj(alphaPlus)`

    With that mapping:
    - mean particle number estimator: <alpha * alphaPlus>
    - factorial second moment estimator: <(alpha * alphaPlus)^2>
    - first-order coherence estimator: <alpha>
    """

    state = xp.zeros((num_of_samples, 1, 2), dtype=xp.complex128)

    if initial_state_type == "coherent":
        coherent_amplitude = xp.sqrt(xp.asarray(num_of_particles, dtype=xp.float64))
        state[:, 0, ALPHA_INDEX] = coherent_amplitude
        state[:, 0, ALPHA_PLUS_INDEX] = coherent_amplitude
        return state

    eta1 = rng.standard_normal(num_of_samples)
    eta2 = rng.standard_normal(num_of_samples)
    mu = (eta1 + 1j * eta2) / xp.sqrt(2.0)

    fock_gamma_radius_sq = rng.gamma(shape=num_of_particles + 1.0, scale=1.0, size=num_of_samples)
    fock_gamma_phase = rng.uniform(0.0, 2.0 * xp.pi, size=num_of_samples)
    fock_gamma_sample = xp.sqrt(fock_gamma_radius_sq) * xp.exp(1j * fock_gamma_phase)

    state[:, 0, ALPHA_INDEX] = fock_gamma_sample + mu
    state[:, 0, ALPHA_PLUS_INDEX] = xp.conjugate(fock_gamma_sample) - xp.conjugate(mu)
    return state


def _compute_u0_positive_p_observables(
    *,
    initial_state_type: str,
    num_of_particles: float,
    gamma: float,
    time_values_backend,
    num_of_samples: int,
    xp,
    rng,
):
    if initial_state_type == "coherent":
        coherent_amplitude = xp.sqrt(xp.asarray(num_of_particles, dtype=xp.float64))
        g1_complex = coherent_amplitude * xp.exp(-0.5 * gamma * time_values_backend)
        mean_particle_number_complex = num_of_particles * xp.exp(-gamma * time_values_backend)
        factorial_second_moment_complex = mean_particle_number_complex**2
        variance_complex = mean_particle_number_complex.copy()
        return (
            g1_complex,
            mean_particle_number_complex,
            variance_complex,
            factorial_second_moment_complex,
            True,
        )

    initial_state = _sample_positive_p_initial_state(
        initial_state_type=initial_state_type,
        num_of_particles=num_of_particles,
        num_of_samples=num_of_samples,
        xp=xp,
        rng=rng,
    )
    alpha0 = initial_state[:, 0, ALPHA_INDEX]
    alpha_plus0 = initial_state[:, 0, ALPHA_PLUS_INDEX]
    decay = xp.exp(-0.5 * gamma * time_values_backend)
    decay_sq = decay**2

    g1_complex = xp.mean(alpha0) * decay
    occupation0 = alpha0 * alpha_plus0
    mean_particle_number_complex = xp.mean(occupation0) * decay_sq
    factorial_second_moment_complex = xp.mean(occupation0**2) * (decay_sq**2)
    variance_complex = (
        factorial_second_moment_complex + mean_particle_number_complex - mean_particle_number_complex**2
    )
    return (
        g1_complex,
        mean_particle_number_complex,
        variance_complex,
        factorial_second_moment_complex,
        False,
    )


def _simulate_positive_p_interacting_single_site(
    *,
    initial_state_type: str,
    num_of_particles: float,
    interaction_strength: float,
    gamma: float,
    dt: float,
    num_of_samples: int,
    time_values_backend,
    xp,
    rng,
):
    """Vectorized single-site Positive-P solver for nonzero interaction."""

    initial_state = _sample_positive_p_initial_state(
        initial_state_type=initial_state_type,
        num_of_particles=num_of_particles,
        num_of_samples=num_of_samples,
        xp=xp,
        rng=rng,
    )
    alpha = initial_state[:, 0, ALPHA_INDEX].copy()
    alpha_plus = initial_state[:, 0, ALPHA_PLUS_INDEX].copy()

    num_times = int(time_values_backend.shape[0])
    g1_complex = xp.empty(num_times, dtype=xp.complex128)
    mean_particle_number_complex = xp.empty(num_times, dtype=xp.complex128)
    factorial_second_moment_complex = xp.empty(num_times, dtype=xp.complex128)

    occupation = alpha * alpha_plus
    g1_complex[0] = xp.mean(alpha)
    mean_particle_number_complex[0] = xp.mean(occupation)
    factorial_second_moment_complex[0] = xp.mean(occupation**2)

    sqrt_minus_i_u = xp.sqrt(-1j * interaction_strength)
    sqrt_plus_i_u = xp.sqrt(1j * interaction_strength)
    sqrt_dt = xp.sqrt(dt)

    for time_index in range(1, num_times):
        noise_1 = rng.standard_normal(num_of_samples) * sqrt_dt
        noise_2 = rng.standard_normal(num_of_samples) * sqrt_dt

        drift_alpha = -0.5 * gamma * alpha - 1j * interaction_strength * alpha * alpha_plus * alpha
        drift_alpha_plus = -0.5 * gamma * alpha_plus + 1j * interaction_strength * alpha_plus * alpha * alpha_plus

        alpha = alpha + drift_alpha * dt + sqrt_minus_i_u * alpha * noise_1
        alpha_plus = alpha_plus + drift_alpha_plus * dt + sqrt_plus_i_u * alpha_plus * noise_2

        occupation = alpha * alpha_plus
        g1_complex[time_index] = xp.mean(alpha)
        mean_particle_number_complex[time_index] = xp.mean(occupation)
        factorial_second_moment_complex[time_index] = xp.mean(occupation**2)

    variance_complex = (
        factorial_second_moment_complex + mean_particle_number_complex - mean_particle_number_complex**2
    )
    return g1_complex, mean_particle_number_complex, variance_complex, factorial_second_moment_complex


def simulate_positive_p_method(
    *,
    initial_state_type: str,
    num_of_particles: float,
    interaction_strength: float = 0.0,
    gamma: float,
    time: float,
    dt: float,
    num_of_samples: int,
    seed: int | None = None,
):
    initial_state_type = _validate_initial_state(initial_state_type)

    if dt <= 0:
        raise ValueError("dt must be positive.")
    if time <= 0:
        raise ValueError("time must be positive.")
    if gamma < 0:
        raise ValueError("gamma must be non-negative.")
    if num_of_samples <= 0:
        raise ValueError("num_of_samples must be positive.")
    if initial_state_type == "fock" and not float(num_of_particles).is_integer():
        raise ValueError("For a fock state, num_of_particles should be an integer.")

    if seed is None:
        seed = secrets.randbits(63)

    xp = np
    rng = np.random.default_rng(seed)

    total_start = perf_counter()

    setup_start = perf_counter()
    time_values_backend = xp.arange(0.0, time + dt, dt, dtype=xp.float64)
    setup_runtime_seconds = perf_counter() - setup_start

    tracemalloc.start()
    solve_start = perf_counter()
    if interaction_strength == 0.0:
        (
            g1_complex,
            mean_particle_number_complex,
            variance_complex,
            factorial_second_moment_complex,
            coherent_fast_path_used,
        ) = _compute_u0_positive_p_observables(
            initial_state_type=initial_state_type,
            num_of_particles=num_of_particles,
            gamma=gamma,
            time_values_backend=time_values_backend,
            num_of_samples=num_of_samples,
            xp=xp,
            rng=rng,
        )
        if initial_state_type == "coherent":
            fast_path_used = coherent_fast_path_used
            notes = (
                "interaction_strength = 0 and the coherent-state Positive-P representation is deterministic here, "
                "so num_of_samples was ignored in the fast path."
            )
        else:
            fast_path_used = False
            notes = (
                "interaction_strength = 0, so the time evolution is deterministic, but the Fock-state Positive-P "
                "initial representation was still sampled across num_of_samples trajectories."
            )
    else:
        (
            g1_complex,
            mean_particle_number_complex,
            variance_complex,
            factorial_second_moment_complex,
        ) = _simulate_positive_p_interacting_single_site(
            initial_state_type=initial_state_type,
            num_of_particles=num_of_particles,
            interaction_strength=interaction_strength,
            gamma=gamma,
            dt=dt,
            num_of_samples=num_of_samples,
            time_values_backend=time_values_backend,
            xp=xp,
            rng=rng,
        )
        fast_path_used = False
        notes = (
            "interaction_strength != 0, so the vectorized stochastic Positive-P solver was used. "
            "Trajectories were batched across samples for speed."
        )
    _, solver_peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    solve_runtime_seconds = perf_counter() - solve_start

    postprocess_start = perf_counter()
    time_values = np.asarray(time_values_backend)
    g1_complex = np.asarray(g1_complex, dtype=np.complex128)
    mean_particle_number_complex = np.asarray(mean_particle_number_complex, dtype=np.complex128)
    factorial_second_moment_complex = np.asarray(factorial_second_moment_complex, dtype=np.complex128)
    variance_complex = np.asarray(variance_complex, dtype=np.complex128)
    mean_particle_number = np.real(mean_particle_number_complex)
    factorial_second_moment = np.real(factorial_second_moment_complex)
    mean_particle_number_imag = np.imag(mean_particle_number_complex)
    variance_imag = np.imag(variance_complex)
    factorial_second_moment_imag = np.imag(factorial_second_moment_complex)
    g2 = compute_g2_from_mean_and_factorial_second_moment(mean_particle_number, factorial_second_moment)
    postprocess_runtime_seconds = perf_counter() - postprocess_start

    total_runtime_seconds = perf_counter() - total_start
    return PositivePMethodResult(
        method_name="positiveP",
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
        time_values=time_values,
        mean_particle_number=mean_particle_number,
        variance=np.real(variance_complex),
        factorial_second_moment=factorial_second_moment,
        g1=g1_complex,
        g2=g2,
        mean_particle_number_imag=mean_particle_number_imag,
        variance_imag=variance_imag,
        factorial_second_moment_imag=factorial_second_moment_imag,
    )


def run_positive_p_and_save(
    output_dir: str,
    *,
    initial_state_type: str,
    num_of_particles: float,
    interaction_strength: float = 0.0,
    gamma: float,
    time: float,
    dt: float,
    num_of_samples: int,
    seed: int | None = None,
):
    result = simulate_positive_p_method(
        initial_state_type=initial_state_type,
        num_of_particles=num_of_particles,
        interaction_strength=interaction_strength,
        gamma=gamma,
        time=time,
        dt=dt,
        num_of_samples=num_of_samples,
        seed=seed,
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
            "mean_particle_number_imag": result.mean_particle_number_imag,
            "variance_imag": result.variance_imag,
            "factorial_second_moment_imag": result.factorial_second_moment_imag,
        },
    )
    return result, output_path
