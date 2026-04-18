from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Optional

from scipy import special


DEFAULT_DT = 1e-3
DEFAULT_NUM_SAMPLES = 1
DEFAULT_HILBERT_SIZE: int | None = None
DEFAULT_COHERENT_MEAN_RELATIVE_TOLERANCE = 0.01


def _validate_initial_state(initial_state_type: str) -> str:
    normalized = initial_state_type.lower()
    if normalized not in {"fock", "coherent"}:
        raise ValueError("initial_state_type must be either 'fock' or 'coherent'.")
    return normalized


def recommend_hilbert_size(initial_state_type: str, num_of_particles: float) -> int:
    initial_state_type = _validate_initial_state(initial_state_type)

    if num_of_particles < 0:
        raise ValueError("num_of_particles must be non-negative.")

    if initial_state_type == "fock":
        max_occupation = int(ceil(num_of_particles))
        return max(1, max_occupation + 1)

    return recommend_coherent_hilbert_size(
        num_of_particles=num_of_particles,
        relative_tolerance=DEFAULT_COHERENT_MEAN_RELATIVE_TOLERANCE,
    )


def coherent_mean_from_truncation(num_of_particles: float, max_occupation: int) -> float:
    if num_of_particles < 0:
        raise ValueError("num_of_particles must be non-negative.")
    if max_occupation < 0:
        raise ValueError("max_occupation must be non-negative.")

    if num_of_particles == 0:
        return 0.0
    if max_occupation == 0:
        return 0.0

    # For a Poisson-distributed coherent-state occupation with mean mu,
    # \sum_{n=0}^{x} n P(n) = mu * \sum_{m=0}^{x-1} P(m).
    # Using the Poisson CDF avoids the factorial-heavy direct series and
    # remains stable for large mu.
    return float(num_of_particles * special.pdtr(max_occupation - 1, num_of_particles))


def coherent_missing_mean_from_truncation(num_of_particles: float, max_occupation: int) -> float:
    if num_of_particles < 0:
        raise ValueError("num_of_particles must be non-negative.")
    if max_occupation < 0:
        raise ValueError("max_occupation must be non-negative.")
    if num_of_particles == 0:
        return 0.0
    if max_occupation == 0:
        return float(num_of_particles)

    return float(num_of_particles * special.pdtrc(max_occupation - 1, num_of_particles))


def recommend_coherent_hilbert_size(
    *,
    num_of_particles: float,
    relative_tolerance: float = DEFAULT_COHERENT_MEAN_RELATIVE_TOLERANCE,
) -> int:
    if num_of_particles < 0:
        raise ValueError("num_of_particles must be non-negative.")
    if relative_tolerance <= 0:
        raise ValueError("relative_tolerance must be positive.")
    if num_of_particles == 0:
        return 1

    target_mean_particle_number = num_of_particles
    absolute_tolerance = relative_tolerance * target_mean_particle_number

    def tail_mean_is_small_enough(max_occupation: int) -> bool:
        return coherent_missing_mean_from_truncation(
            num_of_particles=num_of_particles,
            max_occupation=max_occupation,
        ) <= absolute_tolerance

    lower_fail = -1
    upper_pass = max(1, int(ceil(num_of_particles)))
    while not tail_mean_is_small_enough(upper_pass):
        lower_fail = upper_pass
        upper_pass *= 2

    while upper_pass - lower_fail > 1:
        midpoint = (lower_fail + upper_pass) // 2
        if tail_mean_is_small_enough(midpoint):
            upper_pass = midpoint
        else:
            lower_fail = midpoint

    return upper_pass + 1


def resolve_hilbert_size(
    *,
    initial_state_type: str,
    num_of_particles: float,
    hilbert_size: int | None = None,
    fallback_hilbert_size: int | None = DEFAULT_HILBERT_SIZE,
) -> int:
    if hilbert_size is not None:
        return hilbert_size
    if fallback_hilbert_size is not None:
        return fallback_hilbert_size
    return recommend_hilbert_size(initial_state_type, num_of_particles)


@dataclass(slots=True)
class MethodRunConfig:
    method_name: str
    initial_state_type: str
    num_of_particles: float
    gamma: float
    time: float
    dt: Optional[float] = None
    num_of_samples: Optional[int] = None
    hilbert_size: Optional[int] = None
    seed: Optional[int] = None

    def resolved_dt(self, fallback_dt: float = DEFAULT_DT) -> float:
        return fallback_dt if self.dt is None else self.dt

    def resolved_num_of_samples(
        self,
        fallback_num_of_samples: int = DEFAULT_NUM_SAMPLES,
    ) -> int:
        return fallback_num_of_samples if self.num_of_samples is None else self.num_of_samples

    def resolved_hilbert_size(
        self,
        fallback_hilbert_size: int | None = DEFAULT_HILBERT_SIZE,
    ) -> int:
        return resolve_hilbert_size(
            initial_state_type=self.initial_state_type,
            num_of_particles=self.num_of_particles,
            hilbert_size=self.hilbert_size,
            fallback_hilbert_size=fallback_hilbert_size,
        )
