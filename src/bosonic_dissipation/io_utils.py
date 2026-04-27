from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


def _sanitize_number(value: float | int) -> str:
    numeric_value = float(value)
    if np.isclose(numeric_value, round(numeric_value)):
        return str(int(round(numeric_value)))
    return f"{numeric_value:.15g}".replace(".", "p")


def build_state_label(initial_state_type: str, num_of_particles: float) -> str:
    return f"{initial_state_type}{_sanitize_number(num_of_particles)}"


def compute_factorial_second_moment_from_mean_and_variance(
    mean_values: Iterable[float],
    variance_values: Iterable[float],
) -> np.ndarray:
    mean_array = np.asarray(mean_values, dtype=np.float64)
    variance_array = np.asarray(variance_values, dtype=np.float64)
    return variance_array + mean_array**2 - mean_array


def compute_g2_from_mean_and_factorial_second_moment(
    mean_values: Iterable[float],
    factorial_second_moment_values: Iterable[float],
    *,
    zero_tolerance: float = 1e-12,
) -> np.ndarray:
    mean_array = np.asarray(mean_values, dtype=np.float64)
    factorial_array = np.asarray(factorial_second_moment_values, dtype=np.float64)
    g2_values = np.full(mean_array.shape, np.nan, dtype=np.float64)
    denominator = mean_array**2
    valid_mask = denominator > zero_tolerance
    g2_values[valid_mask] = factorial_array[valid_mask] / denominator[valid_mask]
    return g2_values


def build_output_stem(
    method_name: str,
    initial_state_type: str,
    num_of_particles: float,
    interaction_strength: float | None,
    gamma: float,
    time: float,
    dt: float,
    num_of_samples: int,
    hilbert_size: int | None = None,
    seed: int | None = None,
) -> str:
    parts = [
        method_name,
        build_state_label(initial_state_type, num_of_particles),
        f"u{_sanitize_number(interaction_strength)}" if interaction_strength is not None else None,
        f"gamma{_sanitize_number(gamma)}",
        f"time{_sanitize_number(time)}",
        f"dt{_sanitize_number(dt)}",
        f"numOfSamples{num_of_samples}",
    ]
    parts = [part for part in parts if part is not None]
    if hilbert_size is not None:
        parts.append(f"hilbertSize{hilbert_size}")
    return "_".join(parts)


def save_method_output_csv(
    output_dir: str | Path,
    *,
    method_name: str,
    initial_state_type: str,
    num_of_particles: float,
    interaction_strength: float | None = None,
    gamma: float,
    time: float,
    dt: float,
    num_of_samples: int,
    hilbert_size: int | None = None,
    seed: int | None = None,
    time_values: Iterable[float],
    mean_values: Iterable[float],
    variance_values: Iterable[float],
    extra_columns: Mapping[str, Iterable[float]] | None = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = build_output_stem(
        method_name=method_name,
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
    output_path = output_dir / f"{stem}.csv"

    column_names = ["time", "mean_particle_number", "variance"]
    column_values = [
        np.asarray(time_values, dtype=np.float64),
        np.asarray(mean_values, dtype=np.float64),
        np.asarray(variance_values, dtype=np.float64),
    ]
    if extra_columns is not None:
        for column_name, values in extra_columns.items():
            column_names.append(column_name)
            column_values.append(np.asarray(values, dtype=np.float64))

    data = np.column_stack(column_values)
    np.savetxt(
        output_path,
        data,
        delimiter=",",
        header=",".join(column_names),
        comments="",
    )
    return output_path
