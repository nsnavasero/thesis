from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


def _sanitize_number(value: float | int) -> str:
    numeric_value = float(value)
    if np.isclose(numeric_value, round(numeric_value)):
        return str(int(round(numeric_value)))
    return f"{numeric_value:.15g}".replace(".", "p")


def build_output_stem(
    method_name: str,
    initial_state_type: str,
    num_of_particles: float,
    gamma: float,
    time: float,
    dt: float,
    num_of_samples: int,
    hilbert_size: int | None = None,
) -> str:
    parts = [
        method_name,
        initial_state_type,
        f"numOfParticles{_sanitize_number(num_of_particles)}",
        f"gamma{_sanitize_number(gamma)}",
        f"time{_sanitize_number(time)}",
        f"dt{_sanitize_number(dt)}",
        f"numOfSamples{num_of_samples}",
    ]
    if hilbert_size is not None:
        parts.append(f"hilbertSize{hilbert_size}")
    return "_".join(parts)


def save_method_output_csv(
    output_dir: str | Path,
    *,
    method_name: str,
    initial_state_type: str,
    num_of_particles: float,
    gamma: float,
    time: float,
    dt: float,
    num_of_samples: int,
    hilbert_size: int | None = None,
    time_values: Iterable[float],
    mean_values: Iterable[float],
    variance_values: Iterable[float],
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = build_output_stem(
        method_name=method_name,
        initial_state_type=initial_state_type,
        num_of_particles=num_of_particles,
        gamma=gamma,
        time=time,
        dt=dt,
        num_of_samples=num_of_samples,
        hilbert_size=hilbert_size,
    )
    output_path = output_dir / f"{stem}.csv"

    data = np.column_stack([time_values, mean_values, variance_values])
    np.savetxt(
        output_path,
        data,
        delimiter=",",
        header="time,mean_particle_number,variance",
        comments="",
    )
    return output_path
