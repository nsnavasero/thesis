from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qutip as qt

from .plotting import configure_plot_style, get_plot_profile, transform_time_values


DEFAULT_OUTPUT_ROOT = Path("outputs") / "two_site_comparison"
MAX_TRAJECTORY_MAGNITUDE = 1e150
MAX_SITE_OCCUPATION = 1.05
MIN_SITE_OCCUPATION = -0.05


@dataclass(slots=True)
class MeanResult:
    time_values: np.ndarray
    mean_occupations: np.ndarray
    completed_full_time: bool = True
    stop_time: float | None = None


def sanitize_suffix(value: float) -> str:
    return f"{value:.15g}".replace("-", "neg").replace(".", "p")


def build_case_name(*, interaction_strength: float, gamma: float, hopping: float, total_time: float) -> str:
    return (
        f"u{sanitize_suffix(interaction_strength)}"
        f"_gamma{sanitize_suffix(gamma)}"
        f"_j{sanitize_suffix(hopping)}"
        f"_t{sanitize_suffix(total_time)}"
    )


def get_case_directories(project_root: Path, case_name: str) -> dict[str, Path]:
    root = project_root / DEFAULT_OUTPUT_ROOT / case_name
    return {
        "root": root,
        "data": root / "data",
        "figures": root / "figures",
    }


def _ensure_case_directories(case_dirs: dict[str, Path]) -> None:
    for directory in case_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)


def _site_fock_positive_p_samples(occupation: int, num_samples: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    eta1 = rng.standard_normal(num_samples)
    eta2 = rng.standard_normal(num_samples)
    mu = (eta1 + 1j * eta2) / np.sqrt(2.0)

    gamma_radius_sq = rng.gamma(shape=occupation + 1.0, scale=1.0, size=num_samples)
    gamma_phase = rng.uniform(0.0, 2.0 * np.pi, size=num_samples)
    gamma_sample = np.sqrt(gamma_radius_sq) * np.exp(1j * gamma_phase)

    alpha = gamma_sample + mu
    alpha_tilde = np.conjugate(gamma_sample) - np.conjugate(mu)
    return alpha.astype(np.complex128), alpha_tilde.astype(np.complex128)


def sample_initial_fock_product_state(
    occupations: list[int],
    num_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    num_sites = len(occupations)
    alpha = np.zeros((num_samples, num_sites), dtype=np.complex128)
    alpha_tilde = np.zeros((num_samples, num_sites), dtype=np.complex128)

    for site_index, occupation in enumerate(occupations):
        alpha[:, site_index], alpha_tilde[:, site_index] = _site_fock_positive_p_samples(
            occupation=occupation,
            num_samples=num_samples,
            rng=rng,
        )

    return alpha, alpha_tilde


def build_neighbor_sum(values: np.ndarray) -> np.ndarray:
    num_sites = values.shape[1]
    neighbor_sum = np.zeros_like(values)
    for site_index in range(num_sites):
        if site_index > 0:
            neighbor_sum[:, site_index] += values[:, site_index - 1]
        if site_index + 1 < num_sites:
            neighbor_sum[:, site_index] += values[:, site_index + 1]
    return neighbor_sum


def simulate_positive_p_chain(
    *,
    num_sites: int,
    initial_occupations: list[int],
    interaction_strength: float,
    gamma: float,
    hopping: float,
    total_time: float,
    dt: float,
    num_samples: int,
    seed: int,
) -> MeanResult:
    rng = np.random.default_rng(seed)
    time_values = np.arange(0.0, total_time + dt, dt, dtype=float)

    alpha, alpha_tilde = sample_initial_fock_product_state(
        occupations=initial_occupations,
        num_samples=num_samples,
        rng=rng,
    )

    mean_rows: list[np.ndarray] = [np.real(np.mean(alpha_tilde * alpha, axis=0))]
    sqrt_minus_i_u = np.sqrt(-1j * interaction_strength)
    sqrt_plus_i_u = np.sqrt(1j * interaction_strength)
    sqrt_dt = np.sqrt(dt)
    completed_full_time = True

    for _time_index in range(1, time_values.size):
        alpha_neighbors = build_neighbor_sum(alpha)
        alpha_tilde_neighbors = build_neighbor_sum(alpha_tilde)

        drift_alpha = (
            -0.5 * gamma * alpha
            - 1j * interaction_strength * alpha_tilde * alpha * alpha
            + 1j * hopping * alpha_neighbors
        )
        drift_alpha_tilde = (
            -0.5 * gamma * alpha_tilde
            + 1j * interaction_strength * alpha_tilde * alpha * alpha_tilde
            - 1j * hopping * alpha_tilde_neighbors
        )

        noise_1 = rng.standard_normal((num_samples, num_sites)) * sqrt_dt
        noise_2 = rng.standard_normal((num_samples, num_sites)) * sqrt_dt

        alpha = alpha + drift_alpha * dt + sqrt_minus_i_u * alpha * noise_1
        alpha_tilde = alpha_tilde + drift_alpha_tilde * dt + sqrt_plus_i_u * alpha_tilde * noise_2

        if not np.isfinite(alpha).all() or not np.isfinite(alpha_tilde).all():
            completed_full_time = False
            break
        if np.max(np.abs(alpha)) > MAX_TRAJECTORY_MAGNITUDE or np.max(np.abs(alpha_tilde)) > MAX_TRAJECTORY_MAGNITUDE:
            completed_full_time = False
            break

        current_mean = np.real(np.mean(alpha_tilde * alpha, axis=0))
        if not np.isfinite(current_mean).all():
            completed_full_time = False
            break
        if np.any(current_mean > MAX_SITE_OCCUPATION) or np.any(current_mean < MIN_SITE_OCCUPATION):
            completed_full_time = False
            break

        mean_rows.append(current_mean)

    if not completed_full_time and len(mean_rows) > 1:
        mean_rows = mean_rows[:-1]

    stable_mean_occupations = np.asarray(mean_rows, dtype=float)
    stable_time_values = time_values[: stable_mean_occupations.shape[0]]
    return MeanResult(
        time_values=stable_time_values,
        mean_occupations=stable_mean_occupations,
        completed_full_time=completed_full_time,
        stop_time=float(stable_time_values[-1]),
    )


def build_density_matrix_result(
    *,
    num_sites: int,
    initial_occupations: list[int],
    interaction_strength: float,
    gamma: float,
    hopping: float,
    total_time: float,
    dt: float,
    local_dim: int = 2,
) -> MeanResult:
    time_values = np.arange(0.0, total_time + dt, dt, dtype=float)

    annihilators: list[qt.Qobj] = []
    number_ops: list[qt.Qobj] = []
    identities = [qt.qeye(local_dim) for _ in range(num_sites)]

    for site_index in range(num_sites):
        factors = identities.copy()
        factors[site_index] = qt.destroy(local_dim)
        a_site = qt.tensor(factors)
        annihilators.append(a_site)
        number_ops.append(a_site.dag() * a_site)

    hamiltonian = 0
    for site_index in range(num_sites):
        hamiltonian += 0.5 * interaction_strength * (annihilators[site_index].dag() ** 2) * (annihilators[site_index] ** 2)

    for site_index in range(num_sites - 1):
        hamiltonian += -hopping * (
            annihilators[site_index].dag() * annihilators[site_index + 1]
            + annihilators[site_index + 1].dag() * annihilators[site_index]
        )

    collapse_ops = [np.sqrt(gamma) * a_site for a_site in annihilators]
    rho0 = qt.tensor([qt.fock_dm(local_dim, occupation) for occupation in initial_occupations])
    result = qt.mesolve(hamiltonian, rho0, time_values, collapse_ops, number_ops)
    mean_occupations = np.column_stack([np.real(np.asarray(site_expect, dtype=np.complex128)) for site_expect in result.expect])
    return MeanResult(time_values=time_values, mean_occupations=mean_occupations)


def save_two_site_csv(path: Path, density: MeanResult, positive_p: MeanResult) -> None:
    positive_p_full = np.full_like(density.mean_occupations, np.nan, dtype=float)
    positive_p_full[: positive_p.mean_occupations.shape[0], :] = positive_p.mean_occupations

    columns = [density.time_values]
    headers = ["time"]

    for site_index in range(2):
        columns.append(density.mean_occupations[:, site_index])
        headers.append(f"density_n_site_{site_index + 1}")
        columns.append(positive_p_full[:, site_index])
        headers.append(f"positive_p_n_site_{site_index + 1}")

    stacked = np.column_stack(columns)
    np.savetxt(path, stacked, delimiter=",", header=",".join(headers), comments="")


def plot_two_site_csv(
    *,
    csv_path: Path,
    output_path: Path,
    site_index: int,
    gamma_value: float,
    plot_profile: str = "time",
) -> Path:
    configure_plot_style()
    profile = get_plot_profile(plot_profile)
    frame = pd.read_csv(csv_path)
    density_col = f"density_n_site_{site_index + 1}"
    positive_p_col = f"positive_p_n_site_{site_index + 1}"
    positive_frame = frame[["time", positive_p_col]].dropna()

    x_values = transform_time_values(frame["time"], gamma_value=gamma_value, plot_profile=profile)
    positive_x_values = transform_time_values(positive_frame["time"], gamma_value=gamma_value, plot_profile=profile)

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.plot(
        x_values,
        frame[density_col],
        color="#0b3c5d",
        linewidth=2.6,
        label="Density Matrix",
    )
    ax.plot(
        positive_x_values,
        positive_frame[positive_p_col],
        color="#c0392b",
        linewidth=2.0,
        linestyle="--",
        label="Positive-P",
    )
    ax.set_title(f"Two Sites: Mean n{site_index + 1}")
    ax.set_xlabel(profile.x_label)
    ax.set_ylabel("Mean Occupation")
    ax.set_xlim(0.0, float(np.max(x_values)))

    stop_time = float(positive_frame["time"].iloc[-1])
    stop_x_value = float(transform_time_values([stop_time], gamma_value=gamma_value, plot_profile=profile)[0])
    stability_text = "Positive-P stable until\n" + f"{profile.title_x_label} = {stop_x_value:.3f}"
    if stop_x_value < float(np.max(x_values)):
        ax.axvline(
            stop_x_value,
            color="#c0392b",
            linewidth=1.6,
            linestyle=":",
            alpha=0.9,
        )
    ax.legend(frameon=False)
    ax.text(
        0.97,
        0.83,
        stability_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#7f1d1d",
    )
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def run_two_site_comparison(
    *,
    project_root: Path,
    interaction_strength: float,
    gamma: float = 1.0,
    hopping: float = 1.0,
    total_time: float = 5.0,
    dt: float = 0.002,
    num_samples: int = 50_000,
    seed: int = 123456,
    plot_profile: str = "time",
    figure_subdir: str | None = None,
) -> dict[str, object]:
    case_name = build_case_name(
        interaction_strength=interaction_strength,
        gamma=gamma,
        hopping=hopping,
        total_time=total_time,
    )
    case_dirs = get_case_directories(project_root, case_name)
    _ensure_case_directories(case_dirs)

    density = build_density_matrix_result(
        num_sites=2,
        initial_occupations=[1, 0],
        interaction_strength=interaction_strength,
        gamma=gamma,
        hopping=hopping,
        total_time=total_time,
        dt=dt,
    )
    positive_p = simulate_positive_p_chain(
        num_sites=2,
        initial_occupations=[1, 0],
        interaction_strength=interaction_strength,
        gamma=gamma,
        hopping=hopping,
        total_time=total_time,
        dt=dt,
        num_samples=num_samples,
        seed=seed + 1,
    )

    csv_path = case_dirs["data"] / "two_site_mean_comparison.csv"
    save_two_site_csv(csv_path, density, positive_p)

    figure_dir = case_dirs["figures"]
    if figure_subdir:
        figure_dir = figure_dir / figure_subdir
    generated_figures = [
        plot_two_site_csv(
            csv_path=csv_path,
            output_path=figure_dir / f"two_site_mean_n{site_index + 1}.png",
            site_index=site_index,
            gamma_value=gamma,
            plot_profile=plot_profile,
        )
        for site_index in (0, 1)
    ]

    matched_density = MeanResult(
        time_values=density.time_values[: positive_p.time_values.size],
        mean_occupations=density.mean_occupations[: positive_p.time_values.size],
    )
    max_abs_error = float(np.max(np.abs(matched_density.mean_occupations - positive_p.mean_occupations)))

    return {
        "case_name": case_name,
        "case_dirs": case_dirs,
        "csv_path": csv_path,
        "figure_paths": generated_figures,
        "positive_p_stop_time": positive_p.stop_time,
        "max_abs_error": max_abs_error,
    }
