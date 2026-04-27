from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
import scienceplots


OUTPUT_DIR = Path(__file__).resolve().parent
DATA_DIR = OUTPUT_DIR / "data"
FIGURE_DIR = OUTPUT_DIR / "figures"

U = 0.1
GAMMA = 1.0
J = 1.0
TOTAL_TIME = 5.0
DT = 0.002
NUM_SAMPLES = 50_000
SEED = 123456
MAX_TRAJECTORY_MAGNITUDE = 1e150
MAX_SITE_OCCUPATION = 1.05
MIN_SITE_OCCUPATION = -0.05


@dataclass(slots=True)
class MeanResult:
    time_values: np.ndarray
    mean_occupations: np.ndarray
    completed_full_time: bool = True
    stop_time: float | None = None


def _ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


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


def make_site_figure(
    *,
    density: MeanResult,
    positive_p: MeanResult,
    site_index: int,
    output_path: Path,
) -> None:
    with plt.style.context(["science", "no-latex", "grid"]):
        fig, ax = plt.subplots(figsize=(6.6, 4.6))
        ax.plot(
            density.time_values,
            density.mean_occupations[:, site_index],
            color="#0b3c5d",
            linewidth=2.6,
            label="Density Matrix",
        )
        ax.plot(
            positive_p.time_values,
            positive_p.mean_occupations[:, site_index],
            color="#c0392b",
            linewidth=2.0,
            linestyle="--",
            label="Positive-P",
        )
        ax.set_title(f"Two Sites: Mean n{site_index + 1}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Mean Occupation")
        ax.set_xlim(0.0, TOTAL_TIME)
        stability_text = f"Positive-P stable until\nt = {positive_p.stop_time:.3f} s"
        if positive_p.stop_time is not None and positive_p.stop_time < TOTAL_TIME:
            ax.axvline(
                positive_p.stop_time,
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
        fig.savefig(output_path, dpi=220, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    _ensure_dirs()

    two_site_density = build_density_matrix_result(
        num_sites=2,
        initial_occupations=[1, 0],
        interaction_strength=U,
        gamma=GAMMA,
        hopping=J,
        total_time=TOTAL_TIME,
        dt=DT,
    )
    two_site_positive_p = simulate_positive_p_chain(
        num_sites=2,
        initial_occupations=[1, 0],
        interaction_strength=U,
        gamma=GAMMA,
        hopping=J,
        total_time=TOTAL_TIME,
        dt=DT,
        num_samples=NUM_SAMPLES,
        seed=SEED + 1,
    )

    save_two_site_csv(DATA_DIR / "two_site_mean_comparison_t5.csv", two_site_density, two_site_positive_p)

    make_site_figure(
        density=two_site_density,
        positive_p=two_site_positive_p,
        site_index=0,
        output_path=FIGURE_DIR / "two_site_mean_n1_science.png",
    )
    make_site_figure(
        density=two_site_density,
        positive_p=two_site_positive_p,
        site_index=1,
        output_path=FIGURE_DIR / "two_site_mean_n2_science.png",
    )

    matched_density = MeanResult(
        time_values=two_site_density.time_values[: two_site_positive_p.time_values.size],
        mean_occupations=two_site_density.mean_occupations[: two_site_positive_p.time_values.size],
    )
    two_site_error = np.max(np.abs(matched_density.mean_occupations - two_site_positive_p.mean_occupations))

    print(f"Saved outputs to: {OUTPUT_DIR}")
    print(f"Positive-P stable through, 2 sites: {two_site_positive_p.stop_time:.3f}")
    print(f"Max absolute mean error, 2 sites: {two_site_error:.6f}")


if __name__ == "__main__":
    main()
