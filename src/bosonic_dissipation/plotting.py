from __future__ import annotations

from dataclasses import dataclass
from math import isclose, sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from .part1_runner import sanitize_number


STYLE_MAP = {
    "exact": {"color": "black", "linestyle": "-", "linewidth": 2.3, "marker": None, "zorder": 5},
    "densityMatrix": {
        "color": "tab:blue",
        "linestyle": "--",
        "linewidth": 2.6,
        "marker": "^",
        "markevery": 400,
        "markersize": 3.6,
        "zorder": 4,
    },
    "monteCarlo": {
        "color": "tab:orange",
        "linestyle": "-.",
        "linewidth": 2.5,
        "marker": "o",
        "markevery": 350,
        "markersize": 3.5,
        "zorder": 3,
    },
    "positiveP": {
        "color": "tab:green",
        "linestyle": ":",
        "linewidth": 2.7,
        "marker": "s",
        "markevery": 450,
        "markersize": 3.2,
        "zorder": 2,
    },
}
DEFAULT_STYLE = {"linewidth": 1.8, "linestyle": "-", "marker": None, "zorder": 1}
DISPLAY_NAME_MAP = {
    "exact": "Exact",
    "densityMatrix": "Density Matrix",
    "monteCarlo": "Monte Carlo",
    "positiveP": "Positive-P",
}
DEFAULT_PLOT_ORDER = ["exact", "densityMatrix", "monteCarlo", "positiveP"]
IGNORED_GROUP_PART_PREFIXES = ("numOfSamples", "hilbertSize")
SCALAR_PLOT_SPECS = [
    ("mean_particle_number", "mean", "Mean Particle Number", "Mean"),
    ("variance", "variance", "Variance", "Variance"),
    ("g1_magnitude", "g1", r"$|g_1|$", r"First-Order Coherence $|g_1|$"),
    ("g2", "g2", r"$g_2$", r"Second-Order Coherence $g_2$"),
]
DIFFERENCE_PLOT_SPECS = [
    ("mean_particle_number", "meanDifference", "Mean difference from exact", "Mean Difference"),
    ("variance", "varDifference", "Variance difference from exact", "Variance Difference"),
    ("g1_magnitude", "g1Difference", r"$|g_1|$ difference from exact", r"First-Order Coherence Difference $|g_1|$"),
    ("g2", "g2Difference", r"$g_2$ difference from exact", r"Second-Order Coherence Difference $g_2$"),
]
U_COMPARISON_STYLE_MAP = {
    "0": {"color": "#0b3c5d", "linestyle": "-", "linewidth": 2.6},
    "1": {"color": "#c0392b", "linestyle": "--", "linewidth": 2.6},
}


@dataclass(frozen=True, slots=True)
class PlotProfile:
    name: str
    x_label: str
    title_x_label: str
    normalize_time_by_gamma: bool = False


PLOT_PROFILES = {
    "time": PlotProfile(
        name="time",
        x_label="Time",
        title_x_label="Time",
        normalize_time_by_gamma=False,
    ),
    "t_over_gamma": PlotProfile(
        name="t_over_gamma",
        x_label=r"$t/\gamma$",
        title_x_label=r"$t/\gamma$",
        normalize_time_by_gamma=True,
    ),
}


def configure_plot_style() -> None:
    try:
        import scienceplots  # noqa: F401

        plt.style.use(["science", "no-latex"])
    except ImportError:
        pass

    plt.rcParams["text.usetex"] = False
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
        }
    )


def get_style(method_name: str) -> dict:
    style = DEFAULT_STYLE.copy()
    style.update(STYLE_MAP.get(method_name, {}))
    return style


def get_difference_style(method_name: str) -> dict:
    style = get_style(method_name)
    style["marker"] = None
    style["linestyle"] = "-"
    style.pop("markevery", None)
    style.pop("markersize", None)
    return style


def get_plot_profile(profile: str | PlotProfile) -> PlotProfile:
    if isinstance(profile, PlotProfile):
        return profile
    try:
        return PLOT_PROFILES[profile]
    except KeyError as exc:
        available_profiles = ", ".join(sorted(PLOT_PROFILES))
        raise ValueError(f"Unknown plot profile '{profile}'. Available profiles: {available_profiles}") from exc


def transform_time_values(time_values, *, gamma_value: float, plot_profile: PlotProfile):
    time_array = np.asarray(time_values, dtype=float)
    if plot_profile.normalize_time_by_gamma:
        if abs(gamma_value) <= 1e-12:
            raise ValueError("Cannot use the 't_over_gamma' profile when gamma is zero.")
        return time_array / gamma_value
    return time_array


def apply_scientific_notation(ax) -> None:
    ax.ticklabel_format(style="sci", axis="both", scilimits=(-3, 3), useMathText=True)


def set_exact_reference_ylim(ax, exact_values) -> None:
    exact_array = pd.Series(exact_values).astype(float).to_numpy()
    finite_values = exact_array[np.isfinite(exact_array)]
    if finite_values.size == 0:
        return

    y_min = float(np.min(finite_values))
    y_max = float(np.max(finite_values))
    value_range = y_max - y_min
    if value_range <= 0.0:
        reference_scale = max(abs(y_min), abs(y_max), 1.0)
        padding = 0.1 * reference_scale
    else:
        padding = max(0.08 * value_range, 1e-12)
    ax.set_ylim(y_min - padding, y_max + padding)


def parse_csv_metadata(csv_path: Path) -> dict:
    stem_parts = csv_path.stem.split("_")
    method_name = stem_parts[0]
    state_label = stem_parts[1]
    if state_label.startswith("fock"):
        initial_state_type = "fock"
    elif state_label.startswith("coherent"):
        initial_state_type = "coherent"
    else:
        raise ValueError(f"Could not parse state label from {csv_path.name}")

    group_parts = [
        part
        for part in stem_parts[2:]
        if not any(part.startswith(prefix) for prefix in IGNORED_GROUP_PART_PREFIXES)
    ]
    return {
        "method_name": method_name,
        "state_label": state_label,
        "initial_state_type": initial_state_type,
        "group_suffix": "_".join(group_parts),
        "group_suffix_without_u": "_".join([part for part in group_parts if not part.startswith("u")]),
    }


def parse_metadata_value(group_suffix: str, prefix: str) -> str | None:
    for part in group_suffix.split("_"):
        if part.startswith(prefix):
            return part[len(prefix):].replace("p", ".")
    return None


def build_state_title_text(state_label: str, initial_state_type: str) -> str:
    if initial_state_type == "fock":
        num_of_particles_text = state_label[len("fock") :]
        return rf"$|n={num_of_particles_text}\rangle$"
    num_of_particles_text = state_label[len("coherent") :]
    num_of_particles = float(num_of_particles_text)
    nearest_integer = round(num_of_particles)
    if isclose(num_of_particles, nearest_integer):
        integer_particles = int(nearest_integer)
        integer_sqrt = int(round(sqrt(integer_particles)))
        if integer_sqrt * integer_sqrt == integer_particles:
            alpha_text = str(integer_sqrt)
        else:
            alpha_text = rf"\sqrt{{{integer_particles}}}"
    else:
        alpha_text = f"{sqrt(num_of_particles):g}"
    return rf"$|\alpha={alpha_text}\rangle$"


def load_grouped_csv_frames(csv_dir: Path, *, method_names: list[str] | tuple[str, ...] | None = None) -> dict[tuple[str, str], dict[str, pd.DataFrame]]:
    grouped_frames: dict[tuple[str, str], dict[str, pd.DataFrame]] = {}
    for csv_path in sorted(csv_dir.glob("*.csv")):
        metadata = parse_csv_metadata(csv_path)
        if method_names is not None and metadata["method_name"] not in method_names:
            continue
        frame = pd.read_csv(csv_path)
        frame.attrs.update(metadata)
        group_key = (metadata["state_label"], metadata["group_suffix"])
        grouped_frames.setdefault(group_key, {})[metadata["method_name"]] = frame
    if not grouped_frames:
        raise ValueError(f"No matching CSV files found in {csv_dir}")
    return grouped_frames


def generate_figures_from_csvs(
    *,
    csv_dir: Path,
    figure_dir: Path,
    method_names: list[str] | tuple[str, ...] | None = None,
    require_exact_for_difference: bool = True,
    plot_profile: str | PlotProfile = "time",
) -> list[Path]:
    configure_plot_style()
    figure_dir.mkdir(parents=True, exist_ok=True)
    resolved_profile = get_plot_profile(plot_profile)

    grouped_frames = load_grouped_csv_frames(csv_dir, method_names=method_names)
    generated_paths: list[Path] = []

    for (state_label, group_suffix), method_frames in grouped_frames.items():
        initial_state_type = next(iter(method_frames.values())).attrs["initial_state_type"]
        state_title_text = build_state_title_text(state_label, initial_state_type)
        interaction_strength_text = parse_metadata_value(group_suffix, "u")
        gamma_text = parse_metadata_value(group_suffix, "gamma")
        if gamma_text is None:
            raise ValueError(f"Could not parse gamma from grouped suffix '{group_suffix}'")
        gamma_value = float(gamma_text)
        interaction_text = (
            rf"$U={interaction_strength_text}$, $\gamma={gamma_text}$"
            if interaction_strength_text is not None
            else rf"$\gamma={gamma_text}$"
        )

        shared_end_time = min(frame["time"].max() for frame in method_frames.values())
        cropped_frames = {name: frame[frame["time"] <= shared_end_time].copy() for name, frame in method_frames.items()}
        plot_order = [method_name for method_name in DEFAULT_PLOT_ORDER if method_name in cropped_frames]
        exact_reference = cropped_frames.get("exact")

        def save_plot(fig, prefix: str) -> Path:
            output_path = figure_dir / f"{state_label}_{prefix}_{group_suffix}.svg"
            fig.savefig(output_path, format="svg", bbox_inches="tight")
            plt.close(fig)
            generated_paths.append(output_path)
            return output_path

        def plot_scalar_observable(
            *,
            column_name: str,
            prefix: str,
            y_label: str,
            title_label: str,
        ) -> None:
            fig, ax = plt.subplots(figsize=(7.5, 5.2))
            has_data = False
            for method_name in plot_order:
                frame = cropped_frames[method_name]
                if column_name not in frame.columns:
                    continue
                has_data = True
                ax.plot(
                    transform_time_values(
                        frame["time"],
                        gamma_value=gamma_value,
                        plot_profile=resolved_profile,
                    ),
                    frame[column_name],
                    label=DISPLAY_NAME_MAP.get(method_name, method_name),
                    **get_style(method_name),
                )
            ax.set_xlabel(resolved_profile.x_label)
            ax.set_ylabel(y_label)
            ax.set_title(rf"{title_label} vs {resolved_profile.title_x_label} ({state_title_text}, {interaction_text})")
            ax.legend()
            ax.grid(True, alpha=0.35)
            if exact_reference is not None and column_name in exact_reference.columns:
                set_exact_reference_ylim(ax, exact_reference[column_name])
            apply_scientific_notation(ax)
            if has_data:
                save_plot(fig, prefix)
            else:
                plt.close(fig)

        for column_name, prefix, y_label, title_label in SCALAR_PLOT_SPECS:
            plot_scalar_observable(
                column_name=column_name,
                prefix=prefix,
                y_label=y_label,
                title_label=title_label,
            )

        exact_frame = cropped_frames.get("exact")
        if exact_frame is None:
            if require_exact_for_difference:
                continue
        else:
            exact_columns = ["time"]
            for column_name, _, _, _ in SCALAR_PLOT_SPECS:
                if column_name in exact_frame.columns:
                    exact_columns.append(column_name)
            exact_frame = exact_frame[exact_columns].reset_index(drop=True)

            for column_name, prefix, y_label, title_label in DIFFERENCE_PLOT_SPECS:
                if column_name not in exact_frame.columns:
                    continue
                diff_fig, diff_ax = plt.subplots(figsize=(7.5, 5.2))
                has_data = False
                for method_name in plot_order:
                    if method_name == "exact":
                        continue
                    frame = cropped_frames[method_name]
                    if column_name not in frame.columns:
                        continue
                    has_data = True
                    difference_values = frame[column_name].to_numpy() - exact_frame[column_name].to_numpy()
                    diff_ax.plot(
                        transform_time_values(
                            frame["time"],
                            gamma_value=gamma_value,
                            plot_profile=resolved_profile,
                        ),
                        difference_values,
                        label=DISPLAY_NAME_MAP.get(method_name, method_name),
                        **get_difference_style(method_name),
                    )
                diff_ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
                diff_ax.set_xlabel(resolved_profile.x_label)
                diff_ax.set_ylabel(y_label)
                diff_ax.set_title(rf"{title_label} vs {resolved_profile.title_x_label} ({state_title_text}, {interaction_text})")
                diff_ax.legend()
                diff_ax.grid(True, alpha=0.35)
                apply_scientific_notation(diff_ax)
                if has_data:
                    save_plot(diff_fig, prefix)
                else:
                    plt.close(diff_fig)

    return generated_paths


def generate_u_comparison_figures_from_csvs(
    *,
    csv_dir: Path,
    figure_dir: Path,
    method_name: str = "exact",
    interaction_strength_values: tuple[float, ...] = (0.0, 1.0),
    plot_profile: str | PlotProfile = "time",
) -> list[Path]:
    configure_plot_style()
    figure_dir.mkdir(parents=True, exist_ok=True)
    resolved_profile = get_plot_profile(plot_profile)

    grouped_frames = load_grouped_csv_frames(csv_dir, method_names=[method_name])
    interaction_labels = {sanitize_number(value): value for value in interaction_strength_values}
    comparison_groups: dict[tuple[str, str], dict[str, pd.DataFrame]] = {}
    for (state_label, group_suffix), frames in grouped_frames.items():
        frame = frames.get(method_name)
        if frame is None:
            continue
        interaction_strength_text = parse_metadata_value(group_suffix, "u")
        if interaction_strength_text is None or interaction_strength_text not in interaction_labels:
            continue
        group_suffix_without_u = frame.attrs["group_suffix_without_u"]
        comparison_groups.setdefault((state_label, group_suffix_without_u), {})[interaction_strength_text] = frame

    generated_paths: list[Path] = []
    for (state_label, group_suffix_without_u), frames_by_u in comparison_groups.items():
        if len(frames_by_u) < 2:
            continue
        initial_state_type = next(iter(frames_by_u.values())).attrs["initial_state_type"]
        state_title_text = build_state_title_text(state_label, initial_state_type)
        gamma_text = parse_metadata_value(group_suffix_without_u, "gamma")
        if gamma_text is None:
            raise ValueError(f"Could not parse gamma from grouped suffix '{group_suffix_without_u}'")
        gamma_value = float(gamma_text)

        for column_name, prefix, y_label, title_label in (
            ("g1_magnitude", "g1UComparison", r"$|g_1|$", r"First-Order Coherence $|g_1|$"),
            ("g2", "g2UComparison", r"$g_2$", r"Second-Order Coherence $g_2$"),
        ):
            fig, ax = plt.subplots(figsize=(7.5, 5.2))
            has_data = False
            for interaction_strength_text, frame in sorted(frames_by_u.items(), key=lambda item: float(item[0])):
                if column_name not in frame.columns:
                    continue
                has_data = True
                style = U_COMPARISON_STYLE_MAP.get(interaction_strength_text, DEFAULT_STYLE)
                ax.plot(
                    transform_time_values(
                        frame["time"],
                        gamma_value=gamma_value,
                        plot_profile=resolved_profile,
                    ),
                    frame[column_name],
                    label=rf"$U={interaction_strength_text}$",
                    **style,
                )
            ax.set_xlabel(resolved_profile.x_label)
            ax.set_ylabel(y_label)
            ax.set_title(
                rf"{title_label} vs {resolved_profile.title_x_label} for $U=0$ vs $U=1$ ({state_title_text}, $\gamma={gamma_text}$)"
            )
            ax.legend()
            ax.grid(True, alpha=0.35)
            apply_scientific_notation(ax)
            if has_data:
                output_path = figure_dir / f"{method_name}_{state_label}_{prefix}_{group_suffix_without_u}.svg"
                fig.savefig(output_path, format="svg", bbox_inches="tight")
                generated_paths.append(output_path)
            plt.close(fig)

    return generated_paths
