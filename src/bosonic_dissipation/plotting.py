from __future__ import annotations

from math import isclose, sqrt
from pathlib import Path

import matplotlib.pyplot as plt
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


def apply_scientific_notation(ax) -> None:
    ax.ticklabel_format(style="sci", axis="both", scilimits=(-3, 3), useMathText=True)


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
) -> list[Path]:
    configure_plot_style()
    figure_dir.mkdir(parents=True, exist_ok=True)

    grouped_frames = load_grouped_csv_frames(csv_dir, method_names=method_names)
    generated_paths: list[Path] = []

    for (state_label, group_suffix), method_frames in grouped_frames.items():
        initial_state_type = next(iter(method_frames.values())).attrs["initial_state_type"]
        state_title_text = build_state_title_text(state_label, initial_state_type)
        gamma_text = parse_metadata_value(group_suffix, "gamma")
        if gamma_text is None:
            raise ValueError(f"Could not parse gamma from grouped suffix '{group_suffix}'")

        shared_end_time = min(frame["time"].max() for frame in method_frames.values())
        cropped_frames = {name: frame[frame["time"] <= shared_end_time].copy() for name, frame in method_frames.items()}
        plot_order = [method_name for method_name in DEFAULT_PLOT_ORDER if method_name in cropped_frames]

        def save_plot(fig, prefix: str) -> Path:
            output_path = figure_dir / f"{state_label}_{prefix}_{group_suffix}.svg"
            fig.savefig(output_path, format="svg", bbox_inches="tight")
            plt.close(fig)
            generated_paths.append(output_path)
            return output_path

        mean_fig, mean_ax = plt.subplots(figsize=(7.5, 5.2))
        for method_name in plot_order:
            frame = cropped_frames[method_name]
            mean_ax.plot(
                frame["time"],
                frame["mean_particle_number"],
                label=DISPLAY_NAME_MAP.get(method_name, method_name),
                **get_style(method_name),
            )
        mean_ax.set_xlabel("Time")
        mean_ax.set_ylabel("Mean Particle Number")
        mean_ax.set_title(rf"Mean vs Time ({state_title_text}, $\gamma={gamma_text}$)")
        mean_ax.legend()
        mean_ax.grid(True, alpha=0.35)
        apply_scientific_notation(mean_ax)
        save_plot(mean_fig, "mean")

        variance_fig, variance_ax = plt.subplots(figsize=(7.5, 5.2))
        for method_name in plot_order:
            frame = cropped_frames[method_name]
            variance_ax.plot(
                frame["time"],
                frame["variance"],
                label=DISPLAY_NAME_MAP.get(method_name, method_name),
                **get_style(method_name),
            )
        variance_ax.set_xlabel("Time")
        variance_ax.set_ylabel("Variance")
        variance_ax.set_title(rf"Variance vs Time ({state_title_text}, $\gamma={gamma_text}$)")
        variance_ax.legend()
        variance_ax.grid(True, alpha=0.35)
        apply_scientific_notation(variance_ax)
        save_plot(variance_fig, "variance")

        exact_frame = cropped_frames.get("exact")
        if exact_frame is None:
            if require_exact_for_difference:
                continue
        else:
            exact_frame = exact_frame[["time", "mean_particle_number", "variance"]].reset_index(drop=True)

            mean_diff_fig, mean_diff_ax = plt.subplots(figsize=(7.5, 5.2))
            for method_name in plot_order:
                if method_name == "exact":
                    continue
                frame = cropped_frames[method_name]
                mean_difference = frame["mean_particle_number"].to_numpy() - exact_frame["mean_particle_number"].to_numpy()
                mean_diff_ax.plot(
                    frame["time"],
                    mean_difference,
                    label=DISPLAY_NAME_MAP.get(method_name, method_name),
                    **get_difference_style(method_name),
                )
            mean_diff_ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
            mean_diff_ax.set_xlabel("Time")
            mean_diff_ax.set_ylabel("Mean difference from exact")
            mean_diff_ax.set_title(rf"Mean Difference vs Time ({state_title_text}, $\gamma={gamma_text}$)")
            mean_diff_ax.legend()
            mean_diff_ax.grid(True, alpha=0.35)
            apply_scientific_notation(mean_diff_ax)
            save_plot(mean_diff_fig, "meanDifference")

            variance_diff_fig, variance_diff_ax = plt.subplots(figsize=(7.5, 5.2))
            for method_name in plot_order:
                if method_name == "exact":
                    continue
                frame = cropped_frames[method_name]
                variance_difference = frame["variance"].to_numpy() - exact_frame["variance"].to_numpy()
                variance_diff_ax.plot(
                    frame["time"],
                    variance_difference,
                    label=DISPLAY_NAME_MAP.get(method_name, method_name),
                    **get_difference_style(method_name),
                )
            variance_diff_ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
            variance_diff_ax.set_xlabel("Time")
            variance_diff_ax.set_ylabel("Variance difference from exact")
            variance_diff_ax.set_title(rf"Variance Difference vs Time ({state_title_text}, $\gamma={gamma_text}$)")
            variance_diff_ax.legend()
            variance_diff_ax.grid(True, alpha=0.35)
            apply_scientific_notation(variance_diff_ax)
            save_plot(variance_diff_fig, "varDifference")

    return generated_paths
