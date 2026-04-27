from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from bosonic_dissipation.plotting import (  # noqa: E402
    DEFAULT_PLOT_ORDER,
    DIFFERENCE_PLOT_SPECS,
    DISPLAY_NAME_MAP,
    SCALAR_PLOT_SPECS,
    apply_scientific_notation,
    build_state_title_text,
    get_difference_style,
    get_plot_profile,
    get_style,
    load_grouped_csv_frames,
    parse_metadata_value,
    set_exact_reference_ylim,
    transform_time_values,
)
from bosonic_dissipation.two_site_comparison import plot_two_site_csv  # noqa: E402


OUTPUT_ROOT = PROJECT_ROOT / "!!to latex" / "t_over_gamma_regenerated"
SINGLE_SITE_OUTPUT = OUTPUT_ROOT / "single_site_gamma1_u0_n1"
TWO_SITE_OUTPUT = OUTPUT_ROOT / "two_site_mean"

SINGLE_SITE_CSV_DIR = PROJECT_ROOT / "outputs" / "01_part1_u0_numParticles1_method_comparison" / "csv"
TWO_SITE_U1_CSV = PROJECT_ROOT / "outputs" / "two_site_comparison" / "u1_gamma1_j1_t5" / "data" / "two_site_mean_comparison.csv"
TWO_SITE_U0P1_CSV = PROJECT_ROOT / "outputs" / "two_site_comparison" / "u0p1_gamma1_j1_t5" / "data" / "two_site_mean_comparison.csv"


def regenerate_single_site_figures() -> list[Path]:
    profile = get_plot_profile("t_over_gamma")
    SINGLE_SITE_OUTPUT.mkdir(parents=True, exist_ok=True)

    grouped_frames = load_grouped_csv_frames(SINGLE_SITE_CSV_DIR)
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

        def save_plot(fig: plt.Figure, prefix: str) -> Path:
            output_path = SINGLE_SITE_OUTPUT / f"{state_label}_{prefix}_{group_suffix}.svg"
            fig.savefig(output_path, format="svg", bbox_inches="tight")
            plt.close(fig)
            generated_paths.append(output_path)
            return output_path

        for column_name, prefix, y_label, title_label in SCALAR_PLOT_SPECS:
            fig, ax = plt.subplots(figsize=(7.5, 5.2))
            has_data = False
            for method_name in plot_order:
                frame = cropped_frames[method_name]
                if column_name not in frame.columns:
                    continue
                has_data = True
                ax.plot(
                    transform_time_values(frame["time"], gamma_value=gamma_value, plot_profile=profile),
                    frame[column_name],
                    label=DISPLAY_NAME_MAP.get(method_name, method_name),
                    **get_style(method_name),
                )
            ax.set_xlabel(profile.x_label)
            ax.set_ylabel(y_label)
            ax.set_title(rf"{title_label} vs {profile.title_x_label} ({state_title_text}, {interaction_text})")
            ax.legend()
            ax.grid(True, alpha=0.35)
            if exact_reference is not None and column_name in exact_reference.columns:
                set_exact_reference_ylim(ax, exact_reference[column_name])
            apply_scientific_notation(ax)
            if has_data:
                save_plot(fig, prefix)
            else:
                plt.close(fig)

        exact_frame = cropped_frames.get("exact")
        if exact_frame is None:
            continue

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
                    transform_time_values(frame["time"], gamma_value=gamma_value, plot_profile=profile),
                    difference_values,
                    label=DISPLAY_NAME_MAP.get(method_name, method_name),
                    **get_difference_style(method_name),
                )
            diff_ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
            diff_ax.set_xlabel(profile.x_label)
            diff_ax.set_ylabel(y_label)
            diff_ax.set_title(rf"{title_label} vs {profile.title_x_label} ({state_title_text}, {interaction_text})")
            diff_ax.legend()
            diff_ax.grid(True, alpha=0.35)
            apply_scientific_notation(diff_ax)
            if has_data:
                save_plot(diff_fig, prefix)
            else:
                plt.close(diff_fig)

    return generated_paths


def make_two_site_figure(
    *,
    csv_path: Path,
    output_path: Path,
    site_index: int,
    gamma_value: float,
) -> Path:
    return plot_two_site_csv(
        csv_path=csv_path,
        output_path=output_path,
        site_index=site_index,
        gamma_value=gamma_value,
        plot_profile="t_over_gamma",
    )


def regenerate_two_site_figures() -> list[Path]:
    TWO_SITE_OUTPUT.mkdir(parents=True, exist_ok=True)
    generated_paths: list[Path] = []

    for csv_path, stem in [
        (TWO_SITE_U1_CSV, ""),
        (TWO_SITE_U0P1_CSV, "_u0p1"),
    ]:
        for site_index in [0, 1]:
            output_path = TWO_SITE_OUTPUT / f"two_site_mean_n{site_index + 1}_science{stem}.png"
            generated_paths.append(
                make_two_site_figure(
                    csv_path=csv_path,
                    output_path=output_path,
                    site_index=site_index,
                    gamma_value=1.0,
                )
            )

    return generated_paths


def main() -> None:
    single_site = regenerate_single_site_figures()
    two_site = regenerate_two_site_figures()
    print(f"Generated {len(single_site)} single-site figures in {SINGLE_SITE_OUTPUT}")
    print(f"Generated {len(two_site)} two-site figures in {TWO_SITE_OUTPUT}")


if __name__ == "__main__":
    main()
