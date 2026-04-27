# Project Structure

## Permanent code

- `src/bosonic_dissipation/`
  Core simulation and plotting code.
- `scripts/`
  Reusable entry points for running data generation and replotting.

## Generated outputs

- `outputs/01_*` to `outputs/17_*`
  Numbered Part 1 result bundles. Each bundle keeps its own `csv/`, `figures/`, and `benchmark/`.
- `outputs/two_site_comparison/`
  Permanent home for the two-site Density-Matrix vs Positive-P comparison runs.

## Plotting convention

- Use `scripts/replot_part1_bundle.py` to replot one existing bundle from its saved CSV files.
- Use `scripts/replot_part1_range.py` to replot a numbered range such as `01` to `16`.
- Use plot profiles to keep styles separate:
  - `time`
  - `t_over_gamma`
- Replot into a profile subfolder such as `figures/t_over_gamma/` so one plotting style does not overwrite another.

## Two-site comparison

- `scripts/run_two_site_comparison.py`
  Permanent runner for the two-site comparison cases.
- Example permanent output folders:
  - `outputs/two_site_comparison/u1_gamma1_j1_t5/`
  - `outputs/two_site_comparison/u0p1_gamma1_j1_t5/`
