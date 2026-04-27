# Temporary Density-Matrix vs Positive-P Comparison

This folder contains standalone scratch code for comparing density-matrix and Positive-P mean occupations for:

- 1 site with initial Fock state `|1>`
- 2 sites with initial Fock state `|1,0>`

Parameters:

- `U = 1`
- `gamma = 1`
- `J = 1`

Notes:

- This code is intentionally isolated from the main project `src/`.
- The 2-site case uses open-boundary nearest-neighbor hopping.
- With a single initial particle and only loss, the onsite interaction does not affect the exact dynamics, because the system never reaches double occupancy. The script still keeps `U = 1` so the comparison matches the requested parameter set.

Run with:

```powershell
py -3 .\outputs\tmp_density_vs_positivep_mean\run_comparison.py
```
