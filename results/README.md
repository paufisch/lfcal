# Results

The paper's results, as `make results` computes them; the [README](../README.md) says how to draw the figures from them and how to recompute them.

Scores and settings are CSV tables with one row per record, time series `.npz` archives. Floats are written at full precision; an empty cell is a field the record does not have. Grid densities are in `kappa` and observation counts in `K`. The fitted hyperparameters are `log_sigma_x2, log_xi_u, log_ell_u` (`dlog_xi_u, dlog_ell_u` are their offsets from the truth), and `chi2_u_lo, chi2_u_hi` is the 95 % interval of the force chi^2 under a calibrated posterior. Records with `ok = False` come from a diverged filter and are not plotted. Each script's `load` reads its tables back (`lfcal.paper.io`).

| Files | Contents |
|---|---|
| `sigma_x_sweep/{system}.csv` | the IEKS, one row per (kappa, sigma_x); `{system}_picks.csv` one row per (kappa, pick); `{system}_meta.csv` the settings |
| `failure_directions.csv`, `.npz` | one row per `diffusion` (`over_enforced`, `max_info`, `under_enforced`): scores of the joint posterior, of `p(U \| Z)` (`data_free_*`) and of `pi(X, U)` (`pi_*`); the archive holds the time grid, truth, data and the marginals `diffusions.{diffusion}.{joint,data_free,pi,prior}.{u_mean,u_sd,x_mean,x_sd}` |
| `force_information/{system}.csv` | one row per (kappa, sigma_x); `{system}_picks.csv` one row per (kappa, pick); `{system}_cells.csv` one row per kappa |
| `objectives/{system}_s{draw}_{method}.csv` | one method's fit at K0 (`obs`, `mi`, `di`, `fenrir_ieks`, `classical_rk4`), with the grid it reports |
| `posteriors/{system}_s{draw}_{method}.npz` | mean and sd of force and scored state on the method's grid, with truth and data: the comparison's wrong fits and their references |
| `robustness/{system}_{kind}_s{draw}_{method}.csv` | one method's fit at K0 with a truth force of `kind` (`matern32`, `sine`, `step`; `matern` on SIR) |
| `failures_robustness/` | posteriors, as in `posteriors/`, of the examples of `fig:robustness-examples` |
