"""Fig. D.1 of the paper: force and state error and chi^2 of the IEKS against the state diffusion sigma_x^2.

One line per grid density kappa = 2 ... 128, with `theta_u` at the truth, scored at the held-out midpoints
between the observations. The IEKS re-linearises the residual row at its smoothed mean, `N_ITERS` iterations
at a time. It runs by sigma_x continuation (`lfcal.paper.iterated.continuation`): from the loose end of the window
down, each point warm-started from its converged larger neighbour, then an upward repair pass. A point that
has not converged is continued in further chunks of `N_ITERS` iterations, at most `MAX_CHUNKS` in all.

The figure, `sigma_x_sweep_all.pdf`, shows every system and marks the maximiser of I(U;Y|Z), taken from
`force_information` on the same grids, and the `ell_ODE` pick.

    uv run python experiments/sigma_x_sweep.py duffing fhn lorenz96   # results/sigma_x_sweep/
    uv run python experiments/sigma_x_sweep.py --figure               # figures/sigma_x_sweep_all.pdf
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import force_information
import jax.numpy as jnp
import numpy as onp
from ode_filters import iterated_smoother, rts_smoother
from scipy import stats

from lfcal.moments import marginal_variances
from lfcal.paper import benchmark, plotting, systems
from lfcal.paper.io import read_csv, write_csv
from lfcal.paper.iterated import STEP_TOL, continuation, make_runner

RESULTS = Path(__file__).resolve().parents[1] / "results" / "sigma_x_sweep"
SYSTEMS = ("duffing", "fhn", "lorenz96")
KAPPAS = (2, 4, 8, 16, 32, 64, 128)
N_ITERS = 5
#: at most this many chunks of `N_ITERS` iterations per point
MAX_CHUNKS = 6
SCORE_AT = "midpoints"


def path(name, part="", root=RESULTS):
    """`{name}.csv` (one row per kappa and sigma_x), `{name}_picks.csv` (one row per kappa and pick),
    `{name}_meta.csv` (one row: the run's settings)."""
    return Path(root) / f"{name}{part}.csv"


def save(name, d, root=RESULTS):
    """`compute(name)` as three CSV tables."""
    Path(root).mkdir(parents=True, exist_ok=True)
    write_csv(path(name, root=root),
              [{"kappa": int(k), "N": d["N"][k], **r} for k in d["sweep"] for r in d["sweep"][k]])
    write_csv(path(name, "_picks", root), [{"kappa": int(k), "pick": pick, **r}
                                           for k, picks in d["picks"].items() for pick, r in picks.items()])
    write_csv(path(name, "_meta", root), [{"system": d["system"], **d["meta"]}])


def load(name, root=RESULTS):
    """Inverse of `save`: the sweep and the picks keyed by `str(kappa)`."""
    rows = read_csv(path(name, root=root))
    kappas = list(dict.fromkeys(r["kappa"] for r in rows))
    sweep = {str(k): [{f: v for f, v in r.items() if f not in ("kappa", "N")} for r in rows if r["kappa"] == k]
             for k in kappas}
    picks = {}
    for r in read_csv(path(name, "_picks", root)):
        picks.setdefault(str(r.pop("kappa")), {})[r.pop("pick")] = r
    meta = read_csv(path(name, "_meta", root))[0]
    return {"system": meta.pop("system"), "kappas": kappas,
            "sigmas": [r["sigma_x"] for r in sweep[str(kappas[0])]],
            "N": {str(k): next(r["N"] for r in rows if r["kappa"] == k) for k in kappas}, "sweep": sweep,
            "picks": picks, "meta": meta}


# ---------------------------------------------------------------- the sweep
def _grid_pick(rows, key):
    """The row with the largest `key` among the finite ones."""
    vals = onp.array([r[key] if r["ok"] else -onp.inf for r in rows])
    return dict(rows[int(onp.argmax(vals))])


def _converged(o):
    return bool(o["ok"][-1]) and float(o["iterations"]["step"][-1]) <= STEP_TOL


def _continued(run, sigma_x, log_theta_u, o):
    """`o`, continued from its last trajectory in chunks of `N_ITERS` iterations until it converges or
    diverges, at most `MAX_CHUNKS` chunks in all."""
    for _ in range(MAX_CHUNKS - 1):
        if _converged(o) or not bool(o["ok"][-1]):
            break
        o = run(sigma_x, log_theta_u, init=o["traj"], keep_traj=True)
    return o


def _row(sigma_x, o):
    """The sweep row of one point: its last iteration's scores and log-likelihoods."""
    it = o["iterations"]
    return {"sigma_x": sigma_x, "ok": bool(o["ok"][-1]),
            **{f: it[f][-1] for f in ("rel_u", "chi2_u", "rel_x", "chi2_x", "ll_obs", "ll_ode", "step")},
            "start": o["start"]}


def _sweep_cell(system, kappa, sigmas):
    """`(N, rows)`: the IEKS at every sigma_x of `sigmas` on the grid of density `kappa`, by continuation."""
    log_theta_u = system.log_theta_u_true
    run = make_runner(system, kappa, systems.EK1, N_ITERS)

    def solve(sigma_x, init):
        return _continued(run, sigma_x, log_theta_u, run(sigma_x, log_theta_u, init=init, keep_traj=True))

    outs = continuation(solve, sigmas, _converged)
    return run.N, [_row(s, o) for s, o in zip(sigmas, outs, strict=True)]


def compute(name, kappas=KAPPAS):
    """The IEKS over the system's sweep window at every kappa of `kappas`, with the `ell_OBS` and `ell_ODE`
    picks per kappa and the run's settings."""
    system = systems.build(name, score_at=SCORE_AT)
    sigmas = systems.sweep_window(name, system)
    N, sweep, picks = {}, {}, {}
    for kappa in kappas:
        t0 = time.time()
        N[str(kappa)], rows = _sweep_cell(system, kappa, sigmas)
        sweep[str(kappa)] = rows
        picks[str(kappa)] = {"ll_obs": _grid_pick(rows, "ll_obs"), "ll_ode": _grid_pick(rows, "ll_ode")}
        pick = picks[str(kappa)]["ll_obs"]
        n_converged = sum(r["ok"] and r["step"] <= STEP_TOL for r in rows)
        print(f"{name} kappa={kappa:3d} (N={N[str(kappa)]}, {time.time() - t0:.0f} s): ell_obs pick "
              f"{pick['sigma_x']:.3g} -> IEKS force error {pick['rel_u']:.4f} (chi2_u {pick['chi2_u']:.2f}); "
              f"converged {n_converged}/{len(sigmas)}", flush=True)
    res = {"system": name, "kappas": list(kappas), "sigmas": sigmas, "N": N, "sweep": sweep, "picks": picks}
    meta = {"n_iters": N_ITERS, "score_at": SCORE_AT, "q_x": int(system.q_x), "continuation": True,
            "step_tol": STEP_TOL, "max_chunks": MAX_CHUNKS}
    return {**res, "meta": {**meta, **neff_at_pick(system, res)}}


def chi2_effective_sample_size(z):
    """Satterthwaite's `N_eff = N / (1 + 2 sum_l (1 - l/N) rho_l^2)` of the mean of `z^2`, the lag sum
    truncated at the first non-positive autocorrelation."""
    z = onp.asarray(z, dtype=float)
    z = z[onp.isfinite(z)]
    n = z.size
    if n < 8:
        return float(n)
    v = z - z.mean()
    denom = float(v @ v)
    if denom <= 0 or not onp.isfinite(denom):
        return float(n)
    total = 0.0
    for lag in range(1, min(n - 1, n // 4) + 1):
        rho = float(v[:-lag] @ v[lag:] / denom)
        if rho <= 0.0:
            break
        total += (1.0 - lag / n) * rho ** 2
    return float(n / (1.0 + 2.0 * total))


def neff_at_pick(system, res, kappa_ref=32):
    """`N_eff` of the standardised force and state errors at the IEKS `ell_obs` pick on the largest kappa
    up to `kappa_ref`. It sets the state's chi^2 band of the figure, `chi2_{N_eff} / N_eff`; the force's is
    `benchmark.chi2_band`."""
    kappa = max(k for k in res["kappas"] if k <= kappa_ref)
    sigma_x = res["picks"][str(kappa)]["ll_obs"]["sigma_x"]
    grid = system.discretize(kappa, systems.EK1)
    prior, r = system.filter(grid, jnp.asarray(float(sigma_x)), jnp.asarray(system.log_theta_u_true))
    res = iterated_smoother(r.m[0], r.P_sqr[0], prior, grid.ode_measure, grid.ts, n_iters=N_ITERS,
                            init=rts_smoother(r).m[1:], obs_model=grid.obs_model)
    m_s, var, idx = onp.asarray(res.m), onp.asarray(marginal_variances(res.P_sqr)), grid.score_idx
    z_u = (m_s[idx, grid.u_col] - grid.u_true[idx]) / onp.sqrt(var[idx, grid.u_col])
    z_x = (m_s[idx, grid.x_col] - grid.x_true[idx]) / onp.sqrt(var[idx, grid.x_col])
    return {"n_eff_u": chi2_effective_sample_size(z_u), "n_eff_x": chi2_effective_sample_size(z_x),
            "n_points": len(idx), "n_eff_kappa": int(kappa), "n_eff_sigma_x": float(sigma_x)}


# ---------------------------------------------------------------- figures

CHANNELS = (("u", "force $U$"), ("x", "state $X$"))
METRICS = (("rel", "relative error"), ("chi2", r"$\chi^2$"))
#: the picks marked: the maximiser of I(U;Y|Z) (`_with_info_picks`) and the ell_ODE pick
PICKS = ("max_info", "ll_ode")


def _drawn(o):
    """A point is drawn if its IEKS is finite and converged."""
    return o["ok"] and o["step"] <= STEP_TOL


def _drawn_kappas(res):
    """The grids with a converged point. Each curve is drawn where the IEKS converged, so a coarse grid on
    which it diverges somewhere shows only part of the window."""
    return [k for k in res["kappas"] if k >= 2 and any(_drawn(o) for o in res["sweep"][str(k)])]


def _info_pick(res, kappa, sigma):
    """The scores at `sigma` on the grid `kappa`, read off the drawn points by log-log interpolation in
    sigma_x^2; NaN outside them."""
    log_s = onp.log(onp.asarray(res["sigmas"]))
    pick = {"sigma_x": sigma, "step": 0.0}
    for f in ("rel_u", "chi2_u", "rel_x", "chi2_x"):
        y = onp.array([r[f] if _drawn(r) else onp.nan for r in res["sweep"][kappa]])
        m = onp.isfinite(y) & (y > 0)
        inside = m.sum() > 1 and log_s[m].min() <= onp.log(sigma) <= log_s[m].max()
        pick[f] = float(onp.exp(onp.interp(onp.log(sigma), log_s[m], onp.log(y[m])))) if inside else onp.nan
    return {**pick, "ok": bool(onp.isfinite([pick["rel_u"], pick["rel_x"]]).all())}


def _with_info_picks(name, res):
    """`res` with the maximiser of I(U;Y|Z) per kappa added to its picks (`max_info`).

    The maximiser comes from `force_information`, on the same grids and sigma_x^2 window and refined between
    its points; its scores are read off this sweep (`_info_pick`).
    """
    info = force_information.load(name)
    picks = {k: {**p, "max_info": _info_pick(res, k, info[k]["picks"]["max_info"]["sigma_x"])} if k in info else p
             for k, p in res["picks"].items()}
    return {**res, "picks": picks}


def _panel(ax, res, metric, ch, kappas, colour):
    """`metric` (`rel`, `chi2`) of channel `ch` against sigma_x^2, one line per kappa over the converged
    points, with the picks marked. A chi^2 panel shows the 95 % band of a calibrated posterior: the force's
    as in the comparison figures (`benchmark.chi2_band`), the state's from `N_eff` (`neff_at_pick`)."""
    sigmas = onp.asarray(res["sigmas"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    for k in kappas:
        ax.plot(sigmas, [o[f"{metric}_{ch}"] if _drawn(o) else onp.nan for o in res["sweep"][str(k)]],
                color=colour[k], lw=1.2)
    for k in kappas:
        for pick in PICKS:
            o = res["picks"][str(k)].get(pick)
            if o is not None and _drawn(o) and onp.isfinite(o[f"{metric}_{ch}"]):
                ax.plot([o["sigma_x"]], [o[f"{metric}_{ch}"]], **plotting.pick_style(pick, colour[k]), zorder=6)
    if metric == "chi2":
        plotting.hline(ax, 1.0, ls=plotting.DASHED, lw=1.0)
        if ch == "u":
            lo, hi = benchmark.chi2_band(res["system"], int(res["meta"]["n_points"]))
        else:
            n_eff = res["meta"][f"n_eff_{ch}"]
            lo, hi = stats.chi2.ppf([0.025, 0.975], n_eff) / n_eff
        plotting.shade_band(ax, lo, hi)
        ax.set_ylim(0.1, 100)
        ax.set_yticks([0.1, 1.0, 10.0, 100.0])
    else:
        plotting.autoscale_log(ax, cap=2.0)
        plotting.ensure_log_ticks(ax)


def _finish(fig, axes, kappas, norm, cmap):
    """sigma_x^2 labels, the grid-density colour bar and the picks' legend."""
    for ax in axes[-1]:
        ax.set_xlabel(r"$\sigma_x^2$")
    fig.align_ylabels(axes[:, 0])
    plotting.grid_density_colorbar(fig, axes, kappas, cmap, norm, pad=0.02, aspect=30)
    handles = [plotting.pick_handle("max_info", r"max $I(U;Y \mid Z)$"), plotting.pick_handle("ll_ode")]
    plotting.shared_legend(fig, handles=handles, ncol=2)


def figure_all(names=SYSTEMS, out="sigma_x_sweep_all"):
    """Every system in a column; rows: force error, state error, force chi^2, state chi^2. A system without
    results is left out."""
    names = [n for n in names if path(n).exists()]
    if missing := [n for n in SYSTEMS if n not in names]:
        print(f"sigma_x sweep: no results for {', '.join(missing)} (`sigma_x_sweep.py {' '.join(missing)}`)")
    results = {n: _with_info_picks(n, load(n)) for n in names}
    drawn = {n: _drawn_kappas(results[n]) for n in names}
    kappas = sorted({k for ks in drawn.values() for k in ks})
    cmap, norm, colour = plotting.grid_density_colours(kappas)
    rows = (("rel", "u", "force error"), ("rel", "x", "state error"), ("chi2", "u", r"force $\chi^2$"),
            ("chi2", "x", r"state $\chi^2$"))
    fig, axes = plotting.subplots(len(rows), len(names), sharex="col", squeeze=False, height_to_width_ratio=0.4,
                                  grid=True)
    for j, name in enumerate(names):
        for i, (metric, ch, label) in enumerate(rows):
            _panel(axes[i, j], results[name], metric, ch, drawn[name], colour)
            if j == 0:
                axes[i, j].set_ylabel(label)
        axes[0, j].set_title(plotting.SYSTEM_NAMES[name])
    _finish(fig, axes, kappas, norm, cmap)
    print(f"wrote {plotting.save(fig, out)}")


#: the figure, by name (`--figure NAME`)
FIGURES = {"sigma_x_sweep_all": figure_all}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("systems", nargs="*", choices=SYSTEMS, default=[], help="compute these systems' sweeps")
    parser.add_argument("--figure", nargs="*", choices=FIGURES, metavar="NAME",
                        help=f"draw these figures from results/ (default: all of {', '.join(FIGURES)})")
    args = parser.parse_args()
    for name in args.systems:
        save(name, compute(name))
        print(f"{name}: wrote {path(name)} (+ _picks, _meta)")
    if args.figure is not None or not args.systems:
        for name in args.figure or FIGURES:
            FIGURES[name]()


if __name__ == "__main__":
    main()
