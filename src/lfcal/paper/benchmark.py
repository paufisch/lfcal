"""The paper's benchmark: three systems with their observation designs, the draws, the methods compared,
and how the fits on one dataset are scored against each other.

A dataset is a (system, draw) pair. Draw 0 is the recorded force path and noise (seed 42), and draw `s` uses
seed `1000 + s`. Within a draw the data on the observation ladder are nested: the observations at every `K`
are a subset of those at the largest `K`, with the same noise at shared times (`models.nested_normal`).

A method's entry is the record of its reported fit (`run_method`): for the algorithm with a criterion J,
`run_algorithm`, the fit of `lfcal.calibrate` on the system scored against its truth; for the classical LFM,
its grid loop. On each dataset the methods are compared by their relative force error over the best method's,
and a fit is wrong when that ratio exceeds `WRONG`.
"""

from __future__ import annotations

import functools
import time
from typing import NamedTuple

import jax.numpy as jnp
import numpy as onp
from scipy.linalg import toeplitz

from lfcal.algorithm import CRITERIA, KAPPAS, calibrate, force_at_obs
from lfcal.moments import covariances
from lfcal.paper import classical, systems
from lfcal.paper.io import phi_fields, phi_u_offsets
from lfcal.paper.scores import chi2_intervals, weighted_chi2_interval

__all__ = ["ALGORITHMS", "CLASSICAL", "COMPARED", "DESIGNS", "DRAWS", "PROCESS_NOISE", "WRONG", "chi2_band",
           "classical_report", "draw_seeds", "error_ratios", "fit_scores", "force_error", "force_off_gain",
           "posterior_marginals", "read_posterior", "run_algorithm", "run_method", "summary", "system_kw"]


class Design(NamedTuple):
    """A system's observation design: `K0` observations for the comparison, and the observation ladder `Ks`,
    across which the observation noise of a draw is nested (`system_kw`), so the data at `K0` are those of
    the ladder."""

    K0: int
    Ks: tuple


DESIGNS = {
    "duffing_accel": Design(K0=128, Ks=(64, 128, 256, 512)),
    "fhn_rough": Design(K0=300, Ks=(150, 300, 600, 1200)),
    "lorenz96_da": Design(K0=100, Ks=(100, 200, 300, 600)),
}
#: the draws of the comparison
DRAWS = tuple(range(20))

#: the methods that run the algorithm, each labelled by its criterion for sigma_x^2
ALGORITHMS = CRITERIA
CLASSICAL = "classical_rk4"
#: the methods of the comparison, in the order of the figures' x-axes
COMPARED = ("obs", "fenrir_ieks", "mi", "di", CLASSICAL)
#: whether the classical model fits a state process noise, per system
PROCESS_NOISE = {"duffing_accel": False, "fhn_rough": False, "lorenz96_da": True, "sir": False}
#: a fit is wrong when its force error exceeds the best method's on its dataset by this factor
WRONG = 1.5
#: the "force switched off" diagnostic: log xi_u lowered by FORCE_OFF, re-linearised to its fixed point
#: (at most RELIN_MAX iterations, until the trajectory moves by less than X_TOL)
FORCE_OFF, RELIN_MAX, X_TOL = 15.0, 10, 1e-6


def draw_seeds(draw):
    """The factory arguments of draw `draw`: the recorded seeds for draw 0, else seed `1000 + draw`."""
    return {"force_seed": 1000 + draw, "obs_seed": 1000 + draw} if draw else {}


def system_kw(name, draw, K):
    """The factory arguments of the benchmark system `name` at draw `draw` with `K` observations: the seeds, and
    the nesting of the noise where `K` lies on the system's observation ladder."""
    design = DESIGNS[name]
    nested = max(design.Ks) % K == 0 and (K % design.K0 == 0 or design.K0 % K == 0)
    return {**({"nest": (design.K0, max(design.Ks))} if nested else {}), **draw_seeds(draw)}


# ------------------------------------------------------------------ the algorithm's records
def fit_scores(cal, phi, traj):
    """Errors and chi^2 of force and state at the held-out midpoints, with the 95 % intervals of the chi^2
    under a calibrated posterior (`scores.chi2_intervals`), and whether the force's chi^2 is inside."""
    post = cal.posterior(jnp.asarray(phi), jnp.asarray(traj))
    out = {k: float(post[k]) for k in ("ll_obs", "rel_u", "rel_x", "chi2_u", "chi2_x")}
    out["ok"] = bool(all(onp.isfinite(out[k]) for k in ("rel_u", "rel_x"))
                     and not cal.grid.has_diverged(post["max_abs_x"]))
    if out["ok"]:
        interval = chi2_intervals(cal, phi, traj)
        u, x = interval["u"], interval["x"]
        out.update(chi2_u_lo=u["lo"], chi2_u_hi=u["hi"], chi2_x_lo=x["lo"], chi2_x_hi=x["hi"],
                   inside=bool(u["lo"] <= out["chi2_u"] <= u["hi"]))
    return out


def force_off_gain(cal, phi, traj, ll_obs):
    """`ell_OBS` of the fit minus `ell_OBS` with the force switched off (log xi_u lowered by `FORCE_OFF`,
    re-linearised to its own fixed point)."""
    phi_off = onp.asarray(phi, float).copy()
    phi_off[1] -= FORCE_OFF
    traj_off = jnp.asarray(traj)
    for _ in range(RELIN_MAX):
        traj_off, step = cal.relinearize(jnp.asarray(phi_off), traj_off)
        if float(step) < X_TOL:
            break
    return ll_obs - force_at_obs(cal, phi_off, traj_off).ll_obs


def run_algorithm(name, K, criterion="obs", kappas=KAPPAS, keep_fit=False, start=None, reset_force=True,
                  **system_kw):
    """The algorithm with `criterion` (`lfcal.calibrate`) on the system `name` with `K` observations, scored
    against the truth at the held-out midpoints.

    Returns the reported grid `kappa`, the fitted hyperparameters and their offsets from the truth, the
    scores (`fit_scores`), `cons` (the consistency statistic), `identified` (a grid was accepted), the
    per-grid `log`, the cost (filter passes weighted by kappa), and `dll`, the gain in `ell_OBS` over the
    force switched off. The reported grid is the accepted one, else the last warm grid that was fitted.

    `keep_fit`: also the reported fit itself, `fit = (Calibration, phi, trajectory)`. `start` and
    `reset_force` go to `lfcal.calibrate`; `system_kw` to the system's factory.
    """
    t0 = time.time()
    system = systems.build(name, score_at="midpoints", n_obs=K, **system_kw)
    fit = calibrate(system, criterion, kappas=kappas, start=start, reset_force=reset_force)
    if fit.failed:
        return {"failed": True, "identified": False, "cost": fit.cost, "log": fit.log, "wall_s": time.time() - t0}
    cal, phi, traj = fit.calibration, fit.phi, fit.traj
    res = {"failed": False, "identified": fit.accepted, "kappa": cal.kappa, "cost": fit.cost,
           "cons": fit.consistency, **phi_fields(phi), **phi_u_offsets(phi, system.log_theta_u_true), "log": fit.log,
           **fit_scores(cal, phi, traj)}
    res["dll"] = force_off_gain(cal, phi, traj, force_at_obs(cal, phi, traj).ll_obs)
    res["wall_s"] = time.time() - t0
    if keep_fit:
        res["fit"] = (cal, phi, traj)
    return res


# ------------------------------------------------------------------ running a method
def classical_report(ladder):
    """The classical model's entry from its grid loop: the accepted grid, else the last fitted one."""
    row = classical.reported_row(ladder)
    if row is None:
        return {"failed": True}
    accepted = bool(row.get("accepted", False))
    return {**{k: v for k, v in row.items() if k != "u_obs"}, "accepted": accepted, "identified": accepted,
            "cost": row.get("cost_cum"), "wall_s": sum(r.get("wall_s", 0.0) for r in ladder["rows"])}


def run_method(name, K, method, *, process_noise=None, **kw):
    """One method's entry on the system `name` with `K` observations: the reported fit of an algorithm (its
    per-grid log dropped), or of the classical model's grid loop. `kw` go to `run_algorithm` (the factory
    arguments, `start`, `reset_force`) or to the system factory. `process_noise`: the classical model's,
    instead of `PROCESS_NOISE[name]`."""
    if method in ALGORITHMS:
        entry = run_algorithm(name, K, method, **kw)
        return {k: v for k, v in entry.items() if k != "log"}
    use_q = PROCESS_NOISE[name] if process_noise is None else process_noise
    return classical_report(classical.ladder(classical.problem(name, K, **kw), use_q))


def summary(entry):
    """One printed line of a method's entry."""
    if entry.get("failed"):
        return entry.get("stop", "FAILED")
    line = f"kappa {entry['kappa']} rel_u {entry['rel_u']:.3f} chi2_u {entry['chi2_u']:.2f}"
    line += (f" cons {entry['cons']:.2f}" if "cons" in entry else "")
    line += f" identified {entry.get('identified', entry.get('accepted'))}"
    return line + (f" ({entry['wall_s']:.0f} s)" if "wall_s" in entry else "")


# ------------------------------------------------------------------ comparing the fits on a dataset
def force_error(entry):
    """The relative force error of an entry, infinite if it is missing, failed or diverged."""
    if entry is None or entry.get("failed") or not entry.get("ok", True) or "rel_u" not in entry:
        return onp.inf
    return entry["rel_u"]


def error_ratios(entries):
    """`{method: force error over the best method's}` on one dataset, and that best error."""
    errors = {m: force_error(e) for m, e in entries.items()}
    best = min(errors.values())
    return {m: e / best for m, e in errors.items()}, best


@functools.cache
def chi2_band(name, K=None):
    """The 95 % interval of the force chi^2 of a calibrated posterior on the system `name` with `K` (default
    `K0`) scored midpoints, its standardised errors correlated as the model's force prior.

    `K chi^2` is then a sum of chi^2_1 variables weighted by the eigenvalues of that correlation
    (`scores.weighted_chi2_interval`). The band depends only on the number of midpoints and the true
    length scale, so it is the same for every fit on the system; each fit's own interval, from its
    posterior's correlation, is recorded as `chi2_u_lo` and `chi2_u_hi`.
    """
    K = K or DESIGNS[name].K0
    system = systems.build(name, score_at="midpoints", n_obs=K)
    prior = system.force.prior(1.0)
    P_sqr = onp.asarray(system.force.stationary_cov_sqr(1.0))
    P = P_sqr.T @ P_sqr
    t0, t1 = system.tspan
    lags = onp.arange(1, K) * (t1 - t0) / K               # the midpoints are equispaced, one interval apart
    rho = onp.concatenate([[1.0], [(onp.asarray(prior.A(float(lag))) @ P)[0, 0] / P[0, 0] for lag in lags]])
    lo, hi = weighted_chi2_interval(onp.clip(onp.linalg.eigvalsh(toeplitz(rho)), 0.0, None), K)
    return float(lo), float(hi)


# ------------------------------------------------------------------ posteriors
#: the number of points of the truth stored with a posterior
N_FINE = 2001


def posterior_marginals(name, K, method, **kw):
    """The reported fit of `method` on the system `name` with `K` observations (factory arguments `kw`): the
    posterior mean and sd of the force (physical units) and of the scored state on the method's grid, with
    the truth, the data, the force error and the grid."""
    system = systems.build(name, n_obs=K, **kw)
    std = system.force_scale                                # the force is filtered in units of its prior std
    if method in ALGORITHMS:
        res = run_algorithm(name, K, method, keep_fit=True, **kw)
        cal, phi, traj = res["fit"]
        post = cal.posterior(jnp.asarray(phi), jnp.asarray(traj))
        g = cal.grid
        m_s, var = onp.asarray(post["m_s"]), onp.asarray(post["var"])
        ts = g.ts
        u_mean, u_sd = m_s[:, g.u_col] * std, onp.sqrt(var[:, g.u_col]) * std
        x_mean, x_sd = m_s[:, g.x_col], onp.sqrt(var[:, g.x_col])
        rel_u, kappa = res["rel_u"], res["kappa"]
    else:
        ladder = classical.ladder(classical.problem(name, K, **kw), PROCESS_NOISE[name], keep_fits=True)
        row = classical.reported_row(ladder)
        lfm, phi, traj = row["fit"]
        m_s, P_s, _, _ = lfm.smoother(jnp.asarray(phi), float(PROCESS_NOISE[name]), jnp.asarray(traj))
        m_s, P = onp.asarray(m_s), onp.asarray(covariances(P_s))
        d, ix = lfm.d, lfm.problem.x_index
        ts = lfm.problem.t0 + lfm.h * onp.arange(lfm.N + 1)
        u_mean, u_sd, x_mean, x_sd = m_s[:, d] * std, onp.sqrt(P[:, d, d]) * std, m_s[:, ix], onp.sqrt(P[:, ix, ix])
        rel_u, kappa = row["rel_u"], row["kappa"]
    t_fine = onp.linspace(*system.tspan, N_FINE)
    t_obs, y = system.observations()
    return {"ts": ts, "u_mean": u_mean, "u_sd": u_sd, "x_mean": x_mean, "x_sd": x_sd, "t_fine": t_fine,
            "u_true": onp.asarray(system.force_path(t_fine)), "x_true": system.solve(t_fine)[:, system.x_index],
            "t_obs": t_obs, "y": onp.asarray(y)[:, 0], "rel_u": rel_u, "kappa": kappa}


def read_posterior(path):
    """A stored posterior (`posterior_marginals`) with its times `ts`, `t_fine` and `t_obs` as the fraction
    `t / T` of the window."""
    with onp.load(path) as z:
        d = dict(z)
    t0, t1 = d["t_fine"][0], d["t_fine"][-1]
    return {**d, **{k: (d[k] - t0) / (t1 - t0) for k in ("ts", "t_fine", "t_obs")}}
