"""What the data say about the force given the ODE, on the IEKS-converged linearisation.

`ode_filters.information_criteria` gives `I(U; Z)`, the paper's criterion `I(U; Y | Z) = I(U; Y, Z) - I(U; Z)`
(Eqs. `mi` and `info-data`), and the directed information `D` of the data about the force. `scan_cell`
sweeps `sigma_x` over a window on one grid and picks the maximisers of these criteria.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as onp
from ode_filters import affine_filter, conditional_loglik, information_criteria, iterated_smoother, rts_smoother

from lfcal.moments import marginal_variances, rms_change
from lfcal.paper.iterated import STEP_TOL, continuation
from lfcal.paper.scores import posterior_scores

__all__ = ["N_ITERS", "make_runner", "scan_cell"]

#: IEKS iterations per point (with 4, FitzHugh--Nagumo stays unconverged at small sigma_x)
N_ITERS = 30


def _refine_argmax(sigmas, values):
    """The maximiser of `values` over `sigmas`, refined by a parabola in `log sigma_x` through the grid
    argmax and its two neighbours. Returns the maximiser and whether it sits on the window's edge."""
    s = onp.log(onp.asarray(sigmas, float))
    v = onp.asarray(values, float)
    finite = onp.isfinite(v)
    j = int(onp.nanargmax(onp.where(finite, v, -onp.inf)))
    if j == 0 or j == len(v) - 1 or not (finite[j - 1] and finite[j + 1]):
        return float(onp.exp(s[j])), True
    x, y = s[j - 1:j + 2], v[j - 1:j + 2]
    c = onp.polyfit(x, y, 2)
    if c[0] >= 0:
        return float(onp.exp(s[j])), False
    return float(onp.exp(onp.clip(-c[1] / (2 * c[0]), x[0], x[2]))), False


def make_runner(system, kappa, correction, n_iters):
    """`run(sigma_x, log_theta_u, init=None)`: the criteria on the converged linearisation, jitted.

    One run: the joint EK1 pass and its smoothed mean (or `init`) start `n_iters` IEKS iterations. A final
    data pass at the converged linearisation is scored, with the informations at its `H_n` and the
    conditional evidence `logp = log p(Y | Z)`: `ll_ode + ll_obs` of the pass minus `ll_z = log p(Z)` of
    the same pass without the data. A point is `ok` when everything, the starting EK1 pass included, is
    finite and the posterior mean has not diverged.
    """
    grid = system.discretize(kappa, correction)

    @jax.jit
    def core(sigma_x, log_theta_u, init=None):
        prior, r = system.filter(grid, sigma_x, log_theta_u)
        m0, P0 = r.m[0], r.P_sqr[0]
        res = iterated_smoother(m0, P0, prior, grid.ode_measure, grid.ts, n_iters=n_iters,
                                init=rts_smoother(r).m[1:] if init is None else init, obs_model=grid.obs_model,
                                diagnostics=lambda it: rms_change(it.m_smooth[1:], it.previous_point, grid.x_col))
        traj = res.m[1:]
        surrogate = grid.linearize(traj)
        out = affine_filter(m0, P0, prior, surrogate, grid.ts, obs_model=grid.obs_model)
        free = affine_filter(m0, P0, prior, surrogate, grid.ts)
        smoothed = rts_smoother(out)
        info = information_criteria(m0, P0, prior, surrogate, grid.ts, grid.obs_model)
        return {**posterior_scores(grid, smoothed.m, marginal_variances(smoothed.P_sqr)),
                "ll_ode": out.log_likelihood, "ll_obs": out.log_likelihood_obs, "ll_z": free.log_likelihood,
                "logp": conditional_loglik(out, free), "ll_obs_ek1": r.log_likelihood_obs,
                "step": res.diagnostics[-1], "I_Z": info.I_UZ, "I_YZ": info.I_UY_given_Z, "D": info.D}, traj

    def run(sigma_x, log_theta_u, init=None):
        raw, traj = core(jnp.asarray(float(sigma_x)), jnp.asarray(log_theta_u), init)
        out = {k: float(v) for k, v in raw.items()}
        out["ok"] = bool(all(onp.isfinite(v) for v in out.values()) and not grid.has_diverged(out["max_abs_x"]))
        out.update(sigma_x=float(sigma_x), warm=init is not None, traj=traj)
        return out

    run.N, run.K = grid.N, grid.K
    return run


def _converged(o):
    return o["ok"] and o["step"] <= STEP_TOL


#: the fields of a sweep point that `scan_cell` records, and those of a pick
ROW_FIELDS = ("I_Z", "I_YZ", "D", "rel_u", "chi2_u", "ll_obs", "ll_ode", "logp", "step", "ok", "warm")
PICK_FIELDS = ("rel_u", "chi2_u", "I_Z", "I_YZ", "D", "ll_obs", "logp")


def scan_cell(system, kappa, window, log_theta_u, correction, n_iters=N_ITERS):
    """One grid density: every `sigma_x` of `window`, then the picks of each criterion.

    The window runs by continuation from its loose end (`iterated.continuation`). The picks are the
    maximisers of `I(U;Y|Z)` (`max_info`), of `D` (`max_di`) and of `log p(Y | Z)` (`max_logp`), and the
    point of largest `ell_OBS` (`ell_obs`). Each pick is re-run at its own `sigma_x`, warm from the nearest
    converged point (the one above first) or cold if that does not converge.
    """
    run = make_runner(system, kappa, correction, n_iters)
    outs = continuation(lambda s, init: run(s, log_theta_u, init), [float(s) for s in window], _converged)
    rows = [{"sigma_x": float(s), **{f: o[f] for f in ROW_FIELDS}} for s, o in zip(window, outs, strict=True)]
    ok_rows = [r for r in rows if r["ok"]]
    best = min(r["rel_u"] for r in ok_rows)
    sigmas = [r["sigma_x"] for r in rows]
    s_info, info_on_edge = _refine_argmax(sigmas, [r["I_YZ"] if r["ok"] else onp.nan for r in rows])
    s_di, di_on_edge = _refine_argmax(sigmas, [r["D"] if r["ok"] else onp.nan for r in rows])
    targets = {"max_info": s_info, "max_di": s_di,
               "ell_obs": max(ok_rows, key=lambda r: r["ll_obs"])["sigma_x"],
               "max_logp": max(ok_rows, key=lambda r: r["logp"])["sigma_x"]}
    converged = [i for i, o in enumerate(outs) if _converged(o)]

    def rerun(s):
        near = sorted(converged, key=lambda i: (window[i] < s, abs(onp.log(window[i] / s))))
        o = run(s, log_theta_u, outs[near[0]]["traj"]) if near else None
        return o if o is not None and _converged(o) else run(s, log_theta_u)

    picks = {}
    for pick, s in targets.items():
        o = rerun(s)
        picks[pick] = {"sigma_x": float(s), **{f: o[f] for f in PICK_FIELDS}, "cost": o["rel_u"] / best,
                       "step": o["step"]}
    picks["max_info"]["on_window_edge"] = bool(info_on_edge)
    picks["max_di"]["on_window_edge"] = bool(di_on_edge)
    return {"kappa": kappa, "N": run.N, "K": run.K, "n_iters": n_iters, "rows": rows, "picks": picks,
            "best_rel_u": best}
