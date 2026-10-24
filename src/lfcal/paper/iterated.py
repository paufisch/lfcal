"""The IEKS, and the `sigma_x` continuation that carries a converged linearisation along a window.

The IEKS is the joint filter with the residual row re-linearised at its own smoothed mean
(`ode_filters.iterated_smoother`). One run of `make_runner(system, kappa, correction, n_iters)` does
`n_iters + 1` iterations on the grid of density `kappa`: linearise the ODE residual along the current
trajectory, run the affine filter and smoother, and take the smoothed mean as the next trajectory. It starts
from the smoothed mean of one joint EK1 pass, or from a given trajectory (`init`, which `continuation` uses).

Each iteration also runs the data-free filter on the same linearisation, for `ll_Z` and for Fenrir's
`log p(Y | Z)` from its backward conditionals. An iteration counts as `ok` only if these are finite too.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as onp
from ode_filters import affine_filter, fenrir_loglik, iterated_smoother, rts_smoother

from lfcal.moments import marginal_variances, rms_change
from lfcal.paper.scores import posterior_scores

__all__ = ["STEP_TOL", "continuation", "make_runner"]

#: the linearisation counts as converged when the last iteration moves the trajectory by less than this (rms)
STEP_TOL = 1e-3


def make_runner(system, kappa, correction, n_iters):
    """`run(sigma_x, log_theta_u, init=None, keep_traj=False)` on the grid of density `kappa`, jitted.

    A run returns `iterations` (per iteration: the scores, the log-likelihoods, and `step`, the rms change
    of the trajectory's scored component) and `ok` per iteration, and with `keep_traj` also `traj`, the
    last trajectory. `run.N` and `run.K` are the grid's steps and observations.
    """
    grid = system.discretize(kappa, correction)

    @jax.jit
    def core(sigma_x, log_theta_u, init=None):
        prior, r = system.filter(grid, sigma_x, log_theta_u)
        m0, P0 = r.m[0], r.P_sqr[0]

        def record(it):
            """An iteration's scores and log-likelihoods, the data-free pass on its linearisation included."""
            free = affine_filter(m0, P0, prior, it.surrogate, grid.ts)
            return {"ll_Z": free.log_likelihood, "ll_fenrir": fenrir_loglik(free, grid.obs_model),
                    "ll_ode": it.filter_result.log_likelihood, "ll_obs": it.filter_result.log_likelihood_obs,
                    "step": rms_change(it.m_smooth[1:], it.previous_point, grid.x_col),
                    **posterior_scores(grid, it.m_smooth, marginal_variances(it.P_smooth_sqr))}

        res = iterated_smoother(m0, P0, prior, grid.ode_measure, grid.ts, n_iters=n_iters + 1,
                                init=rts_smoother(r).m[1:] if init is None else init, obs_model=grid.obs_model,
                                diagnostics=record)
        return res.diagnostics, res.m[1:]

    def run(sigma_x, log_theta_u, init=None, keep_traj=False):
        its, traj = core(jnp.asarray(float(sigma_x)), jnp.asarray(log_theta_u),
                         None if init is None else jnp.asarray(init))
        its = {k: onp.asarray(v, dtype=float) for k, v in its.items()}
        ok = [all(onp.isfinite(v[i]) for v in its.values()) and not grid.has_diverged(its["max_abs_x"][i])
              for i in range(n_iters + 1)]
        out = {"sigma_x": float(sigma_x), "start": "data" if init is None else "warm",
               "iterations": {k: v.tolist() for k, v in its.items()}, "ok": [bool(v) for v in ok]}
        if keep_traj:
            out["traj"] = traj
        return out

    run.N, run.K = grid.N, grid.K
    return run


def continuation(solve, sigmas, converged):
    """Every `sigma_x` of `sigmas` (ascending), each started from a converged neighbour where possible.

    `solve(sigma_x, init)` returns a result with its trajectory under `"traj"`, starting cold from the EK1
    smoothed mean when `init` is None. A downward pass starts each point from the converged trajectory of
    the next-larger `sigma_x`, or cold at the top, after a break, or when the warm start does not converge.
    An upward pass then repairs each unconverged point from its converged lower neighbour.
    """
    outs = [None] * len(sigmas)
    prev = None
    for i in reversed(range(len(sigmas))):
        warm = solve(sigmas[i], prev) if prev is not None else None
        outs[i] = warm if warm is not None and converged(warm) else solve(sigmas[i], None)
        prev = outs[i]["traj"] if converged(outs[i]) else None
    prev = None
    for i in range(len(sigmas)):
        if not converged(outs[i]) and prev is not None:
            repaired = solve(sigmas[i], prev)
            if converged(repaired):
                outs[i] = repaired
        prev = outs[i]["traj"] if converged(outs[i]) else None
    return outs
