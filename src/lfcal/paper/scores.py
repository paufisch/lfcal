"""A posterior's scores against the truth, and the chi^2 interval a calibrated posterior would show.

Scores compare a posterior with the truth (`posterior_scores`), at the nodes a `ScoredGrid` scores. They are
computed in the same jitted pass as the posterior itself (`ScoredGrid.posterior_extras`). `chi2_interval`
gives the 95 % interval of the chi^2 that a calibrated posterior would show, from the posterior's own
correlation (`smoother_cross_covariance`) and Imhof's formula (`weighted_chi2_interval`); `chi2_intervals`
gives it for a fit on a `Calibration`'s grid.

Square-root convention of `ode_filters`: `P = P_sqr.T @ P_sqr`.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as onp
from ode_filters import rts_smoother
from scipy.integrate import quad
from scipy.optimize import brentq

from lfcal.moments import covariances

__all__ = [
    "channel_scores",
    "chi2_interval",
    "chi2_intervals",
    "posterior_scores",
    "smoother_cross_covariance",
    "weighted_chi2_cdf",
    "weighted_chi2_interval",
]


# ------------------------------------------------------------------ scores
def channel_scores(err, var, rms, channel):
    """`rel_{channel}`, the rms error over the truth's rms, and `chi2_{channel}`, the mean squared
    standardised error."""
    return {f"rel_{channel}": jnp.sqrt(jnp.mean(err ** 2)) / rms, f"chi2_{channel}": jnp.mean(err ** 2 / var)}


def posterior_scores(grid, m_s, var):
    """Scores of the scored state component and of the force at `grid.score_idx`, and `max_abs_x`."""
    idx = grid.score_idx
    err_x = m_s[idx, grid.x_col] - grid.x_true[idx]
    err_u = m_s[idx, grid.u_col] - grid.u_true[idx]
    return {**channel_scores(err_x, var[idx, grid.x_col], grid.rms_x, "x"),
            **channel_scores(err_u, var[idx, grid.u_col], grid.rms_u, "u"),
            "max_abs_x": jnp.max(jnp.abs(m_s[:, grid.x_col]))}


def smoother_cross_covariance(P, G, items):
    """`C[i, j] = Cov(a_i s_{n_i}, a_j s_{n_j})` under an RTS smoother.

    `P` are the smoother's marginal covariances at nodes 0..N and `G` its backward gains, so that
    `Cov(s_n, s_m) = G_n Cov(s_{n+1}, s_m)` for `n < m`. `items` is a list of `(node n_i, vector a_i)`.
    The recursion runs backwards over the nodes and carries `W[:, i] = Cov(s_n, s_{n_i}) a_i` for every
    item already reached.
    """
    P, G = onp.asarray(P), onp.asarray(G)
    at_node = {}
    for i, (n, a) in enumerate(items):
        at_node.setdefault(int(n), []).append((i, onp.asarray(a)))
    M = len(items)
    W, C, reached = onp.zeros((P.shape[1], M)), onp.zeros((M, M)), onp.zeros(M, dtype=bool)
    for n in range(G.shape[0], -1, -1):
        for i, a in at_node.get(n, []):
            W[:, i] = P[n] @ a
            reached[i] = True
        for i, a in at_node.get(n, []):
            C[i, reached] = a @ W[:, reached]
        if n > 0:
            W[:, reached] = G[n - 1] @ W[:, reached]
    return onp.triu(C) + onp.triu(C, 1).T


def weighted_chi2_cdf(x, lam):
    """`P(sum_j lam_j z_j^2 <= x)` for independent `z_j ~ N(0, 1)` and weights `lam_j > 0`.

    Imhof (1961): one integral over the inverted characteristic function, evaluated by quadrature.
    """
    lam = onp.asarray(lam, dtype=float)

    def integrand(u):
        if u == 0.0:
            return 0.5 * (lam.sum() - x)
        theta = 0.5 * onp.sum(onp.arctan(lam * u)) - 0.5 * x * u
        log_rho = 0.25 * onp.sum(onp.log1p((lam * u) ** 2))
        return onp.sin(theta) / (u * onp.exp(log_rho))

    return 0.5 - quad(integrand, 0.0, onp.inf, limit=2000, epsabs=1e-10)[0] / onp.pi


def weighted_chi2_interval(lam, n, probs=(0.025, 0.975)):
    """The `probs` quantiles of `(1/n) sum_j lam_j z_j^2`, each solved from `weighted_chi2_cdf`.

    Weights below `1e-12` of their sum are dropped.
    """
    lam = onp.asarray(lam, dtype=float)
    lam = lam[lam > 1e-12 * lam.sum()]
    hi = lam.sum() + 40.0 * onp.sqrt(2.0 * onp.sum(lam ** 2))
    return tuple(brentq(lambda x, p=p: weighted_chi2_cdf(x, lam) - p, 1e-10, hi, xtol=1e-10) / n for p in probs)


def chi2_interval(C):
    """The 95 % interval of `mean(z^2)` that a calibrated posterior with cross covariance `C` would give.

    The standardised errors are then `z ~ N(0, corr(C))`, so `M mean(z^2)` is a sum of chi^2_1 variables
    weighted by the correlation's eigenvalues. Also returns `n_eff = M^2 / sum lambda^2`.
    """
    sd = onp.sqrt(onp.diag(C))
    lam = onp.clip(onp.linalg.eigvalsh(C / onp.outer(sd, sd)), 0.0, None)
    M = len(lam)
    lo, hi = weighted_chi2_interval(lam, M)
    return {"lo": float(lo), "hi": float(hi), "n_eff": float(M ** 2 / onp.sum(lam ** 2))}


def chi2_intervals(cal, phi, traj, channels=("u", "x")):
    """Per channel (force `u`, scored state `x`), the 95 % interval of the chi^2 at the fit `(phi, traj)` on
    the `Calibration` `cal` if its posterior were calibrated, from the posterior correlation across the scored
    nodes (`chi2_interval`)."""
    out = cal.data_pass(jnp.asarray(phi), cal.linearize(jnp.asarray(traj)))
    P, G = onp.asarray(covariances(rts_smoother(out).P_sqr)), onp.asarray(out.G_back)
    cols = {"u": cal.grid.u_col, "x": cal.grid.x_col}
    unit = onp.eye(P.shape[1])
    return {ch: chi2_interval(smoother_cross_covariance(P, G, [(n, unit[cols[ch]]) for n in cal.grid.score_idx]))
            for ch in channels}
