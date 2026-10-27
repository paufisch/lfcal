"""Fig. 1 of the paper: the two failure directions of the state diffusion sigma_x^2.

Duffing with its position observed (K = 32) on the grid kappa = 32, with `theta_u` at the truth, after
`N_ITERS` IEKS iterations. Three diffusions make the three columns: the maximiser of `I(U; Y | Z)` that
`force_information.py` recorded on this grid, and that maximiser times `10^-8` (the ODE over-enforced) and
times `10^6` (under-enforced), `DECADES`. Per diffusion three posteriors are recorded:

  ODE only     `p(X, U | Z)` at the converged linearisation
  pi           `pi(X, U)`, the classical LFM posterior with the PN solve as its forward model; it differs from
               the posterior by the residual likelihood ratio `V(U; Z) = p(U | Z) / p(U)`
  posterior    `p(X, U | Z, Y)`

The figure, `failure_directions_pi_no_ode_only`, shows pi and the posterior, one column per diffusion: the
force (top) and the position with the data (bottom).

    uv run python experiments/failure_directions.py duffing    # results/failure_directions.{csv,npz}
    uv run python experiments/failure_directions.py --figure   # figures/failure_directions_pi_no_ode_only.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import NamedTuple

import force_information
import jax
import jax.numpy as jnp
import numpy as onp
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from ode_filters import AffineSurrogate, JointPrior, affine_filter, iterated_smoother, rts_smoother
from ode_filters.priors import Transition

from lfcal.moments import covariances, marginal_variances, rms_change
from lfcal.paper import plotting, systems
from lfcal.paper.io import read_csv, read_npz, write_csv, write_npz
from lfcal.paper.scores import channel_scores, smoother_cross_covariance

RESULTS = Path(__file__).resolve().parents[1] / "results"
NAME = "failure_directions"
#: per diffusion, the posterior marginals on the filter grid, and the force prior's (stored in the .npz)
SERIES = ("joint", "data_free", "pi", "prior")
TOP = ("system", "kappa", "n_iters")
ARRAYS = ("ts", "u_true", "x_true", "t_obs", "z_obs")
SYSTEM = "duffing"
KAPPA = 32
#: IEKS re-linearisations per diffusion
N_ITERS = 10
#: decades below and above the `I(U; Y | Z)` maximiser of the over- and under-enforced columns
DECADES = (8, 6)
#: the columns: diffusion, heading, how its `sigma_x^2` compares with the maximiser `sigma_x^star`
COLUMNS = (("over_enforced", "Failure: ODE over-enforced", r"$\sigma_x^2 \ll \sigma_x^{\star 2}$"),
           ("max_info", r"\textbf{Ours}", r"$\sigma_x^{\star 2}$ maximizes $I(U;Y \mid Z)$"),
           ("under_enforced", "Failure: ODE under-enforced", r"$\sigma_x^2 \gg \sigma_x^{\star 2}$"))


def save(out, root=RESULTS, name=NAME):
    """`{name}.csv`, one row per diffusion with its settings and scores, and `{name}.npz`, the time grid,
    truth, data and each diffusion's marginals `diffusions.{diffusion}.{series}.{u_mean|u_sd|x_mean|x_sd}`."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rows = [dict({f: out[f] for f in TOP}, diffusion=diffusion, **{f: v for f, v in a.items() if f not in SERIES})
            for diffusion, a in out["diffusions"].items()]
    write_csv(root / f"{name}.csv", rows)
    write_npz(root / f"{name}.npz", dict({f: out[f] for f in ARRAYS},
                                          diffusions={diffusion: {s: a[s] for s in SERIES}
                                                      for diffusion, a in out["diffusions"].items()}))


def load(root=RESULTS, name=NAME):
    """Inverse of `save`."""
    root = Path(root)
    rows, arrays = read_csv(root / f"{name}.csv"), read_npz(root / f"{name}.npz")
    out = dict({f: rows[0][f] for f in TOP}, **{f: arrays[f] for f in ARRAYS}, diffusions={})
    for r in rows:
        diffusion = r.pop("diffusion")
        out["diffusions"][diffusion] = dict({f: v for f, v in r.items() if f not in TOP},
                                            **arrays["diffusions"][diffusion])
    return out


# ------------------------------------------------------------------ the posteriors at one diffusion
class Linearisation(NamedTuple):
    """The converged linearisation of one diffusion, and the joint and data-free smoothers on it."""

    prior: JointPrior            # the joint prior, and its transition over one step
    transition: Transition
    mu0: jnp.ndarray             # the initial law
    P0_sqr: jnp.ndarray
    surrogate: AffineSurrogate   # the residual row linearised along the converged trajectory
    G_free: onp.ndarray          # backward gains of the data-free pass
    m_joint: onp.ndarray         # the joint smoother's means and covariance factors
    P_joint: onp.ndarray
    m_free: onp.ndarray          # the data-free smoother's
    P_free: onp.ndarray


def _ieks(system, grid, sigma_x, log_theta_u):
    """The IEKS at `sigma_x`: `(joint smoother, data-free smoother at the same converged linearisation,
    the last step, the linearisation)`."""
    prior, r = system.filter(grid, jnp.asarray(sigma_x), log_theta_u)
    mu0, P0 = r.m[0], r.P_sqr[0]
    res = iterated_smoother(mu0, P0, prior, grid.ode_measure, grid.ts, n_iters=N_ITERS, init=rts_smoother(r).m[1:],
                            obs_model=grid.obs_model,
                            diagnostics=lambda it: rms_change(it.m_smooth[1:], it.previous_point,
                                                              list(grid.ode_state_cols)))
    surrogate = grid.linearize(res.m[1:])
    data_free = affine_filter(mu0, P0, prior, surrogate, grid.ts)
    m_free, P_free = rts_smoother(data_free)
    # the joint pass at the final linearisation too: pi and its gate live on this one linearisation
    m_joint, P_joint = rts_smoother(affine_filter(mu0, P0, prior, surrogate, grid.ts, obs_model=grid.obs_model))
    lin = Linearisation(prior, prior.transition(grid.h), mu0, P0, surrogate, onp.asarray(data_free.G_back),
                        onp.asarray(m_joint), onp.asarray(P_joint), onp.asarray(m_free), onp.asarray(P_free))
    return ((onp.asarray(res.m), onp.asarray(res.P_sqr)), (onp.asarray(m_free), onp.asarray(P_free)),
            float(res.diagnostics[-1]), lin)


# ------------------------------------------------------------------ pi(X, U)
# `pi(X, U) = p(Y | X) p(X | U, Z) p(U) / p_pi(Y)` on the converged linearisation. It is computed through the
# force entries `v`: the force block at node 0 and the force value at nodes 1..N. The residual rows read only
# the force value, and the consistent initial law couples `X(0)` to the whole block at node 0. Given `v`, the
# state is a linear-Gaussian chain, so
#   (a) `p(v)` is the force prior, propagated;
#   (b) the state-only pass conditioned on `Z` gives `p(Y | v, Z) = N(0; L v + ell, S_y)`, a K-row Gaussian, so
#       `pi(v)` is one K-dimensional Kalman update of `p(v)`;
#   (c) `pi(X | v) = p(X | v, Z, Y)` is the state-only pass with the data, affine in `v`.
# The gate: (b) and (c) with `p(v | Z)` (the data-free joint pass) in place of `p(v)` must reproduce the joint
# posterior's marginals.

class _ForceGiven(NamedTuple):
    """`pi` given the force entries: the data model `p(Y | v, Z) = N(0; L v + ell, S_y)` and the state mean
    `x_offset + x_gain v` with variance `x_var` given `v` and all data."""

    L: onp.ndarray
    ell: onp.ndarray
    S_y: onp.ndarray
    x_offset: onp.ndarray
    x_gain: onp.ndarray
    x_var: onp.ndarray


def _force_entries(N, n_force):
    """`(node, component)` of each force entry: the whole force block at node 0, the force value after."""
    return [(0, k) for k in range(n_force)] + [(n, 0) for n in range(1, N + 1)]


def _affine_map(fn, n, chunk=128):
    """`(fn(0), J)` of a map `fn` that is affine on R^n, its linear part assembled in column chunks."""
    f0, linear = jax.linearize(fn, jnp.zeros(n))
    columns = jax.jit(jax.vmap(linear))
    eye = jnp.eye(n)
    return onp.asarray(f0), onp.concatenate([onp.asarray(columns(eye[i:i + chunk])).T for i in range(0, n, chunk)],
                                            axis=1)


def _check_separable(lin, obs_model, u_col):
    """pi needs the residual rows to read only the force value, the data to read no force, and the prior's
    transition not to couple state and force."""
    A, Q_sqr, H = (onp.asarray(v) for v in (lin.transition.A, lin.transition.Q_sqr, lin.surrogate.H))
    if abs(H[:, :, u_col + 1:]).max() > 0 or abs(onp.asarray(obs_model.H)[:, u_col:]).max() > 0:
        raise ValueError("pi: the residual rows read force derivatives, or the data read the force")
    if max(abs(A[:u_col, u_col:]).max(), abs(Q_sqr[:u_col, u_col:]).max(), abs(Q_sqr[u_col:, :u_col]).max()) > 0:
        raise ValueError("pi: the joint prior's transition couples state and force")


def _given_force(grid, lin) -> _ForceGiven:
    """Steps (b) and (c): the data and the state given the force entries, from the state-only passes."""
    obs_model = grid.obs_model
    N, u_col, x_col = grid.N, grid.u_col, grid.x_col
    H, c, R = (onp.asarray(v) for v in lin.surrogate)
    mu0, P0_sqr = onp.asarray(lin.mu0), onp.asarray(lin.P0_sqr)
    n_force = len(mu0) - u_col
    P0 = P0_sqr.T @ P0_sqr
    # the state's initial law given the force block: mean `mu0_x + B0 (block - mu0_u)`, covariance `P0_cond`
    B0 = P0[:u_col, u_col:] @ onp.linalg.inv(P0[u_col:, u_col:])
    P0_cond = P0[:u_col, :u_col] - B0 @ P0[:u_col, u_col:].T
    lam, V = onp.linalg.eigh(0.5 * (P0_cond + P0_cond.T))
    P0_cond_sqr = (V * onp.sqrt(onp.clip(lam, 0.0, None))).T
    H_x, H_u = H[:, :, :u_col], H[:, :, u_col]
    obs_model_x = obs_model._replace(H=obs_model.H[:, :u_col])

    def state_pass(v, obs):
        """The state-only smoother given the force entries `v`, with the data `obs` or without. The joint prior
        is block diagonal, so the state block's transition is the state prior's."""
        out = affine_filter(mu0[:u_col] + B0 @ (v[:n_force] - mu0[u_col:]), P0_cond_sqr, lin.prior.prior_x,
                            AffineSurrogate(H_x, c + H_u * v[n_force:, None], R), grid.ts, obs_model=obs)
        m_s, P_s = rts_smoother(out)
        return m_s, P_s, out.G_back

    n_entries = n_force + N
    v0 = jnp.zeros(n_entries)
    H_obs, c_obs = onp.asarray(obs_model.H)[:, :u_col], onp.asarray(obs_model.c_seq)
    R_obs = onp.asarray(obs_model.R_sqr).T @ onp.asarray(obs_model.R_sqr)
    active = onp.asarray(obs_model.mask).any(axis=1)
    data_rows = [(i + 1, k) for i in range(N) if active[i] for k in range(H_obs.shape[0])]
    nodes = onp.array([n for n, _ in data_rows])
    H_rows = jnp.asarray(onp.stack([H_obs[k] for _, k in data_rows]))
    # (b) p(Y | v, Z)
    _, P_s0, G0 = map(onp.asarray, state_pass(v0, None))
    f0, L = _affine_map(lambda v: jnp.sum(H_rows * state_pass(v, None)[0][nodes], axis=1), n_entries)
    ell = f0 + onp.array([c_obs[n - 1, k] for n, k in data_rows])
    S_y = smoother_cross_covariance(covariances(P_s0), G0, [(n, H_obs[k]) for n, k in data_rows])
    S_y += onp.diag([R_obs[k, k] for _, k in data_rows])
    # (c) pi(X | v) = p(X | v, Z, Y)
    _, P_sY, _ = state_pass(v0, obs_model_x)
    x_var = onp.asarray(covariances(P_sY))[:, x_col, x_col]
    x_offset, x_gain = _affine_map(lambda v: state_pass(v, obs_model_x)[0][:, x_col], n_entries)
    return _ForceGiven(L, ell, S_y, x_offset, x_gain, x_var)


def _condition(given: _ForceGiven, mean, cov, entries):
    """`(force mean, force variance, state mean, state variance)` at the nodes, with the force entries
    `N(mean, cov)` updated by the data."""
    CL = cov @ given.L.T
    gain = onp.linalg.solve(given.L @ CL + given.S_y, CL.T).T
    m, Sigma = mean - gain @ (given.L @ mean + given.ell), cov - gain @ CL.T
    values = [i for i, (_, k) in enumerate(entries) if k == 0]          # the force value at nodes 0..N
    return (m[values], onp.diag(Sigma)[values], given.x_offset + given.x_gain @ m,
            given.x_var + onp.sum((given.x_gain @ Sigma) * given.x_gain, axis=1))


def _prior(lin, u_col, N, entries):
    """Step (a): the force prior's marginals at nodes 0..N, and the mean and covariance of the force entries."""
    A, b, Q_sqr = (onp.asarray(v) for v in (lin.transition.A, lin.transition.b, lin.transition.Q_sqr))
    mu0, P0_sqr = onp.asarray(lin.mu0), onp.asarray(lin.P0_sqr)
    A_u, b_u = A[u_col:, u_col:], b[u_col:]
    Q_u = Q_sqr[u_col:, u_col:].T @ Q_sqr[u_col:, u_col:]
    means, covs = [mu0[u_col:]], [(P0_sqr.T @ P0_sqr)[u_col:, u_col:]]
    for _ in range(N):
        means.append(A_u @ means[-1] + b_u)
        covs.append(A_u @ covs[-1] @ A_u.T + Q_u)
    n_force = A_u.shape[0]
    vectors = [onp.eye(n_force)[k] for _, k in entries]
    # Cov(s_m, s_n) = A_u^(m - n) P_n for m >= n: a forward pass carrying `carried[:, q] = Cov(s_n, s_{n_q}) a_q`
    # for every entry `q` already reached
    cov = onp.zeros((len(entries),) * 2)
    carried, reached = onp.zeros((n_force, len(entries))), onp.zeros(len(entries), dtype=bool)
    at_node = {}
    for q, (n, _) in enumerate(entries):
        at_node.setdefault(n, []).append(q)
    for n in range(N + 1):
        if n > 0:
            carried[:, reached] = A_u @ carried[:, reached]
        new = at_node.get(n, [])
        for q in new:
            carried[:, q] = covs[n] @ vectors[q]
            reached[q] = True
        for q in new:
            cov[q, reached] = vectors[q] @ carried[:, reached]
    cov = onp.tril(cov) + onp.tril(cov, -1).T
    mean = onp.array([means[n][k] for n, k in entries])
    marginals = {"u_mean": [float(m[0]) for m in means], "u_sd": [float(onp.sqrt(P[0, 0])) for P in covs]}
    return mean, cov, marginals


def _gate(grid, lin, given, entries):
    """The largest deviations of (b) and (c) with `p(v | Z)` in place of `p(v)` from the joint posterior's
    force and state marginals; all near zero when pi is computed right."""
    u_col, x_col = grid.u_col, grid.x_col
    D = len(lin.mu0)
    P_free = onp.asarray(covariances(lin.P_free))
    cov = smoother_cross_covariance(P_free, lin.G_free, [(n, onp.eye(D)[u_col + k]) for n, k in entries])
    mean = onp.array([lin.m_free[n, u_col + k] for n, k in entries])
    u_mean, u_var, x_mean, x_var = _condition(given, mean, cov, entries)
    P_joint, m_joint = onp.asarray(covariances(lin.P_joint)), lin.m_joint
    positive = P_joint[:, x_col, x_col] > 0
    return {"u_mean": float(onp.max(abs(u_mean - m_joint[:, u_col]))),
            "u_sd": float(onp.max(abs(onp.sqrt(u_var / P_joint[:, u_col, u_col]) - 1))),
            "x_mean": float(onp.max(abs(x_mean - m_joint[:, x_col]))),
            "x_sd": float(onp.max(abs(onp.sqrt(x_var[positive] / P_joint[positive, x_col, x_col]) - 1)))}


def _pi_marginals(grid, lin: Linearisation):
    """`(pi's marginals, the gate's deviations, the force prior's marginals)` on the converged linearisation."""
    _check_separable(lin, grid.obs_model, grid.u_col)
    entries = _force_entries(grid.N, len(lin.mu0) - grid.u_col)
    given = _given_force(grid, lin)
    prior_mean, prior_cov, prior = _prior(lin, grid.u_col, grid.N, entries)
    u_mean, u_var, x_mean, x_var = _condition(given, prior_mean, prior_cov, entries)
    pi = {"u_mean": u_mean.tolist(), "u_sd": onp.sqrt(u_var).tolist(), "x_mean": x_mean.tolist(),
          "x_sd": onp.sqrt(onp.maximum(x_var, 0.0)).tolist()}
    return pi, _gate(grid, lin, given, entries), prior


# ------------------------------------------------------------------ compute
def _series(grid, m_s, P_s):
    var = onp.asarray(marginal_variances(P_s))
    return {"u_mean": m_s[:, grid.u_col].tolist(), "u_sd": onp.sqrt(var[:, grid.u_col]).tolist(),
            "x_mean": m_s[:, grid.x_col].tolist(), "x_sd": onp.sqrt(var[:, grid.x_col]).tolist()}


def _scores(series, grid):
    idx, out = grid.score_idx, {}
    for ch, truth, rms in (("u", grid.u_true, grid.rms_u), ("x", grid.x_true, grid.rms_x)):
        mean, sd = onp.asarray(series[f"{ch}_mean"])[idx], onp.asarray(series[f"{ch}_sd"])[idx]
        out.update({k: float(v) for k, v in channel_scores(mean - truth[idx], sd ** 2, rms, ch).items()})
    return out


def _diffusion(system, grid, sigma_x, log_theta_u, label):
    """The three posteriors at one diffusion, their scores, and the force prior's marginals."""
    (m, P), (m0, P0), step, lin = _ieks(system, grid, sigma_x, log_theta_u)
    joint, data_free = _series(grid, m, P), _series(grid, m0, P0)
    pi, gate, prior = _pi_marginals(grid, lin)
    pi_scores = _scores(pi, grid)
    res = {"sigma_x": float(sigma_x), "step": step, "joint": joint, "data_free": data_free, "prior": prior,
           **_scores(joint, grid), "data_free_rel_u": _scores(data_free, grid)["rel_u"], "pi": pi, "pi_gate": gate,
           **{f"pi_{k}": v for k, v in pi_scores.items()}}
    print(f"{label:14s} sigma_x={sigma_x:.3g} step={step:.1e}: rel_u {res['rel_u']:.3f} chi2_u {res['chi2_u']:.3g} "
          f"rel_x {res['rel_x']:.3f} chi2_x {res['chi2_x']:.3g} | p(u|z) rel_u {res['data_free_rel_u']:.2f} | "
          f"pi rel_u {pi_scores['rel_u']:.3f} chi2_u {pi_scores['chi2_u']:.3g} | gate {max(gate.values()):.1e}",
          flush=True)
    return res


def compute(decades=DECADES):
    """The three columns: the posteriors at the `I(U; Y | Z)` maximiser and `decades` below and above it."""
    system = systems.build(SYSTEM, score_at="midpoints")
    grid = system.discretize(KAPPA, systems.EK1)
    log_theta_u = jnp.asarray(system.log_theta_u_true)
    pick = force_information.load(SYSTEM)[str(KAPPA)]["picks"]["max_info"]["sigma_x"]
    t_obs, z_obs = system.observations()
    sigmas = {"over_enforced": pick * 10.0 ** -decades[0], "max_info": pick,
              "under_enforced": pick * 10.0 ** decades[1]}
    return {"system": SYSTEM, "kappa": KAPPA, "n_iters": N_ITERS, "ts": grid.ts.tolist(),
            "u_true": grid.u_true.tolist(), "x_true": grid.x_true.tolist(), "t_obs": onp.asarray(t_obs).tolist(),
            "z_obs": z_obs.reshape(-1).tolist(),
            "diffusions": {d: _diffusion(system, grid, s, log_theta_u, d) for d, s in sigmas.items()}}


# ------------------------------------------------------------------ figures
FIGURE = "failure_directions_pi_no_ode_only"
#: the posteriors drawn, back to front, by their legend label: pi outlined under the filled posterior, so that it
#: shows where the two coincide. The ODE-only posterior is recorded, not drawn.
SHOWN = {"pi": r"LFM with PN solver $\pi(X, U)$", "joint": r"posterior $p(X, U \mid Y, Z)$"}
#: the posterior blue and pi purple of the submitted figure, fixed here rather than taken from the method colours
#: (`plotting.METHOD_COLOURS`): pi and the posterior are not methods
COLOURS = {"pi": plotting.POSTERIOR_PURPLE, "joint": plotting.POSTERIOR_BLUE}
#: the width of the mean lines and of pi's outline
MEAN_LW, OUTLINE_LW = 0.9, 0.5
#: the truth: dashed, drawn over the posteriors
TRUTH = {"color": plotting.NEUTRAL, "lw": 0.9, "ls": "--"}
#: the posteriors are drawn as mean and +/- BAND sd
BAND = 2.0


class _Curves(NamedTuple):
    """The figure's time axes and truth: the filter grid (thinned for file size) and the data times, as t/T."""

    thin: slice
    ts: onp.ndarray
    u_true: onp.ndarray
    x_true: onp.ndarray
    t_obs: onp.ndarray


def _curves(d):
    thin = slice(None, None, max(1, len(d["ts"]) // 1500))   # curves thinned for file size; data kept whole
    ts = onp.asarray(d["ts"])
    t0, t1 = d["ts"][0], d["ts"][-1]
    return _Curves(thin, ((ts - ts[0]) / (ts[-1] - ts[0]))[thin], onp.asarray(d["u_true"])[thin],
                   onp.asarray(d["x_true"])[thin], (onp.asarray(d["t_obs"]) - t0) / (t1 - t0))


def _draw_column(axes, diffusion, curves, z_obs):
    """One diffusion: force (top) and position with the data (bottom), each posterior's band and mean."""
    for ax, ch, truth in ((axes[0], "u", curves.u_true), (axes[1], "x", curves.x_true)):
        for key in SHOWN:
            colour = COLOURS[key]
            mean = onp.asarray(diffusion[key][f"{ch}_mean"])[curves.thin]
            sd = onp.asarray(diffusion[key][f"{ch}_sd"])[curves.thin]
            if key == "pi":
                for edge in (mean - BAND * sd, mean + BAND * sd):
                    ax.plot(curves.ts, edge, color=colour, lw=OUTLINE_LW)
            else:
                ax.fill_between(curves.ts, mean - BAND * sd, mean + BAND * sd, color=colour, alpha=plotting.BAND_ALPHA,
                                lw=0)
            ax.plot(curves.ts, mean, color=colour, lw=MEAN_LW)
        ax.plot(curves.ts, truth, **TRUTH)
        if ch == "x":
            ax.plot(curves.t_obs, z_obs, ls="none", marker="o", ms=2.0, color=plotting.NEUTRAL, mec="white", mew=0.3)


def _force_limits(d, curves):
    """The force axis from the truth and the max_info column's posterior band, not from the widest band,
    which would leave the other panels empty; the failure columns may leave the axes."""
    posterior = d["diffusions"]["max_info"]["joint"]
    mean = onp.asarray(posterior["u_mean"])[curves.thin]
    sd = onp.asarray(posterior["u_sd"])[curves.thin]
    lo = min(float(onp.min(curves.u_true)), float(onp.min(mean - BAND * sd)))
    hi = max(float(onp.max(curves.u_true)), float(onp.max(mean + BAND * sd)))
    pad = 0.04 * (hi - lo)
    return lo - pad, hi + pad


def _legend(fig):
    handles = [(Patch(facecolor="none", edgecolor=COLOURS[key], lw=OUTLINE_LW) if key == "pi"
                else Patch(facecolor=COLOURS[key], alpha=plotting.BAND_ALPHA),
                Line2D([], [], color=COLOURS[key], lw=MEAN_LW)) for key in SHOWN]
    handles += [plotting.truth_handle(TRUTH), Line2D([], [], ls="none", marker="o", ms=2.5, color=plotting.NEUTRAL)]
    labels = [*SHOWN.values(), "truth", "observations $Y$"]
    plotting.shared_legend(fig, handles=handles, labels=labels, ncol=len(labels))


def figure():
    """Force (top) and position (bottom) at the three diffusions, as mean and +/- BAND sd, saved as
    `figures/{FIGURE}.pdf`."""
    d = load()
    curves = _curves(d)
    fig, axes = plotting.subplots(2, 3, sharex=True, sharey="row", height_to_width_ratio=0.55)
    for j, (diffusion, head, rule) in enumerate(COLUMNS):
        _draw_column(axes[:, j], d["diffusions"][diffusion], curves, d["z_obs"])
        axes[0, j].set_title(f"{head}\n{rule}")
        if diffusion == "max_info":                      # our column: a heavier frame
            for ax in axes[:, j]:
                for spine in ax.spines.values():
                    spine.set_linewidth(1.75 * spine.get_linewidth())
    lo, hi = _force_limits(d, curves)
    axes[0, 0].set_ylim(lo, hi)
    axes[0, 0].set_ylabel("force $u(t)$")
    x_lo = min(float(onp.min(curves.x_true)), float(onp.min(d["z_obs"])))
    x_hi = max(float(onp.max(curves.x_true)), float(onp.max(d["z_obs"])))
    axes[1, 0].set_ylim(x_lo - 0.15 * (x_hi - x_lo), x_hi + 0.15 * (x_hi - x_lo))
    axes[1, 0].set_ylabel("state $x(t)$")
    for ax in axes[:, 0]:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
    fig.align_ylabels(axes[:, 0])
    for ax in axes[-1]:
        ax.set_xlabel("time $t / T$")
        ax.set_xlim(0.0, 1.0)
    _legend(fig)
    print(f"wrote {plotting.save(fig, FIGURE)}")


#: the figure, by name (`--figure NAME`)
FIGURES = {FIGURE: figure}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("systems", nargs="*", choices=(SYSTEM,), default=[],
                        help="compute the results")
    parser.add_argument("--figure", nargs="*", choices=FIGURES, metavar="NAME",
                        help="draw these figures from results/ (default: all)")
    args = parser.parse_args()
    if args.systems:
        save(compute())
    if args.figure is not None or not args.systems:
        for name in args.figure or FIGURES:
            FIGURES[name]()


if __name__ == "__main__":
    main()
