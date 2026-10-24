"""The classical state-space latent force model with an RK4 discretisation, the reference the ODE filter is
compared against.

The state is `(x, U)` with the same Matern-5/2 force prior as the ODE filter (Hartikainen & Sarkka 2012).
The ODE is discretised by RK4 with `kappa` steps per observation interval, the force evaluated at the stage
times from the force state, and the discretisation is treated as exact. Inference is an iterated extended
Kalman smoother. With `use_q` (on Lorenz-96, where exact RK4 fails from the start), a white process noise
acts on the state and its log density is fitted with the force hyperparameters.

Inference follows the algorithm with the criterion `ell_OBS`, without the ODE filter: the linearisation
alternates with L-BFGS-B on `ell_OBS` at the frozen linearisation, and the grid loop `ladder` runs over `kappa = 1, ..., 16`, warm, with
a grid accepted when the force mean at the observation times moved by less than one posterior sd. The
force hyperparameters start where the algorithm starts them, the process-noise density at the observation-noise
variance per observation interval.

Scores are taken at the held-out interval midpoints. On an odd grid there is no node there, so the force is
bridged between its two neighbouring nodes and the state is not scored.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as onp
from ode_filters import AffineModel, FilterResult, Gaussian, rts_smoother
from ode_filters.gaussian import invert, log_likelihood, marginalize, update

from lfcal.algorithm import N_MAX, START, TAU_L, TAU_U, prolong, warm_path
from lfcal.calibration import lbfgs_phi
from lfcal.force import MaternForce
from lfcal.moments import covariances
from lfcal.paper import systems
from lfcal.paper.io import phi_fields
from lfcal.paper.scores import chi2_interval, smoother_cross_covariance
from lfcal.paper.systems import DIVERGED
from lfcal.problem import rk4_step

__all__ = ["KAPPAS", "ClassicalFit", "ClassicalLFM", "ClassicalProblem", "ladder", "problem",
           "reported_row"]

KAPPAS = (1, 2, 4, 8, 16)
#: the force hyperparameters start where the algorithm starts them (fractions of their boxes)
START_U = START[1:]
#: the state's process-noise jitter of the exact model, relative to R / dt_obs
JITTER_REL = 1e-12
#: the cost of a classical filter pass in ODE-filter passes: RK4 evaluates the vector field four times a step
RK4_STAGES = 4


@dataclass(frozen=True)
class ClassicalProblem:
    """What the classical model needs from a system with `K` observations, in first-order form."""

    rhs: Callable                       # the ODE, (y, u, t) -> y', force in model units
    observe: Callable                   # (y, u, t) -> observed channels
    y0: onp.ndarray
    R: onp.ndarray                      # observation noise variance per channel
    process_noise_mask: onp.ndarray     # the state components a process noise acts on
    x_index: int                        # the scored state component
    force: MaternForce
    t0: float
    T: float
    K: int
    y: onp.ndarray                      # the data, (K, d_obs)
    s2: float                           # observation noise variance per observation interval
    bounds: list                        # boxes of (log process-noise density, log xi_u, log ell_u)
    u_mid: onp.ndarray                  # the truth at the held-out interval midpoints
    x_mid: onp.ndarray
    rms_u: float
    rms_x: float
    amp: float                          # largest true state amplitude, the scale of `DIVERGED`


def problem(name, K, **system_kw) -> ClassicalProblem:
    """The classical model's view of the system `name` with `K` observations."""
    system = systems.build(name, n_obs=K, **system_kw)
    t_obs, y = system.observations()
    t_mid = t_obs - 0.5 * system.T / K
    u_mid, x_mid = system.u_true(t_mid), system.solve(t_mid)[:, system.x_index]
    R = onp.asarray(system.noise_stds, dtype=float) ** 2
    s2 = float(onp.mean(R) / (system.T / K))
    return ClassicalProblem(
        rhs=system.rhs, observe=system.observe, y0=onp.asarray(system.y0, dtype=float), R=R,
        process_noise_mask=system.process_noise_mask, x_index=system.x_index, force=system.force,
        t0=float(system.tspan[0]), T=system.T, K=K, y=onp.asarray(y, dtype=float), s2=s2,
        bounds=[(onp.log(s2) - 25.0, onp.log(s2) + 10.0), *system.theta_u_bounds], u_mid=u_mid, x_mid=x_mid,
        rms_u=float(onp.sqrt(onp.mean(u_mid ** 2))), rms_x=float(onp.sqrt(onp.mean(x_mid ** 2))),
        amp=float(onp.abs(system.solve(onp.sort(onp.concatenate([t_obs, t_mid])))).max()))


# ------------------------------------------------------------------ the filter on one grid
class _Discretisation(NamedTuple):
    """A problem on the grid of density `kappa`: `N` RK4 steps of size `h`, state `x` of size `d` and joint
    state `(x, U)` of size `D`, and the data at nodes 1..N (zero where `observed` is false)."""

    problem: ClassicalProblem
    N: int
    h: float
    d: int
    D: int
    ts: jnp.ndarray
    data: jnp.ndarray
    observed: jnp.ndarray
    R_sqr: jnp.ndarray


def _discretise(problem: ClassicalProblem, kappa) -> _Discretisation:
    N = kappa * problem.K
    d = len(problem.y0)
    observed = onp.zeros(N + 1, bool)
    observed[kappa * onp.arange(1, problem.K + 1)] = True
    data = onp.zeros((N + 1, problem.y.shape[1]))
    data[observed] = problem.y
    return _Discretisation(problem=problem, N=N, h=problem.T / N, d=d, D=d + problem.force.q + 1,
                           ts=jnp.asarray(problem.t0 + problem.T / N * onp.arange(N + 1)),
                           data=jnp.asarray(data[1:]), observed=jnp.asarray(observed[1:]),
                           R_sqr=jnp.diag(jnp.sqrt(jnp.asarray(problem.R))))


def _matrices(disc: _Discretisation, phi, use_q):
    """The force prior's transitions over a step and a half step, the joint process-noise factor, and the
    initial law, at `phi = (log process-noise density, log xi_u, log ell_u)`."""
    p, h, d, D = disc.problem, disc.h, disc.d, disc.D
    prior_u = p.force.prior(jnp.exp(phi[1]), jnp.exp(phi[2]))
    A, A_half, Q_u = prior_u.A(h), prior_u.A(0.5 * h), prior_u.Q_sqr(h)
    mask = jnp.asarray(p.process_noise_mask)
    q_state = jnp.sqrt(h * (use_q * jnp.exp(phi[0]) * mask + JITTER_REL * p.s2))
    Q_sqr = jnp.zeros((D, D)).at[:d, :d].set(jnp.diag(q_state)).at[d:, d:].set(Q_u)
    P0 = jnp.zeros((D, D)).at[d:, d:].set(p.force.stationary_cov_sqr(jnp.exp(phi[1]), jnp.exp(phi[2])))
    return A, A_half, Q_sqr, jnp.concatenate([jnp.asarray(p.y0), jnp.zeros(D - d)]), P0


def _transition(disc: _Discretisation, s, t, A, A_half):
    """One RK4 step of the joint state, the force at the stage times read off the force state."""
    x, U = s[:disc.d], s[disc.d:]
    x_next = rk4_step(disc.problem.rhs, x, t, disc.h, U[0], (A_half @ U)[0], (A @ U)[0])
    return jnp.concatenate([x_next, A @ U])


def _update(disc: _Discretisation, m_pred, P_pred, s_lin, t, y, active):
    """The observation update linearised at `s_lin`, where `active`; returns its log-likelihood too."""
    def observe(s):
        return disc.problem.observe(s[:disc.d], s[disc.d], t)

    H = jax.jacfwd(observe)(s_lin)
    c = observe(s_lin) - H @ s_lin - y
    posterior, innovation = update(AffineModel(H, c, disc.R_sqr), Gaussian(m_pred, P_pred))
    ll = jax.lax.select(active, log_likelihood(innovation), jnp.array(0.0))
    return jax.lax.select(active, posterior.m, m_pred), jax.lax.select(active, posterior.P_sqr, P_pred), ll


def _filter_pass(disc: _Discretisation, phi, use_q, traj) -> FilterResult:
    """The Kalman filter of the model linearised along `traj` (nodes 0..N), or an EKF with `traj=None`.

    The data are its only measurement: its log-likelihood is `log_likelihood_obs`, and the ODE-information
    evidence `log_likelihood` of an ODE filter is zero.
    """
    A, A_half, Q_sqr, m0, P0 = _matrices(disc, phi, use_q)

    def transition(s, t):
        return _transition(disc, s, t, A, A_half)

    def step(carry, inp):
        m, P, ll = carry
        s_a, s_b, t, y, active = inp
        s_a = m if traj is None else s_a
        F = jax.jacfwd(transition)(s_a, t)
        b = transition(s_a, t) - F @ s_a
        x = Gaussian(m, P)
        predicted = marginalize(F, b, Q_sqr, x)
        backward = invert(F, Q_sqr, x, predicted)
        m_new, P_new, dll = _update(disc, predicted.m, predicted.P_sqr, predicted.m if traj is None else s_b,
                                    t + disc.h, y, active)
        return (m_new, P_new, ll + dll), (m_new, P_new, backward)

    lin = (jnp.zeros((disc.N, disc.D)),) * 2 if traj is None else (traj[:-1], traj[1:])
    (_, _, ll), (m_f, P_f, backward) = jax.lax.scan(
        step, (m0, P0, jnp.array(0.0)), (*lin, disc.ts[:-1], disc.data, disc.observed))
    return FilterResult(t=disc.ts, m=jnp.concatenate([m0[None], m_f]), P_sqr=jnp.concatenate([P0[None], P_f]),
                        log_likelihood=jnp.array(0.0), log_likelihood_obs=ll, G_back=backward.G, d_back=backward.d,
                        P_back_sqr=backward.P_sqr)


def _smoothed(out: FilterResult):
    """`(means, covariance factors, backward gains, ell_OBS)`."""
    return (*rts_smoother(out), out.G_back, out.log_likelihood_obs)


@dataclass(frozen=True)
class ClassicalFit:
    """The result of `ClassicalLFM.fit`; a failed fit keeps the trajectory it started the last cycle from."""

    phi: onp.ndarray
    traj: onp.ndarray
    passes: int
    cycles: int
    failed: bool
    step: float = float("nan")


class ClassicalLFM:
    """The classical LFM of a `ClassicalProblem` on the grid of density `kappa`, with parameters
    `phi = (log process-noise density, log xi_u, log ell_u)`."""

    def __init__(self, problem: ClassicalProblem, kappa):
        self.problem, self.kappa = problem, kappa
        disc = _discretise(problem, kappa)
        self.N, self.h, self.d, self.D = disc.N, disc.h, disc.d, disc.D
        self.ekf_smoother = jax.jit(lambda phi, use_q: _smoothed(_filter_pass(disc, phi, use_q, None)))
        self.smoother = jax.jit(lambda phi, use_q, traj: _smoothed(_filter_pass(disc, phi, use_q, traj)))
        self.neg_ll_obs = jax.jit(jax.value_and_grad(
            lambda phi, use_q, traj: -_filter_pass(disc, phi, use_q, traj).log_likelihood_obs))

    def start(self, start_u=START_U):
        """`phi^(0)`: the process-noise density at `s2`, and the force hyperparameters at the fractions
        `start_u` of their boxes."""
        b = self.problem.bounds
        return onp.array([onp.log(self.problem.s2)]
                         + [lo + f * (hi - lo) for (lo, hi), f in zip(b[1:], start_u, strict=True)])

    def finite(self, traj):
        """Whether a smoothed trajectory is finite and has not diverged."""
        traj = onp.asarray(traj)
        return bool(onp.all(onp.isfinite(traj)) and onp.abs(traj[:, :self.d]).max() < DIVERGED * self.problem.amp)

    def fit(self, phi, traj, use_q) -> ClassicalFit:
        """The iterated extended Kalman smoother alternated with L-BFGS-B on `-ell_OBS`."""
        free = [0, 1, 2] if use_q else [1, 2]
        use_q = float(use_q)
        phi = onp.array(phi, dtype=float)
        passes, nll_prev, step = 0, None, float("inf")
        for cycle in range(1, N_MAX + 1):
            traj_j = jnp.asarray(traj)
            res = lbfgs_phi(lambda p, traj_j=traj_j: self.neg_ll_obs(jnp.asarray(p), use_q, traj_j), phi, free,
                            self.problem.bounds)
            phi = res.x
            m_s = onp.asarray(self.smoother(jnp.asarray(phi), use_q, traj_j)[0])
            passes += res.n_evals + 1
            if not self.finite(m_s):
                return ClassicalFit(phi, traj, passes, cycle, failed=True)
            step = float(onp.sqrt(onp.mean((m_s[:, :self.d] - onp.asarray(traj)[:, :self.d]) ** 2)))
            traj = m_s
            if nll_prev is not None and abs(res.value - nll_prev) < TAU_L:
                break
            nll_prev = res.value
        return ClassicalFit(phi, traj, passes, cycle, failed=False, step=step)

    def midpoint_force_bridge(self, phi, m_s, P, G):
        """Force marginals at the interval midpoints on an odd grid: the Matern bridge between the smoothed
        force states at the two neighbouring nodes."""
        d, kappa = self.d, self.kappa
        prior_u = self.problem.force.prior(float(onp.exp(phi[1])), float(onp.exp(phi[2])))
        A = onp.asarray(prior_u.A(0.5 * self.h))
        Q_sqr = onp.asarray(prior_u.Q_sqr(0.5 * self.h))
        Q_inv = onp.linalg.inv(Q_sqr.T @ Q_sqr)
        C = onp.linalg.inv(Q_inv + A.T @ Q_inv @ A)
        M_a, M_b = C @ Q_inv @ A, C @ A.T @ Q_inv
        mean, var = onp.zeros(self.problem.K), onp.zeros(self.problem.K)
        for k in range(self.problem.K):
            n = kappa * k + (kappa - 1) // 2
            P_a, P_b = P[n][d:, d:], P[n + 1][d:, d:]
            P_ab = (G[n] @ P[n + 1])[d:, d:]
            mean[k] = (M_a @ m_s[n, d:] + M_b @ m_s[n + 1, d:])[0]
            var[k] = (C + M_a @ P_a @ M_a.T + M_b @ P_b @ M_b.T + M_a @ P_ab @ M_b.T + M_b @ P_ab.T @ M_a.T)[0, 0]
        return mean, var

    def scores(self, phi, traj, use_q):
        """Errors and chi^2 at the held-out midpoints, with the force's 95 % chi^2 interval on an even grid,
        and the force mean (`u_obs`) and its rms sd (`sd_obs`) at the observation times."""
        m_s, P_s, G, ll = (onp.asarray(v) for v in self.smoother(jnp.asarray(phi), float(use_q), jnp.asarray(traj)))
        P = onp.asarray(covariances(P_s))
        p, d, kappa = self.problem, self.d, self.kappa
        obs = kappa * onp.arange(1, p.K + 1)
        out = {"ll_obs": float(ll), "u_obs": m_s[obs, d].tolist(), "sd_obs": float(onp.sqrt(onp.mean(P[obs, d, d])))}
        if kappa % 2 == 0:
            mid = obs - kappa // 2
            mean, var = m_s[mid, d], P[mid, d, d]
            err_x = m_s[mid, p.x_index] - p.x_mid
            out["rel_x"] = float(onp.sqrt(onp.mean(err_x ** 2)) / p.rms_x)
            out["chi2_x"] = float(onp.mean(err_x ** 2 / P[mid, p.x_index, p.x_index]))
            interval = chi2_interval(smoother_cross_covariance(P, G, [(n, onp.eye(self.D)[d]) for n in mid]))
            out.update(chi2_u_lo=interval["lo"], chi2_u_hi=interval["hi"])
        else:
            mean, var = self.midpoint_force_bridge(phi, m_s, P, G)
        err_u = mean - p.u_mid
        out.update(rel_u=float(onp.sqrt(onp.mean(err_u ** 2)) / p.rms_u), chi2_u=float(onp.mean(err_u ** 2 / var)),
                   sd_u=float(onp.sqrt(onp.mean(var)) / p.rms_u))
        return out


# ------------------------------------------------------------------ fits and the grid loop
def _ekf_start(lfm, phi, use_q):
    """The smoothed means of an EKF pass at `phi`, the cold start's linearisation."""
    return onp.asarray(lfm.ekf_smoother(jnp.asarray(phi), float(use_q))[0])


@dataclass(frozen=True)
class _Grid:
    """One grid of the classical ladder: the model, its fit (None if the EKF start diverged), how it started."""

    lfm: ClassicalLFM
    use_q: bool
    warm: bool
    t0: float
    fit: ClassicalFit | None = None

    @property
    def failed(self) -> bool:
        return self.fit is None or self.fit.failed

    @property
    def passes(self) -> int:
        """Filter passes on this grid, the cold start's EKF pass included."""
        return (0 if self.warm else 1) + (0 if self.fit is None else self.fit.passes)

    @cached_property
    def scores(self) -> dict:
        return self.lfm.scores(self.fit.phi, self.fit.traj, self.use_q)


def _borrow_odd_intervals(rows):
    """The rows with the force's chi^2 interval of an odd grid (which has none of its own) taken from the
    next grid that has one, and whether each fit's chi^2 is inside its interval."""
    out = []
    for i, row in enumerate(rows):
        if "chi2_u" in row and "chi2_u_lo" not in row:
            nxt = next((r for r in rows[i + 1:] if "chi2_u_lo" in r), None)
            row = {**row, "chi2_u_lo": nxt["chi2_u_lo"] if nxt else float("nan"),
                   "chi2_u_hi": nxt["chi2_u_hi"] if nxt else float("nan")}
        if "chi2_u" in row:
            row = {**row, "inside": bool(row["chi2_u_lo"] <= row["chi2_u"] <= row["chi2_u_hi"])}
        out.append(row)
    return out


def ladder(problem, use_q, kappas=KAPPAS, keep_fits=False, start_u=START_U, reset_force=True):
    """The classical model's grid loop over the whole ladder, warm, marking the first grid the one-sd check
    accepts.

    `keep_fits`: each fitted row also carries `fit = (ClassicalLFM, phi, trajectory)`. `start_u`: the force
    start (`ClassicalLFM.start`). `reset_force` (the default): a warm grid keeps the linearisation and the
    process noise but restarts the force hyperparameters from `start_u`, as in `lfcal.calibrate`.
    """
    def cold(kappa):
        t0, lfm = time.time(), ClassicalLFM(problem, kappa)
        phi = lfm.start(start_u)
        traj = _ekf_start(lfm, phi, use_q)
        fit = lfm.fit(phi, traj, use_q) if lfm.finite(traj) else None
        return _Grid(lfm, use_q, warm=False, t0=t0, fit=fit)

    def warm(kappa, previous):
        t0, lfm = time.time(), ClassicalLFM(problem, kappa)
        phi = onp.array(previous.fit.phi)
        if reset_force:
            phi[1:] = lfm.start(start_u)[1:]
        return _Grid(lfm, use_q, warm=True, t0=t0, fit=lfm.fit(phi, prolong(previous.fit.traj, from_t0=True), use_q))

    rows, accepted, cost = [], None, 0
    for kappa, previous, grid in warm_path(kappas, cold, warm, failed=lambda g: g.failed):
        cost += RK4_STAGES * grid.passes * kappa
        if grid.fit is None:
            rows.append({"kappa": kappa, "failed": True, "stop": "start diverged", "wall_s": time.time() - grid.t0})
            continue
        result = grid.fit
        row = {"kappa": kappa, "passes": result.passes, "cycles": result.cycles, "failed": result.failed,
               "warm": grid.warm, **phi_fields(result.phi), "cost_cum": cost}
        if result.failed:
            rows.append({**row, "stop": "fit diverged", "wall_s": time.time() - grid.t0})
            continue
        row.update(step=result.step, **grid.scores)
        if previous is not None:
            u_prev, u_obs = onp.asarray(previous.scores["u_obs"]), onp.asarray(grid.scores["u_obs"])
            row["du_sd"] = moved = float(onp.sqrt(onp.mean((u_obs - u_prev) ** 2)) / grid.scores["sd_obs"])
            if accepted is None and moved < TAU_U:
                accepted = kappa
                row["accepted"] = True
        row["wall_s"] = time.time() - grid.t0
        if keep_fits:
            row["fit"] = (grid.lfm, result.phi, result.traj)
        rows.append(row)
    return {"rows": _borrow_odd_intervals(rows), "accepted": accepted}


def reported_row(ladder_result):
    """The classical model's reported grid of a `ladder`: the accepted one, else the last fitted one, or
    None if no grid was fitted."""
    fitted = [r for r in ladder_result["rows"] if "rel_u" in r]
    return next((r for r in fitted if r.get("accepted")), fitted[-1] if fitted else None)
