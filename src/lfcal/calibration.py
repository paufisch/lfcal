"""One latent force problem on the grid of density `kappa`: the pieces the algorithm of `lfcal.algorithm`
is built from.

`Calibration(problem, kappa)` builds the filter on the grid and exposes jitted functions of the
hyperparameters `phi = (log sigma_x^2, log xi_u, log ell_u)`, whose box is `bounds` (`problem.bounds`):

  linearize                 the residual row's frozen linearisation along a trajectory (an `AffineSurrogate`)
  transition                the joint prior's transition over one step
  data_pass                 the affine data pass at a frozen linearisation
  ode_evidence              `ell_ODE` of the data-free pass at a frozen linearisation
  relinearize               one IEKS iteration at the smoothed mean
  posterior                 the smoothed posterior (with the grid's `posterior_extras`: a benchmark's scores)
  ek1_predicted_means       the one-pass EK1 predicted means, the first linearisation

On top of these: L-BFGS-B on the negative observation evidence `-ell_OBS` at a frozen linearisation
(`fit_hyperparameters`), the information criteria for the diffusion (`information_value_and_grad`),
`log p(Y | Z)` at a frozen linearisation (`frozen_data_likelihood_value_and_grad`), and the ODE-consistency
statistic (`consistency`).
"""

from __future__ import annotations

from functools import cached_property
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as onp
from ode_filters import affine_filter, conditional_loglik, information_criteria, iterated_smoother
from scipy.optimize import minimize

from lfcal.moments import marginal_variances, rms_change
from lfcal.problem import rk4_step

__all__ = ["LBFGS_SCALE", "PENALTY", "SUB", "W_OBS", "Calibration", "Minimum", "lbfgs", "lbfgs_phi"]

#: the value L-BFGS-B sees where the objective or its gradient is not finite
PENALTY = 1e10
#: L-BFGS-B works on `LBFGS_SCALE * phi` (`lbfgs_phi`), i.e. on the length scale as `log ell_u^2`
LBFGS_SCALE = (1.0, 1.0, 2.0)
#: the ODE-consistency check solves the ODE in windows of `W_OBS` observation intervals, `SUB` RK4 steps per
#: interval. Shorter windows hide a wrong force: a weakly enforced fit restarts each window on the data.
W_OBS, SUB = 16, 32


class Minimum(NamedTuple):
    """The result of `lbfgs`: the argmin, the minimum, and how often the objective was evaluated."""

    x: onp.ndarray
    value: float
    n_evals: int


def lbfgs(value_and_grad, x0, bounds, maxiter=100) -> Minimum:
    """L-BFGS-B on `value_and_grad(x) -> (value, gradient)` in the box `bounds`.

    A non-finite value or gradient counts as `PENALTY` with a zero gradient.
    """
    n_evals = 0

    def fun(x):
        nonlocal n_evals
        n_evals += 1
        v, g = value_and_grad(x)
        v, g = float(v), onp.asarray(g, dtype=float)
        if not (onp.isfinite(v) and onp.all(onp.isfinite(g))):
            return PENALTY, onp.zeros_like(x)
        return v, g

    res = minimize(fun, onp.asarray(x0, dtype=float), jac=True, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": maxiter})
    return Minimum(res.x, float(res.fun), n_evals)


def lbfgs_phi(value_and_grad, phi0, free, bounds) -> Minimum:
    """`lbfgs` on `value_and_grad(phi) -> (value, gradient in phi)` over the coordinates `free` of
    `phi = (log sigma_x^2 or the classical LFM's log process noise, log xi_u, log ell_u)`, the others held at
    `phi0`, within `bounds`. Returns the whole `phi` at the minimum.

    L-BFGS-B runs on `LBFGS_SCALE * phi`, i.e. on `log ell_u^2`, the coordinate of ode-filters 0.7 in which the
    recorded results were fitted. L-BFGS-B is not scale invariant: in `log ell_u` it stops at a slightly
    different point, and the algorithm can accept another grid.
    """
    phi0, free = onp.asarray(phi0, dtype=float), list(free)
    scale = onp.asarray(LBFGS_SCALE)[free]

    def with_free(q):
        phi = phi0.copy()
        phi[free] = q / scale
        return phi

    def objective(q):
        v, g = value_and_grad(with_free(q))
        return v, onp.asarray(g)[free] / scale

    box = [(lo * s, hi * s) for (lo, hi), s in zip([bounds[i] for i in free], scale, strict=True)]
    res = lbfgs(objective, phi0[free] * scale, box)
    return Minimum(with_free(res.x), res.value, res.n_evals)


class Calibration:
    """The `LatentForceProblem` `problem` on the grid of density `kappa`."""

    def __init__(self, problem, kappa):
        self.kappa = kappa
        self.problem = problem
        self.grid = grid = problem.discretize(kappa)
        self.bounds = problem.bounds

        def model_at(phi):
            """`(prior, mu0, P0_sqr)`: the joint prior and its initial law at `phi`."""
            return problem.prior(jnp.exp(phi[0]), phi[1:]), *problem.initial_law(phi[1:])

        def data_pass(phi, surrogate, obs_model):
            prior, mu0, P0 = model_at(phi)
            return affine_filter(mu0, P0, prior, surrogate, grid.ts, obs_model=obs_model)

        def iteration(phi, traj):
            """One IEKS iteration from `traj`."""
            prior, mu0, P0 = model_at(phi)
            return iterated_smoother(mu0, P0, prior, grid.ode_measure, grid.ts, n_iters=1, init=traj,
                                     obs_model=grid.obs_model)

        def relin(phi, traj):
            m_s = iteration(phi, traj).m
            return m_s[1:], rms_change(m_s[1:], traj, grid.x_col)

        def posterior(phi, traj):
            res = iteration(phi, traj)
            var = marginal_variances(res.P_sqr)
            return {"m_s": res.m, "var": var, "ll_obs": res.filter.log_likelihood_obs,
                    **grid.posterior_extras(res.m, var)}

        # `_model_at` stays unjitted, to be traced into the jitted functions below
        self._model_at = model_at
        self.linearize = jax.jit(grid.linearize)
        self.transition = jax.jit(lambda phi: problem.prior(jnp.exp(phi[0]), phi[1:]).transition(grid.h))
        self.data_pass = jax.jit(lambda phi, surrogate: data_pass(phi, surrogate, grid.obs_model))
        self.ode_evidence = jax.jit(lambda phi, surrogate: data_pass(phi, surrogate, None).log_likelihood)
        self.relinearize = jax.jit(relin)
        self.posterior = jax.jit(posterior)
        self.neg_ll_obs = jax.jit(jax.value_and_grad(
            lambda phi, surrogate: -data_pass(phi, surrogate, grid.obs_model).log_likelihood_obs))
        self.ek1_predicted_means = jax.jit(lambda phi: problem.filter(grid, jnp.exp(phi[0]), phi[1:])[1].m_pred)

    def fit_hyperparameters(self, surrogate, phi0, free) -> Minimum:
        """L-BFGS-B on `-ell_OBS` at the frozen linearisation `surrogate` over the coordinates `free` of
        `phi`, the others held at `phi0`. Returns the whole `phi` at the minimum."""
        return lbfgs_phi(lambda phi: self.neg_ll_obs(jnp.asarray(phi), surrogate), phi0, free, self.bounds)

    @cached_property
    def information_value_and_grad(self):
        """Value and gradient in `log sigma_x^2` of the information criteria at a frozen linearisation,
        `phi_u` held: `{"mi": I(U; Y | Z), "di": the directed information}`."""
        grid = self.grid

        def criteria(log_sigma_x2, phi, surrogate):
            prior, mu0, P0 = self._model_at(phi.at[0].set(log_sigma_x2))
            info = information_criteria(mu0, P0, prior, surrogate, grid.ts, grid.obs_model)
            return {"mi": info.I_UY_given_Z, "di": info.D}

        return {key: jax.jit(jax.value_and_grad(lambda s, phi, surrogate, key=key: criteria(s, phi, surrogate)[key]))
                for key in ("mi", "di")}

    @cached_property
    def frozen_data_likelihood_value_and_grad(self):
        """Jitted `(log sigma_x^2, phi, surrogate) -> (log p(Y | Z), its derivative in log sigma_x^2)` at the
        frozen linearisation `surrogate`, `phi_u` held.

        `log p(Y | Z)` is `ell_ODE + ell_OBS` of the data pass minus `ell_Z` of the data-free pass. At one
        shared linearisation it is both Fenrir's and Dalton's criterion.
        """
        grid = self.grid

        @jax.jit
        def data_likelihood(phi, surrogate):
            prior, mu0, P0 = self._model_at(phi)
            return conditional_loglik(affine_filter(mu0, P0, prior, surrogate, grid.ts, obs_model=grid.obs_model),
                                      affine_filter(mu0, P0, prior, surrogate, grid.ts))

        return jax.jit(jax.value_and_grad(lambda s, phi, surrogate: data_likelihood(phi.at[0].set(s), surrogate)))

    @cached_property
    def _consistency_statistic(self):
        problem, grid = self.problem, self.grid
        _, data = problem.observations()
        noise = onp.asarray(problem.noise_stds, dtype=float)
        K, kappa = grid.K, grid.kappa
        t0, dT = problem.tspan[0], problem.T / K
        dt = dT / SUB
        ts = jnp.asarray(grid.ts)
        starts = onp.arange(0, K, W_OBS)
        obs_i = starts[:, None] + onp.arange(W_OBS)[None, :]
        valid = obs_i < K
        window_data = jnp.asarray(data)[onp.minimum(obs_i, K - 1)]

        def window_prediction(y0, start, u_nodes):
            def u_at(t):
                return jnp.interp(t, ts, u_nodes)

            def step(y, i):
                t = t0 + start * dT + i * dt
                y = rk4_step(problem.rhs, y, t, dt, u_at(t), u_at(t + 0.5 * dt), u_at(t + dt))
                return y, y

            _, ys = jax.lax.scan(step, y0, jnp.arange(W_OBS * SUB))
            ys = ys[SUB - 1::SUB]                                   # the states at the window's observation times
            tk = t0 + start * dT + dT * jnp.arange(1, W_OBS + 1)
            return jax.vmap(lambda y, t: problem.observe(y, u_at(t), t))(ys, tk)

        def statistic(u_nodes, states):
            pred = jax.vmap(window_prediction, in_axes=(0, 0, None))(states[starts * kappa],
                                                                      jnp.asarray(starts, dtype=float), u_nodes)
            r = jnp.where(valid[..., None], (window_data - pred) / jnp.asarray(noise), 0.0)
            return jnp.sqrt(jnp.sum(r ** 2) / (valid.sum() * data.shape[1]))

        return jax.jit(statistic)

    def consistency(self, phi, traj):
        """The ODE-consistency statistic `r_kappa` of the fit `(phi, traj)`.

        In windows of `W_OBS` observation intervals, each started from the fit's smoothed state, RK4 solves
        the ODE (`SUB` steps per interval) with the force mean as a known input. The statistic is the
        root-mean-square whitened residual of the observations, about 1 for a calibrated fit.
        """
        post = self.posterior(jnp.asarray(phi), jnp.asarray(traj))
        m_s, cols = post["m_s"], list(self.grid.ode_state_cols)
        v = float(self._consistency_statistic(m_s[:, self.grid.u_col], m_s[:, cols]))
        return v if onp.isfinite(v) else float("inf")
