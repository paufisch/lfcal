"""The Matern prior over the scalar latent force, and its sample paths.

`MaternForce` is set by its marginal std and its length scale. `xi`, the `Xi` of `ode_filters`, is the marginal
variance of a `MaternPrior`, so `xi = marginal_std^2`. The length scale is the textbook one (Rasmussen &
Williams 2006, eq. 4.14): the SDE rate is `lambda = sqrt(2q + 1) / length_scale`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as onp
from ode_filters import MaternPrior

__all__ = ["MaternForce", "sample_path"]


def sample_path(prior, tspan, seed=0, n_grid=2000) -> Callable:
    """`u(t)`: one reproducible sample path of `prior`, started from its stationary law.

    The path is sampled once on `n_grid` points and interpolated linearly. The initial state has its own
    key (`fold_in(key, 1)`), so the increment stream does not depend on it.
    """
    ts = onp.linspace(*tspan, n_grid)
    h = float(ts[1] - ts[0])
    A, Q_sqr, E0 = prior.A(h), prior.Q_sqr(h), prior.E0
    D = A.shape[0]
    key = jax.random.PRNGKey(seed)
    s0 = jax.random.normal(jax.random.fold_in(key, 1), (D,)) @ prior.stationary_cov_sqr()

    def step(s, k):
        s_next = A @ s + Q_sqr.T @ jax.random.normal(k, (D,))
        return s_next, E0 @ s_next

    _, path = jax.lax.scan(step, s0, jax.random.split(key, n_grid - 1))
    values = jnp.concatenate([(E0 @ s0)[None], path])[:, 0]

    def u(t):
        return jnp.interp(jnp.asarray(t), ts, values)

    return u


@dataclass(frozen=True)
class MaternForce:
    """A stationary Matern prior over a scalar latent force, in marginal-std units. Sampled from, it is the
    truth; filtered with, it is the model."""

    q: int = 2
    length_scale: float = 1.0
    marginal_std: float = 1.0

    def prior(self, xi=None, length_scale=None):
        """The `MaternPrior` at `self.xi` / `self.length_scale` unless overridden (may be traced)."""
        xi = self.xi if xi is None else xi
        length_scale = self.length_scale if length_scale is None else length_scale
        return MaternPrior(self.q, 1, length_scale=length_scale, Xi=xi * jnp.eye(1))

    @property
    def xi(self) -> float:
        """The `Xi` that gives `marginal_std`: the prior's marginal variance."""
        return float(self.marginal_std ** 2)

    def stationary_cov_sqr(self, xi=None, length_scale=None):
        """Upper-triangular factor of the stationary covariance: the force block's `P0`."""
        return self.prior(xi, length_scale).stationary_cov_sqr()

    def sample(self, tspan, *, seed: int = 42, n_grid: int = 2000) -> Callable:
        """One reproducible path `u(t)` of this prior, from its stationary law."""
        return sample_path(self.prior(), tspan, seed=seed, n_grid=n_grid)
