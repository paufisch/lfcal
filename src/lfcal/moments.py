"""Covariances and marginal variances of square-root factors, and the change between two trajectories.

Square-root convention of `ode_filters`: `P = P_sqr.T @ P_sqr`.
"""

from __future__ import annotations

import jax.numpy as jnp

__all__ = ["cov", "covariances", "marginal_variances", "rms_change"]


def cov(P_sqr):
    """The covariance of a square-root factor."""
    return P_sqr.T @ P_sqr


def covariances(P_sqr):
    """The covariances of a sequence of square-root factors."""
    return jnp.einsum("nki,nkj->nij", P_sqr, P_sqr)


def marginal_variances(P_sqr):
    """The marginal variances of a sequence of square-root factors, shape `(n, D)`."""
    return jnp.einsum("nij,nij->nj", P_sqr, P_sqr)


def rms_change(new, old, cols):
    """Root-mean-square change of the columns `cols` between two trajectories."""
    return jnp.sqrt(jnp.mean((new[:, cols] - old[:, cols]) ** 2))
