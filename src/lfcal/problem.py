"""A latent force problem: an ODE driven by a scalar latent force, observed with Gaussian noise.

`LatentForceProblem` is what `lfcal.calibrate` takes: the ODE (first or second order, in `d` components), its
known initial value, the data at `K` equidistant times, and the force's reference scale and length-scale
guess, which centre its search boxes.

`discretize(kappa)` builds a `Grid` with `N = kappa K` filter steps, so every observation time is a node at
every density. The grid holds everything that does not depend on the hyperparameters, so it is built once
with numpy. `filter(grid, sigma_x, log_theta_u)` then runs the joint EK1 filter on it, and the
hyperparameters may be traced by JAX.

Units: the filter's force `u` has unit marginal std under its prior, and the vector field receives
`force_scale * u`. The filter runs in the problem's own time units. `theta_u = (xi_u, ell_u)` are the Matern
variance scale and length scale (`force.MaternForce`), and `sigma_x` is the diffusion of the state prior (a
variance).

The ODE also comes in first-order form with the force as an input (`rhs`, `observe`), which the
ODE-consistency check solves with RK4.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import jax.numpy as jnp
import numpy as onp
from ode_filters import (
    IWP,
    AffineSurrogate,
    JointPrior,
    Measurement,
    ODEInformationWithHidden,
    SecondOrderODEInformationWithHidden,
    TaylorCorrection,
    gaussian_filter,
    joint_taylor_initialization,
    prepare_observations,
    taylor_surrogate,
)

from lfcal.force import MaternForce

__all__ = ["DIFFUSION_BOX", "EK1", "Grid", "LatentForceProblem", "diffusion_shift", "rk4_step", "state_order"]

#: The residual row is linearised by first-order Taylor expansion (EK1); the iterated linearisation re-uses
#: the same Taylor tangent.
EK1 = TaylorCorrection(order=1)
#: the default box of sigma_x^2, in units of `diffusion_shift` (the paper's boxes span 10^3 to 10^20 in them)
DIFFUSION_BOX = (1e3, 1e20)


def state_order(ode_order: int, q_u: int) -> int:
    """The paper's order of the state prior, `q_x = m + q_u - 1` for an additive force."""
    return int(ode_order) + int(q_u) - 1


def diffusion_shift(T, q_x) -> float:
    """The unit of the sigma_x^2 boxes on a window of length `T` with a state prior of order `q_x`:
    `T^(2 (q_x - 2)) (1 / T)^(2 q_x + 1)`, that is `T^-5`, as the paper gives its boxes."""
    q, T = int(q_x), float(T)
    return T ** (2 * (q - 2)) * (1.0 / T) ** (2 * q + 1)


def rk4_step(rhs, y, t, dt, u_start, u_mid, u_end):
    """One classical Runge--Kutta step of `y' = rhs(y, u, t)`, with the input given at the stage times."""
    k1 = rhs(y, u_start, t)
    k2 = rhs(y + 0.5 * dt * k1, u_mid, t + 0.5 * dt)
    k3 = rhs(y + 0.5 * dt * k2, u_mid, t + 0.5 * dt)
    k4 = rhs(y + dt * k3, u_end, t + dt)
    return y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


@dataclass(frozen=True, kw_only=True)
class Grid:
    """A problem on the grid of density `kappa`: `N = kappa K` steps, with every observation time a node.

    It holds the parts of the joint filter that depend on neither `sigma_x` nor `theta_u`.
    """

    kappa: int
    tspan: tuple[float, float]
    N: int
    ode_measure: Any           # the ODE-residual information (exact: no noise)
    obs_model: Any             # the data (`ObsModel`); None for the data-free pass
    correction: Any            # how the EK1 pass linearises the residual
    u_col: int                 # the force's column in the joint state
    ode_state_cols: tuple      # columns of the ODE's first-order state (x, and x' for second order)
    x_col: int = 0             # the state column whose IEKS step is recorded

    @property
    def K(self) -> int:
        return self.N // self.kappa

    @property
    def h(self) -> float:
        return (self.tspan[1] - self.tspan[0]) / self.N

    @property
    def ts(self):
        return onp.linspace(*self.tspan, self.N + 1)

    @property
    def t_steps(self):
        """The times of the `N` filter steps (nodes 1..N)."""
        return jnp.asarray(self.ts[1:])

    @property
    def obs_nodes(self):
        return self.kappa * onp.arange(1, self.K + 1)

    @property
    def mid_idx(self):
        """The nodes at the midpoints of the `K` observation intervals, which hold no data (an even `kappa`)."""
        if self.kappa % 2:
            raise ValueError(f"the interval midpoints are nodes only for an even kappa; got {self.kappa}")
        return (2 * onp.arange(self.K) + 1) * (self.kappa // 2)

    def linearize(self, traj) -> AffineSurrogate:
        """The residual row linearised along `traj` (nodes 1..N): the frozen linearisation."""
        return taylor_surrogate(self.ode_measure, traj, ts=self.t_steps)

    def has_diverged(self, max_abs_x) -> bool:
        """Whether a posterior mean that reaches `max_abs_x` in the column `x_col` has diverged: here, unless it
        is finite."""
        return not float(max_abs_x) < onp.inf

    def posterior_extras(self, m_s, var) -> dict:
        """Further quantities of a smoothed posterior, computed in the same jitted pass (none here)."""
        return {}


@dataclass(frozen=True, kw_only=True, eq=False)
class LatentForceProblem:
    """An ODE driven by a scalar latent force, with a known initial value and data at `K` equidistant times.

    The ODE is `x' = f(x, u, t)` (`order=1`) or `x'' = f(x, x', u, t)` (`order=2`) in `d` components, with the
    vector field `vector_field(x, u, *, t)` or `vector_field(x, dx, u, *, t)`: `x` and `dx` of shape `(d,)`,
    `u` a scalar, the force in its own units (add it to one component with `f.at[i].add(u)`; it may also enter
    nonlinearly, e.g. as `exp(u)`).

    The data are `data[k]`, the components `obs_indices` of the derivative `obs_derivative` of `x` (0: `x`
    itself; 2 for a second-order ODE: the acceleration), at `times[k] = t0 + (k + 1) T / K`, with Gaussian
    noise of std `noise_std` (one per observed component, or one for all).

    The force has a Matern prior of order `force_order` (2: Matern-5/2). Its marginal std is fitted in
    `[e^-10, e^5] force_scale`, from `e^0.5 force_scale`, and its length scale within a factor `e^2` of
    `length_scale`, from `0.45 length_scale`. The diffusion `sigma_x^2` of the state prior is fitted in
    `diffusion_box`, by default `diffusion_shift(T, q_x) DIFFUSION_BOX = T^-5 [10^3, 10^20]`, which assumes a
    state of order one and on the paper's systems gives the same fits as their own boxes.
    """

    vector_field: Callable
    x0: Any
    times: Any
    noise_std: Any
    force_scale: float
    length_scale: float
    order: int = 1
    v0: Any = None                         # x'(t0) of a second-order ODE; default zero
    t0: float = 0.0
    data: Any = None                       # shape (K, len(obs_indices))
    obs_indices: tuple = (0,)
    obs_derivative: int = 0
    force_order: int = 2
    diffusion_box: tuple[float, float] | None = None
    q_x: int | None = None                 # state prior order; default `state_order`
    p0_jitter: float | None = None         # added to the initial covariance factor; default 1e-12 on order 1

    def __post_init__(self):
        def set_(name, value):
            object.__setattr__(self, name, value)

        if self.order not in (1, 2):
            raise ValueError(f"order must be 1 or 2, got {self.order!r}")
        x0 = onp.atleast_1d(onp.asarray(self.x0, dtype=float))
        if x0.ndim != 1:
            raise ValueError(f"x0 must be a vector, got shape {x0.shape}")
        set_("x0", x0)
        if self.order == 1 and self.v0 is not None:
            raise ValueError("v0 is the initial velocity of a second-order ODE; this one has order 1")
        if self.order == 2:
            v0 = onp.zeros_like(x0) if self.v0 is None else onp.atleast_1d(onp.asarray(self.v0, dtype=float))
            if v0.shape != x0.shape:
                raise ValueError(f"v0 must have the shape of x0, {x0.shape}; got {v0.shape}")
            set_("v0", v0)
        times = onp.asarray(self.times, dtype=float)
        if times.ndim != 1 or len(times) < 2:
            raise ValueError(f"times must hold at least two observation times, got shape {times.shape}")
        dt = (times[-1] - self.t0) / len(times)
        if not (dt > 0 and onp.allclose(times, self.t0 + dt * onp.arange(1, len(times) + 1), rtol=0, atol=1e-6 * dt)):
            raise ValueError("the observation times must be equidistant, the first one interval after t0: "
                             "times[k] = t0 + (k + 1) dt")
        set_("times", times)
        set_("obs_indices", tuple(int(i) for i in self.obs_indices))
        if not all(0 <= i < len(x0) for i in self.obs_indices):
            raise ValueError(f"obs_indices {self.obs_indices} must be components of x, 0..{len(x0) - 1}")
        if not 0 <= self.obs_derivative <= self.order:
            raise ValueError(f"obs_derivative must be between 0 and the order {self.order}, got {self.obs_derivative}")
        if self.data is not None:
            data = onp.asarray(self.data, dtype=float)
            data = data[:, None] if data.ndim == 1 else data
            if data.shape != (len(times), len(self.obs_indices)):
                raise ValueError(f"data must have shape (K, len(obs_indices)) = {(len(times), len(self.obs_indices))}, "
                                 f"got {data.shape}")
            set_("data", data)
        if self.q_x is None:
            set_("q_x", state_order(self.order, self.force_order))
        if self.q_x < max(self.order, self.obs_derivative):
            raise ValueError(f"q_x = {self.q_x} must be at least the order and the observed derivative")
        if self.p0_jitter is None:
            set_("p0_jitter", 1e-12 if self.order == 1 else 0.0)

    # ------------------------------------------------------------------ the ODE
    @property
    def d_x(self) -> int:
        return int(self.x0.shape[0])

    @property
    def y0(self):
        """The initial state in first-order form."""
        return jnp.asarray(self.x0) if self.order == 1 else jnp.concatenate([jnp.asarray(self.x0), jnp.asarray(self.v0)])

    def _unit_field(self, *args, t):
        """The vector field with the force in model units: `(x, u)` or `(x, dx, u)`."""
        *states, u = args
        return self.vector_field(*states, jnp.reshape(self.force_scale * u, ()), t=t)

    def ode_rhs(self, y, u_phys, t):
        """The ODE in first-order form, `y' = f(y, u_phys, t)`, with the force in its own units."""
        u_phys = jnp.reshape(u_phys, ())
        if self.order == 1:
            return self.vector_field(y, u_phys, t=t)
        d = self.d_x
        return jnp.concatenate([y[d:], jnp.reshape(self.vector_field(y[:d], y[d:], u_phys, t=t), (d,))])

    def rhs(self, y, u, t):
        """The ODE in first-order form, with the force in model units (unit std)."""
        return self.ode_rhs(y, self.force_scale * u, t)

    def observe(self, y, u, t):
        """The observed channels of the first-order state `y`, with the force `u` in model units."""
        d, idx = self.d_x, jnp.asarray(self.obs_indices)
        k = self.obs_derivative
        if k < self.order:
            return y[k * d:(k + 1) * d][idx]
        states = (y,) if self.order == 1 else (y[:d], y[d:])
        return jnp.reshape(self._unit_field(*states, u, t=t), (d,))[idx]

    @property
    def noise_stds(self) -> tuple:
        """The observation noise std per observed channel."""
        stds = onp.broadcast_to(onp.asarray(self.noise_std, dtype=float), (len(self.obs_indices),))
        return tuple(float(s) for s in stds)

    def _ode_measure(self, layout):
        if self.order == 1:
            return ODEInformationWithHidden(self._unit_field, E0=layout.E0_x, E1=layout.E1,
                                            E0_hidden=layout.E0_hidden, constraints=None)
        return SecondOrderODEInformationWithHidden(self._unit_field, E0=layout.E0_x, E1=layout.E1, E2=layout.E2,
                                                   E0_hidden=layout.E0_hidden)

    @property
    def _inits(self):
        """The initial value the Taylor-mode initialization expands: `x0`, or `(x0, v0)` for a second-order
        ODE."""
        return jnp.asarray(self.x0) if self.order == 1 else (jnp.asarray(self.x0), jnp.asarray(self.v0))

    def _obs_matrix_and_projection(self, layout):
        """`(A, E)`: the data observe `A E s` of the joint state `s`."""
        A = jnp.asarray(onp.eye(self.d_x)[list(self.obs_indices)])
        return A, (layout.E0_x, layout.E1, layout.E2)[self.obs_derivative]

    # ------------------------------------------------------------------ time and data
    @property
    def n_obs(self) -> int:
        return len(self.times)

    @property
    def tspan(self) -> tuple[float, float]:
        return float(self.t0), float(self.times[-1])

    @property
    def T(self) -> float:
        return float(self.times[-1] - self.t0)

    @property
    def state_dim(self) -> int:
        return (self.q_x + 1) * self.d_x

    @property
    def obs_times(self):
        return self.times

    def node_times(self, kappa: int):
        return onp.linspace(self.t0, self.times[-1], kappa * self.n_obs + 1)

    def observations(self):
        """`(times, values)` of the data, values of shape `(K, d_obs)`."""
        if self.data is None:
            raise ValueError("the problem has no data: pass data=...")
        return self.times, self.data

    # ------------------------------------------------------------------ the joint filter
    @cached_property
    def force(self) -> MaternForce:
        """The force's prior in model units: unit marginal std at `xi_u = 1`."""
        return MaternForce(q=self.force_order, length_scale=self.length_scale, marginal_std=1.0)

    @property
    def theta_u_bounds(self):
        """Boxes of `(log xi_u, log ell_u)`. The length scale is kept within a factor e^2 of `length_scale`."""
        log_ell = float(onp.log(self.force.length_scale))
        return ((-20.0, 10.0), (log_ell - 2.0, log_ell + 2.0))

    @property
    def bounds(self):
        """Boxes of `phi = (log sigma_x^2, log xi_u, log ell_u)`."""
        shift = diffusion_shift(self.T, self.q_x)
        lo, hi = self.diffusion_box or (DIFFUSION_BOX[0] * shift, DIFFUSION_BOX[1] * shift)
        return [(float(onp.log(lo)), float(onp.log(hi))), *self.theta_u_bounds]

    def _grid_fields(self, kappa, correction) -> dict:
        ts = self.node_times(kappa)
        layout = JointPrior(IWP(self.q_x, self.d_x), self.force.prior(1.0))     # for the projections only
        t_obs, z = self.observations()
        A, E = self._obs_matrix_and_projection(layout)
        obs = Measurement(A=A, z=jnp.asarray(z), z_t=jnp.asarray(t_obs), noise=jnp.asarray(self.noise_stds) ** 2)
        return {"kappa": kappa, "tspan": (float(self.tspan[0]), float(self.tspan[1])), "N": int(kappa * self.n_obs),
                "ode_measure": self._ode_measure(layout),
                "obs_model": prepare_observations([obs], E, jnp.asarray(ts)), "correction": correction,
                "u_col": self.state_dim, "ode_state_cols": tuple(range(self.order * self.d_x))}

    def discretize(self, kappa: int, correction=EK1) -> Grid:
        return Grid(**self._grid_fields(kappa, correction))

    def prior(self, sigma_x, log_theta_u):
        """The joint prior `JointPrior(IWP(q_x), Matern)`, traceable in the hyperparameters."""
        xi, ell = jnp.exp(log_theta_u[0]), jnp.exp(log_theta_u[1])
        return JointPrior(IWP(self.q_x, self.d_x, Xi=sigma_x * jnp.eye(self.d_x)), self.force.prior(xi, ell))

    def initial_law(self, log_theta_u):
        """`(mu0, P0_sqr)`, the consistent joint initial law, traceable in `theta_u`: the force block at its
        stationary law and mean zero, the state block at its Taylor-mode initial value, which carries the
        force's spread (`ode_filters.joint_taylor_initialization`). The state diffusion does not enter."""
        mu0, P0 = joint_taylor_initialization(self._unit_field, self._inits, prior=self.prior(1.0, log_theta_u),
                                              t0=self.t0, order=self.order)
        if self.p0_jitter:
            P0 = P0 + self.p0_jitter * jnp.eye(P0.shape[0])
        return mu0, P0

    def filter(self, grid: Grid, sigma_x, log_theta_u):
        """`(prior, FilterResult)` of the joint EK1 filter, without per-step calibration."""
        prior = self.prior(sigma_x, log_theta_u)
        mu0, P0 = self.initial_law(log_theta_u)
        result = gaussian_filter(mu0, P0, prior, grid.ode_measure, grid.ts, correction=grid.correction,
                                 calibration="none", obs_model=grid.obs_model)
        return prior, result
