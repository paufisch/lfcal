"""The benchmark systems of the paper: latent force problems with a known truth, their synthetic data, and
their `sigma_x` search windows.

A `BenchmarkSystem` is a `LatentForceProblem` whose force path is known. It draws its own data from the truth
(`observations`), its grids are `ScoredGrid`s that score a posterior against the truth at the observation
times or at the held-out interval midpoints (`score_at`), and its search box of `sigma_x^2` is the paper's
window for the system (`window`).

  duffing          Duffing, position observed, K = 32 on [0, 62.5], x(0) = (1, 0). The failure-directions
                   and force-information figures.
  duffing_accel    Duffing from rest, acceleration observed, K = 128, noise 1 % of the acceleration's rms.
                   The benchmark (Sec. "Systems").
  fhn              FitzHugh--Nagumo, v observed, K = 200 on [0, 200]. The sigma_x-sweep and
                   force-information figures.
  fhn_rough        FitzHugh--Nagumo, K = 300 on [0, 100], noise 20 % of std(v), and a rougher force
                   (length scale 5 instead of 10). The benchmark.
  lorenz96         Lorenz-96, D = 8, force on site 0, sites 0/2/4/6 observed, K = 200, noise 0.5. The
                   force-information figure.
  lorenz96_da      The same with K = 100 and noise std 1. The benchmark.
  sir              SIR with the transmission rate beta_0 exp(u), so the force enters through the state. The
                   infected fraction is observed daily on [0, 150], noise 10 % of its std.

Every truth force is a draw from the model's own Matern-5/2 prior, started from its stationary law, unless
`force_kind` asks for a force outside it (`FORCE_KINDS`). On Lorenz-96 the Matern draws of the recorded
datasets are read from `data/lorenz96_forces.npz` instead (`RECORDED_L96`). Seed 42 draws the force path and the
observation noise, unless a factory is given other seeds (`force_seed`, `obs_seed`). `nest=(K0, Kf)` nests the
observation noise across K (`nested_normal`). The truth is solved adaptively at tight tolerances, except on
Lorenz-96, which is chaotic: there it takes fixed Runge--Kutta steps from stored states (`RECORDED_L96`).
`q_x` follows the paper's rule `m + q_u - 1`: 3 on Duffing, 2 on the first-order systems.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import diffrax
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as onp

from lfcal.calibration import Calibration
from lfcal.force import MaternForce
from lfcal.paper.scores import posterior_scores
from lfcal.problem import EK1, Grid, LatentForceProblem, diffusion_shift, rk4_step

__all__ = ["DIVERGED", "EK1", "FORCE_KINDS", "SYSTEMS", "BenchmarkSystem", "ScoredGrid", "build", "calibration",
           "nested_normal", "score_indices", "sweep_window", "window"]

#: the seed of every force path and of the observation noise
SEED = 42
#: a posterior mean larger than this many times the truth's largest amplitude counts as diverged
DIVERGED = 10.0


# ------------------------------------------------------------ the benchmark system
def score_indices(kappa: int, K: int, score_at: str):
    """The grid nodes at which a posterior is scored.

    `score_at="obs"` scores at the `K` observation times. `"midpoints"` scores at the midpoints of the
    `K` observation intervals, which hold no data, and needs an even `kappa`.
    """
    if score_at == "obs":
        return (onp.arange(K) + 1) * kappa
    if kappa % 2:
        raise ValueError(f"midpoint scoring needs an even kappa; got {kappa}")
    return (2 * onp.arange(K) + 1) * (kappa // 2)


def nested_normal(key, n, nest=None):
    """Standard normal noise for `n` equispaced observations, optionally nested across `n`.

    Without `nest` this is `normal(key, (n,))`. With `nest = (K0, Kf)`, the observation at time `k T / n`
    takes the value of point `j = k Kf / n` on a fine grid of `Kf` points. The points on the `K0` grid keep
    the values of the un-nested draw at `K0`, and the others come from a second stream. So the data at
    `K0` do not change, and two designs see the same noise at the times they share: more data on the
    same experiment. Both `n` and `K0` must divide `Kf`.
    """
    if nest is None:
        return jrandom.normal(key, (n,))
    K0, Kf = nest
    if Kf % n or Kf % K0:
        raise ValueError(f"nested noise needs n | Kf and K0 | Kf, got n={n}, K0={K0}, Kf={Kf}")
    on_coarse = jrandom.normal(key, (K0,))
    on_fine = jrandom.normal(jrandom.fold_in(key, 1), (Kf,))
    j = jnp.arange(1, n + 1) * (Kf // n)
    stride = Kf // K0
    return jnp.where(j % stride == 0, on_coarse[jnp.clip(j // stride, 1, K0) - 1], on_fine[j - 1])


@dataclass(frozen=True, kw_only=True)
class ScoredGrid(Grid):
    """A benchmark system's grid: a `Grid` with the truth at the nodes, and the nodes the posterior is scored
    at. Its smoothed posteriors carry their scores (`posterior_extras`), and a posterior counts as diverged
    when it exceeds `DIVERGED` times the truth's largest amplitude."""

    x_true: onp.ndarray        # the scored state component at the nodes
    u_true: onp.ndarray        # the force (unit std) at the nodes
    score_idx: onp.ndarray     # the nodes the posterior is scored at

    @property
    def rms_x(self) -> float:
        return float(onp.sqrt(onp.mean(self.x_true[self.score_idx] ** 2)))

    @property
    def rms_u(self) -> float:
        return float(onp.sqrt(onp.mean(self.u_true[self.score_idx] ** 2)))

    @property
    def amp(self) -> float:
        """Largest true amplitude of the scored component."""
        return float(onp.abs(self.x_true).max())

    def has_diverged(self, max_abs_x) -> bool:
        """Whether a posterior mean that reaches `max_abs_x` in the scored component has diverged."""
        return not float(max_abs_x) < DIVERGED * self.amp

    def posterior_extras(self, m_s, var) -> dict:
        """The posterior's scores against the truth (`posterior_scores`)."""
        return posterior_scores(self, m_s, var)


@dataclass(frozen=True, kw_only=True, eq=False)
class BenchmarkSystem(LatentForceProblem):
    """A latent force problem with a known truth: the force path `force_path` (physical units), from which the
    system draws its data and scores its posteriors.

    The `n_obs = K` observations are equally spaced and exclude `t0`. The data are the noise-free observed
    channels of the reference solution (`solve`) plus noise drawn with `obs_seed`, nested across K with
    `nest`. The reference solution is adaptive at tolerances `rtol`, `atol`, or with `truth_steps` takes that
    many equal classical Runge--Kutta steps over the window, restarting from `truth_at_obs` at the observation
    times if given. The scored state component is `x_index`; `score_at` is "obs" or
    "midpoints" (held out). With `paper_window` the search box of `sigma_x^2` is the paper's window for the
    system (`window`), else the library's default.
    """

    name: str
    force_path: Callable
    obs_seed: int = SEED
    nest: tuple[int, int] | None = None
    rtol: float = 1e-10
    atol: float = 1e-12
    truth_steps: int | None = None
    truth_at_obs: onp.ndarray | None = None
    x_index: int = 0
    score_at: str = "obs"
    paper_window: bool = True
    # a memo of `solve`, which is a pure function of `ts`: the reference solution is expensive
    _solve_memo: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    @property
    def process_noise_mask(self):
        """Which first-order state components a process noise acts on (the classical LFM's `use_q`): the
        velocities of a second-order ODE, every component of a first-order one."""
        if self.order == 2:
            return onp.concatenate([onp.zeros(self.d_x), onp.ones(self.d_x)])
        return onp.ones(self.d_x)

    def _noise_keys(self):
        """One PRNG key per observed channel, as the recorded data were drawn."""
        if self.order == 2:
            return [jrandom.PRNGKey(self.obs_seed)]
        return list(jrandom.split(jrandom.PRNGKey(self.obs_seed), len(self.obs_indices)))

    def u_true(self, ts):
        """The true force at `ts`, in model units."""
        return onp.reshape(onp.asarray(self.force_path(jnp.asarray(ts))), (-1,)) / self.force_scale

    def solve(self, ts):
        """The true trajectory in first-order form at `ts`, shape `(len(ts), dim)`: at tight tolerances, or on
        the fixed steps of `truth_steps`."""
        ts = onp.asarray(ts, dtype=float)
        if self.truth_steps is not None:
            return self._solve_on_steps(ts)
        key = ts.tobytes()
        if key not in self._solve_memo:
            sol = diffrax.diffeqsolve(
                diffrax.ODETerm(lambda t, y, args: self.ode_rhs(y, self.force_path(t), t)), diffrax.Tsit5(),
                t0=self.tspan[0], t1=self.tspan[1], dt0=1e-3, y0=self.y0, saveat=diffrax.SaveAt(ts=jnp.asarray(ts)),
                stepsize_controller=diffrax.PIDController(rtol=self.rtol, atol=self.atol), max_steps=4_000_000)
            self._solve_memo[key] = onp.asarray(sol.ys)
        return self._solve_memo[key]

    @cached_property
    def _truth_on_steps(self):
        """The classical Runge--Kutta solution at the `truth_steps + 1` equally spaced points of the window,
        shape `(truth_steps + 1, dim)`, restarting from `truth_at_obs` at the observation times, which must lie
        on the steps."""
        (t0, t1), n = self.tspan, self.truth_steps
        half = jnp.asarray(t0 + (t1 - t0) * onp.arange(2 * n + 1) / (2 * n))    # the steps and their midpoints
        u = jnp.reshape(self.force_path(half), (-1,))
        restart = onp.zeros(n, dtype=bool)                    # entry i: the state after step i is given
        given = onp.zeros((n, len(self.y0)))
        if self.truth_at_obs is not None:
            pos = (onp.asarray(self.obs_times) - t0) / (t1 - t0) * n
            on_step = onp.rint(pos).astype(int)
            if onp.abs(pos - on_step).max() > 1e-6:
                raise ValueError("the observation times do not lie on the steps of the truth")
            restart[on_step - 1] = True
            given[on_step - 1] = self.truth_at_obs

        def step(y, x):
            t, u_start, u_mid, u_end, restart, given = x
            y = rk4_step(self.ode_rhs, y, t, (t1 - t0) / n, u_start, u_mid, u_end)
            y = jnp.where(restart, given, y)
            return y, y

        _, ys = jax.lax.scan(step, jnp.asarray(self.y0),
                             (half[:-1:2], u[:-1:2], u[1::2], u[2::2], jnp.asarray(restart), jnp.asarray(given)))
        return onp.concatenate([onp.asarray(self.y0)[None], onp.asarray(ys)])

    def _solve_on_steps(self, ts):
        """The fixed-step truth at `ts`: at a step its value there, between two steps one Runge--Kutta step
        of the remaining length from the step before."""
        (t0, t1), n = self.tspan, self.truth_steps
        pos = (ts - t0) / (t1 - t0) * n
        if pos.size and (pos.min() < -1e-6 or pos.max() > n + 1e-6):
            raise ValueError(f"truth asked outside its window {self.tspan}")
        ys = self._truth_on_steps
        near = onp.clip(onp.rint(pos), 0, n).astype(int)
        between = onp.abs(pos - near) > 1e-6
        out = ys[near]
        if between.any():
            i = onp.floor(pos[between]).astype(int)
            t = t0 + (t1 - t0) * i / n
            h = ts[between] - t

            def force(s):
                return jnp.reshape(self.force_path(jnp.asarray(s)), (-1,))

            partial = jax.vmap(lambda y, t, h, *u: rk4_step(self.ode_rhs, y, t, h, *u))
            out[between] = onp.asarray(partial(jnp.asarray(ys[i]), jnp.asarray(t), jnp.asarray(h), force(t),
                                               force(t + 0.5 * h), force(ts[between])))
        return out

    def observed_truth(self, ts):
        """The noise-free observed channels at `ts`, shape `(len(ts), d_obs)`."""
        ts = onp.asarray(ts, dtype=float)
        return onp.asarray(jax.vmap(self.observe)(jnp.asarray(self.solve(ts)), jnp.asarray(self.u_true(ts)),
                                                  jnp.asarray(ts)))

    @cached_property
    def _data(self):
        t = self.obs_times
        noise = onp.stack([onp.asarray(nested_normal(key, self.n_obs, self.nest)) for key in self._noise_keys()],
                          axis=1)
        return t, self.observed_truth(t) + onp.asarray(self.noise_stds) * noise

    def observations(self):
        """`(times, values)` of the synthetic data, values of shape `(K, d_obs)`."""
        return self._data

    @property
    def log_theta_u_true(self):
        return onp.array([onp.log(self.force.xi), onp.log(self.force.length_scale)])

    @property
    def bounds(self):
        if not self.paper_window:
            return super().bounds
        sigmas = window(self, SYSTEMS[self.name][1])
        return [(float(onp.log(sigmas[0])), float(onp.log(sigmas[-1]))), *self.theta_u_bounds]

    def discretize(self, kappa: int, correction=EK1) -> ScoredGrid:
        ts = self.node_times(kappa)
        return ScoredGrid(**self._grid_fields(kappa, correction), x_col=self.x_index,
                          x_true=self.solve(ts)[:, self.x_index], u_true=self.u_true(ts),
                          score_idx=score_indices(kappa, self.n_obs, self.score_at))


def _times(tspan, n_obs):
    """The `n_obs` equidistant observation times of `tspan`, without its start."""
    return onp.linspace(*tspan, n_obs + 1)[1:]


# ------------------------------------------------------------ the truth forces

#: the kinds of truth force: a draw of the model's Matern-5/2 prior, or, misspecified, a rougher
#: Matern-3/2 draw, a sinusoid, or a piecewise-constant force, each with the prior's marginal std
FORCE_KINDS = ("matern", "matern32", "sine", "step")


def _force(ell, std, tspan, seed, kind="matern"):
    """The true force path in physical units; the model is always Matern-5/2 with length scale `ell`.

    The truth is of `kind` (`FORCE_KINDS`). The
    sinusoid has an angular frequency in [0.7, 1.4] / ell and a random phase. The piecewise-constant force
    switches at the times of a Poisson process of mean spacing 2 ell, to levels drawn from N(0, std^2).
    """
    if kind in ("matern", "matern32"):
        q = 2 if kind == "matern" else 1
        return MaternForce(q=q, length_scale=ell, marginal_std=std).sample(tspan, seed=seed)
    rng = onp.random.default_rng(seed)
    ts = onp.linspace(*tspan, 4001)
    scale = float(ell)
    if kind == "sine":
        omega, phase = rng.uniform(0.7, 1.4) / scale, rng.uniform(0.0, 2.0 * onp.pi)
        values = std * onp.sqrt(2.0) * onp.sin(omega * ts + phase)
    elif kind == "step":
        n = rng.poisson((tspan[1] - tspan[0]) / (2.0 * scale))
        switches = onp.sort(rng.uniform(*tspan, n))
        values = rng.normal(0.0, std, n + 1)[onp.searchsorted(switches, ts)]
    else:
        raise ValueError(f"unknown force kind {kind!r}")
    ts_j, values_j = jnp.asarray(ts), jnp.asarray(values)
    return lambda t: jnp.interp(jnp.asarray(t), ts_j, values_j)


def _relative_noise(build, rel_noise, component, tspan, *, obs_seed, nest):
    """`build(noise_std, ...)` with the noise std `rel_noise` times the std of the state `component` of the
    true trajectory (on 2001 points of `tspan`), which a probe built with unit noise provides."""
    spread = float(onp.std(build(1.0).solve(onp.linspace(*tspan, 2001))[:, component]))
    return build(rel_noise * spread, obs_seed=obs_seed, nest=nest)


# ------------------------------------------------------------------- Duffing

#: `x'' = -x - 2 zeta x' - eps x^3 + u`
DUFFING_ZETA, DUFFING_EPS = 0.1, 0.5
DUFFING_TSPAN = (0.0, 62.5)
DUFFING_ELL, DUFFING_STD = float(onp.sqrt(10.0)), 0.3162      # length scale (3.16), marginal std


def duffing_field(x, dx, u, *, t):
    return -x - 2.0 * DUFFING_ZETA * dx - DUFFING_EPS * x ** 3 + u


def _duffing(name, *, n_obs, noise_std, x0, obs_kind, force_seed=SEED, obs_seed=SEED, nest=None, force_kind="matern"):
    path = _force(DUFFING_ELL, DUFFING_STD, DUFFING_TSPAN, force_seed, force_kind)
    return BenchmarkSystem(name=name, vector_field=duffing_field, order=2, t0=DUFFING_TSPAN[0],
                           times=_times(DUFFING_TSPAN, n_obs), length_scale=DUFFING_ELL, force_path=path,
                           force_scale=DUFFING_STD, x0=x0, obs_derivative={"position": 0, "acceleration": 2}[obs_kind],
                           noise_std=noise_std, obs_seed=obs_seed, nest=nest, rtol=1e-11, atol=1e-11)


def duffing(name="duffing", n_obs=32, **kw):
    """Duffing on [0, 62.5], position observed, x(0) = (1, 0), noise std `0.01 * 2.8 / sqrt(2)`."""
    return _duffing(name, n_obs=n_obs, noise_std=0.01 * 2.8 / onp.sqrt(2.0), x0=1.0, obs_kind="position", **kw)


def duffing_accel(name="duffing_accel", n_obs=128, rel_noise=0.01, **kw):
    """Duffing from rest with the acceleration observed, as in joint input-state estimation from
    accelerometers (Rogers et al. 2020), noise std `rel_noise` times the rms of the noise-free acceleration."""
    probe = _duffing(name, n_obs=n_obs, noise_std=1.0, x0=0.0, obs_kind="acceleration", **kw)
    rms = float(onp.sqrt(onp.mean(probe.observed_truth(probe.node_times(64)) ** 2)))
    return _duffing(name, n_obs=n_obs, noise_std=rel_noise * rms, x0=0.0, obs_kind="acceleration", **kw)


# ------------------------------------------------------- FitzHugh--Nagumo

#: `v' = v - v^3/3 - w + I_bar + u`, `w' = eps (v + a - b w)`
FHN_A, FHN_B, FHN_EPS, FHN_I_BAR = 0.7, 0.8, 0.08, 0.5
FHN_X0 = jnp.array([-1.0, -0.5])
FHN_ELL, FHN_STD = 10.0, 0.35           # length scale, marginal std
FHN_ELL_ROUGH = 5.0                      # the rougher force of `fhn_rough`


def fhn_field(x, u, *, t):
    u = jnp.reshape(jnp.asarray(u), ())
    v, w = x[0], x[1]
    return jnp.array([v - v ** 3 / 3.0 - w + FHN_I_BAR + u, FHN_EPS * (v + FHN_A - FHN_B * w)])


def _fhn(name, tspan, n_obs, noise_std, ell, path, *, obs_seed=SEED, nest=None):
    return BenchmarkSystem(name=name, vector_field=fhn_field, x0=FHN_X0, t0=tspan[0], times=_times(tspan, n_obs),
                           obs_indices=(0,), noise_std=(noise_std,), length_scale=ell, force_path=path,
                           force_scale=FHN_STD, obs_seed=obs_seed, nest=nest)


def fhn(name="fhn", n_obs=200, force_seed=SEED, obs_seed=SEED, nest=None, force_kind="matern"):
    """FitzHugh--Nagumo on [0, 200], v observed with noise std 0.1."""
    tspan = (0.0, 200.0)
    path = _force(FHN_ELL, FHN_STD, tspan, force_seed, force_kind)
    return _fhn(name, tspan, n_obs, 0.1, FHN_ELL, path, obs_seed=obs_seed, nest=nest)


def fhn_rough(name="fhn_rough", n_obs=300, rel_noise=0.2, force_seed=SEED, obs_seed=SEED, nest=None,
              force_kind="matern"):
    """FitzHugh--Nagumo with the observation design of Arnold & Lloyd (2018): [0, 100], three observations per
    time unit, noise std `rel_noise` times std(v). The force is rougher than `fhn`'s, with `length_scale`
    `FHN_ELL_ROUGH` in the truth and in the model prior. It is drawn on [0, 200] and cut to [0, 100]."""
    tspan = (0.0, 100.0)
    path = _force(FHN_ELL_ROUGH, FHN_STD, (0.0, 200.0), force_seed, force_kind)

    def build(noise_std, **kw):
        return _fhn(name, tspan, n_obs, noise_std, FHN_ELL_ROUGH, path, **kw)

    return _relative_noise(build, rel_noise, 0, tspan, obs_seed=obs_seed, nest=nest)


# ------------------------------------------------------------------ Lorenz-96

L96_F = 8.0                              # forcing constant; D = 8 sites (`L96_X0`)
#: a point on the attractor (reached at t = 20 from F with a 0.01 kick on site 0)
L96_X0 = jnp.array([7.380513, 4.000148, 5.233222, 5.023650, -5.383174, 3.057213,
                    0.597713, 3.068415])
L96_TSPAN = (0.0, 10.0)
L96_OBS_SITES = (0, 2, 4, 6)
L96_ELL, L96_STD = 1.0, 2.5
#: Lorenz-96 is chaotic: a change of its truth in the last bit grows to the size of the attractor within the
#: window, and an adaptive solver makes it larger by changing its steps. The truth therefore takes
#: `L96_TRUTH_STEPS` equal Runge--Kutta steps, on which every grid of the paper lies. With `RECORDED_L96` the
#: datasets are those of the recorded results: the forces drawn with ode-filters 0.7.3
#: (`data/lorenz96_forces.npz`, keys `{kind}_s{seed}`; the sinusoid and step forces are drawn by numpy as
#: before) and the states of their adaptive truth at the observation times (`data/lorenz96_truth.npz`, keys
#: `{kind}_s{seed}_K{n_obs}`), from which the steps restart. Otherwise the forces are drawn afresh and the
#: truth runs from `L96_X0` alone.
RECORDED_L96 = True
L96_FORCES = Path(__file__).with_name("data") / "lorenz96_forces.npz"
L96_TRUTH = Path(__file__).with_name("data") / "lorenz96_truth.npz"
L96_TRUTH_STEPS = 128_000


def lorenz96_field(x, u, *, t):
    """`x_i' = (x_{i+1} - x_{i-2}) x_{i-1} - x_i + F`, the force added to site 0."""
    u = jnp.reshape(jnp.asarray(u), ())
    dx = (jnp.roll(x, -1) - jnp.roll(x, 2)) * jnp.roll(x, 1) - x + L96_F
    return dx.at[0].add(u)


def _recorded_lorenz96_force(kind, seed):
    """The truth force of Lorenz-96 of `kind` and `seed` as the recorded results drew it, or None."""
    with onp.load(L96_FORCES) as z:
        key = f"{kind}_s{seed}"
        if key not in z.files:
            return None
        values = z[key]
    ts, values = onp.linspace(*L96_TSPAN, len(values)), jnp.asarray(values)
    return lambda t: jnp.interp(jnp.asarray(t), ts, values)


def _recorded_lorenz96_truth(kind, seed, n_obs):
    """The states of the recorded Lorenz-96 truth of `kind` and `seed` at the `n_obs` observation times, shape
    `(n_obs, 8)`, or None."""
    with onp.load(L96_TRUTH) as z:
        key = f"{kind}_s{seed}_K{n_obs}"
        return z[key] if key in z.files else None


def lorenz96(name="lorenz96", n_obs=200, noise_std=0.5, force_seed=SEED, obs_seed=SEED, nest=None,
             force_kind="matern"):
    """Lorenz-96 on [0, 10], every other site observed."""
    path = _recorded_lorenz96_force(force_kind, force_seed) if RECORDED_L96 else None
    if path is None:
        path = _force(L96_ELL, L96_STD, L96_TSPAN, force_seed, force_kind)
    truth = _recorded_lorenz96_truth(force_kind, force_seed, n_obs) if RECORDED_L96 else None
    return BenchmarkSystem(name=name, vector_field=lorenz96_field, x0=L96_X0, t0=L96_TSPAN[0],
                           times=_times(L96_TSPAN, n_obs), obs_indices=L96_OBS_SITES,
                           noise_std=tuple(noise_std for _ in L96_OBS_SITES), length_scale=L96_ELL, force_path=path,
                           force_scale=L96_STD, obs_seed=obs_seed, nest=nest, truth_steps=L96_TRUTH_STEPS,
                           truth_at_obs=truth)


def lorenz96_da(name="lorenz96_da", n_obs=100, noise_std=1.0, **kw):
    """Lorenz-96 with the data-assimilation noise (std 1), observed every 0.1."""
    return lorenz96(name, n_obs=n_obs, noise_std=noise_std, **kw)


# ------------------------------------------------------------------ SIR

#: `S' = -beta S I`, `I' = beta S I - gamma I` with the transmission rate `beta = beta_0 exp(u)`: the
#: force enters through the state, not additively. Fractions of the population, time in days.
SIR_BETA0, SIR_GAMMA = 0.15, 0.1
SIR_X0 = jnp.array([0.999, 0.001])
SIR_TSPAN = (0.0, 150.0)
SIR_ELL, SIR_STD = 10.0, 0.25            # length scale (days), marginal std


def sir_field(x, u, *, t):
    u = jnp.reshape(jnp.asarray(u), ())
    infections = SIR_BETA0 * jnp.exp(u) * x[0] * x[1]
    return jnp.array([-infections, infections - SIR_GAMMA * x[1]])


def sir(name="sir", n_obs=150, rel_noise=0.1, force_seed=SEED, obs_seed=SEED, nest=None, force_kind="matern"):
    """SIR on [0, 150] days with a time-varying transmission rate, the infected fraction observed daily with
    noise std `rel_noise` times its std."""
    path = _force(SIR_ELL, SIR_STD, SIR_TSPAN, force_seed, force_kind)

    def build(noise_std, **kw):
        return BenchmarkSystem(name=name, vector_field=sir_field, x0=SIR_X0, t0=SIR_TSPAN[0],
                               times=_times(SIR_TSPAN, n_obs), obs_indices=(1,), x_index=1, noise_std=(noise_std,),
                               length_scale=SIR_ELL, force_path=path, force_scale=SIR_STD, **kw)

    return _relative_noise(build, rel_noise, 1, SIR_TSPAN, obs_seed=obs_seed, nest=nest)


# ------------------------------------------------------------ sigma_x windows

#: `sigma_x` windows, log-spaced, as first set: for `q_x = 2` on the unit clock `tau = t / T`. `window`
#: converts them to the system's order and to physical time.
_UNIT_CLOCK_WINDOWS = {
    "default": onp.logspace(3, 13, 31).tolist(),
    "fhn": onp.logspace(6, 20, 43).tolist(),
    "lorenz96": onp.logspace(6, 20, 43).tolist(),
    "sir": onp.logspace(4, 20, 49).tolist(),
}
#: per system: factory, window key
SYSTEMS = {
    "duffing": (duffing, "default"),
    "duffing_accel": (duffing_accel, "default"),
    "fhn": (fhn, "fhn"),
    "fhn_rough": (fhn_rough, "fhn"),
    "lorenz96": (lorenz96, "lorenz96"),
    "lorenz96_da": (lorenz96_da, "lorenz96"),
    "sir": (sir, "sir"),
}
#: top of the sweep window (log10, unit clock) where the state prior runs above q_x = 2: there the
#: weak-enforcement failure sits further above the pick
_SWEEP_TOP = {"duffing": 24.0}


def build(name, *, score_at="obs", **kw):
    """The system `name`, scored at the observation times or at the held-out midpoints."""
    return dataclasses.replace(SYSTEMS[name][0](**kw), score_at=score_at)


def calibration(name, K, kappa, **kw) -> Calibration:
    """The `Calibration` of the system `name` with `K` observations on the grid of density `kappa`, scored at
    the held-out interval midpoints. `kw` go to the system's factory (`force_seed`, `obs_seed`, `nest`)."""
    return Calibration(build(name, score_at="midpoints", n_obs=K, **kw), kappa)


def window(system, key):
    """The window `key` on `system`'s order: the recorded values times `diffusion_shift(T, q_x)`,
    `T^(2 (q_x - 2)) (1 / T)^(2 q_x + 1)`."""
    shift = diffusion_shift(system.T, system.q_x)
    return [float(v) * shift for v in _UNIT_CLOCK_WINDOWS[key]]


def sweep_window(name, system):
    """`window`, extended upward at the same log spacing for a sweep that must show both failure directions
    (Duffing at `q_x = 3`)."""
    w = window(system, SYSTEMS[name][1])
    top = _SWEEP_TOP.get(name)
    if top is None or int(system.q_x) <= 2:
        return w
    top += (2 * int(system.q_x) + 1) * onp.log10(1.0 / float(system.T))
    if onp.log10(w[-1]) >= top:
        return w
    step = onp.log10(w[1]) - onp.log10(w[0])
    extra = onp.arange(onp.log10(w[-1]) + step, top + 1e-9, step)
    return list(w) + [float(10 ** e) for e in extra]
