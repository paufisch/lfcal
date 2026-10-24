"""The algorithm of the paper: the latent force ODE filter with the diffusion selected by a criterion J, the fit
on one grid, and the loop over grids with the stability guard and the ODE-consistency check.

`calibrate(problem, criterion)` walks the grid densities `kappa = 2, 4, ..., 32` (`warm_path`):

  start    On the first grid, and after an unstable fit, the hyperparameters start at `phi^(0)` (the
           fractions `START` of their boxes) and the linearisation at one EK1 pass. The grid is skipped
           unless every step of that pass is stable (`guard_ek1`).
  fit      The IEKS alternation (`fit`): the hyperparameters at the frozen linearisation, then a
           re-linearisation at the smoothed mean, until `ell_OBS` at the cycle's optimum changes by less
           than `TAU_L` nats or after `N_MAX` cycles. Every cycle's frozen pass must be stable
           (`guard_affine`), otherwise the fit counts as unstable.
  warm     Every later grid starts from the previous grid's fit, its trajectory interpolated onto the
           finer grid (`prolong`).
  accept   A grid is accepted when the force mean at the observation times moved by less than `TAU_U`
           posterior sds from the previous grid's fit, and the fit reproduces the data through the ODE
           (`Calibration.consistency <= TAU_R`). Without an accepted grid the force is reported as not
           identified.

The criterion says how `sigma_x^2` is chosen (`CRITERIA`, the paper's Table of criteria). With `"obs"`, all
of `phi = (log sigma_x^2, log xi_u, log ell_u)` are fitted by `ell_OBS`. Every other criterion fits
`log sigma_x^2` first, then the force hyperparameters by `ell_OBS`. At the frozen IEKS linearisation it
maximises the mutual information `I(U; Y | Z)` (`"mi"`, the default), the directed information (`"di"`) or
`log p(Y | Z)` (`"fenrir_ieks"`) by L-BFGS-B.

Stability: the classical local error estimate of each filter step (the step's defect against the prior's
process noise, error on the state, atol = rtol = 1) must not exceed `GUARD`.
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import cached_property
from typing import Any, NamedTuple

import jax.numpy as jnp
import numpy as onp

from lfcal.calibration import Calibration, lbfgs
from lfcal.moments import cov

__all__ = ["CRITERIA", "GUARD", "KAPPAS", "START", "Calibrated", "Fit", "GridFit", "Posterior", "calibrate",
           "cold_grid", "fit", "force_at_obs", "guard_affine", "guard_ek1", "ode_evidence_pick", "prolong",
           "start_point", "warm_grid", "warm_path"]

#: how `sigma_x^2` is selected: the observation evidence (`"obs"`), the mutual information `I(U; Y | Z)`
#: (`"mi"`), the directed information (`"di"`) and `log p(Y | Z)` at the IEKS linearisation (`"fenrir_ieks"`)
CRITERIA = ("obs", "mi", "di", "fenrir_ieks")
#: phi^(0) as fractions of the boxes of (log sigma_x^2, log xi_u, log ell_u), the same for every criterion:
#: sigma_x^2 on the weak side, and a large, rough force (xi_u = e xi_u*, ell_u ~ 0.45 ell_u*). From a small,
#: smooth force, ell_OBS tends to switch the force off.
START = (0.8, 0.70, 0.30)
#: the grid of the `ell_ODE` pick of sigma_x^2 (`ode_evidence_pick`): this many points over the log window
PICK_GRID = 17
KAPPAS = (2, 4, 8, 16, 32)
#: acceptance: the force moved by less than TAU_U sds, and the consistency statistic (about 1 for a
#: calibrated fit) is at most TAU_R. The alternation stops when ell_OBS changes by less than TAU_L nats.
TAU_U, TAU_R, TAU_L = 1.0, 1.5, 1e-3
#: the most IEKS alternation cycles per grid
N_MAX = 5
#: the largest local error a stable filter step may have
GUARD = 1.0
#: filter passes per evaluation of a criterion: four for I(U; Y | Z), two for the directed information, two
#: for log p(Y | Z)
PASSES_PER_EVALUATION = {"mi": 4, "di": 2, "fenrir_ieks": 2}


# ------------------------------------------------------------------ stability guard
def local_error(H, c, m_pred, Q, cols):
    """The weighted local error of each step, on the state columns `cols`.

    `H` is `(N, d, D)`, `c` is `(N, d)` and the predicted means `m_pred` are `(N, D)`. A step whose defect or
    defect covariance is not finite has an infinite error.
    """
    defect = -(onp.einsum("nij,nj->ni", H, m_pred) + c)
    HQH = onp.einsum("nij,jk,nlk->nil", H, Q, H)
    finite = onp.isfinite(defect).all(axis=1) & onp.isfinite(HQH).all(axis=(1, 2))
    s2 = onp.full(len(defect), onp.inf)
    d = defect.shape[1]
    # the minimum-norm solution of `HQH s = defect`, as `lstsq` gives it: singular values below `d eps` times
    # the largest count as zero
    solution = onp.einsum("nij,nj->ni", onp.linalg.pinv(HQH[finite], rcond=d * onp.finfo(float).eps),
                          defect[finite])
    s2[finite] = onp.maximum(onp.einsum("ni,ni->n", defect[finite], solution) / d, 0.0)
    err = onp.sqrt(s2[:, None] * onp.diag(Q)[cols][None, :])
    with onp.errstate(invalid="ignore"):                    # inf / inf on a diverged step: inf below
        weighted = err / (1.0 + onp.abs(m_pred[:, cols]))
    out = onp.sqrt(onp.mean(weighted ** 2, axis=1))
    out[~onp.isfinite(out)] = onp.inf
    return out


def guard_ek1(cal, phi):
    """The largest local error of the EK1 pass at `phi`, and its predicted means (the first linearisation)."""
    m_pred = cal.ek1_predicted_means(jnp.asarray(phi))
    if not bool(jnp.all(jnp.isfinite(m_pred))):
        return float("inf"), m_pred
    surrogate = cal.linearize(m_pred)
    Q = onp.asarray(cov(cal.transition(jnp.asarray(phi)).Q_sqr))
    err = local_error(onp.asarray(surrogate.H), onp.asarray(surrogate.c), onp.asarray(m_pred), Q,
                      list(cal.grid.ode_state_cols))
    return float(onp.max(err)), m_pred


def guard_affine(cal, phi, traj):
    """The largest local error of the pass at `phi` linearised along `traj`."""
    phi = jnp.asarray(phi)
    surrogate = cal.linearize(jnp.asarray(traj))
    m_f = onp.asarray(cal.data_pass(phi, surrogate).m)
    if not onp.all(onp.isfinite(m_f)):
        return float("inf")
    transition = cal.transition(phi)
    A, b, Q = onp.asarray(transition.A), onp.asarray(transition.b), onp.asarray(cov(transition.Q_sqr))
    m_pred = m_f[:-1] @ A.T + b
    return float(onp.max(local_error(onp.asarray(surrogate.H), onp.asarray(surrogate.c), m_pred, Q,
                                     list(cal.grid.ode_state_cols))))


# ------------------------------------------------------------------ the fit
def _sigma_by_gradient(cal, phi, surrogate, criterion):
    """The `log sigma_x^2` that maximises `I(U; Y | Z)` (`"mi"`), the directed information (`"di"`) or
    `log p(Y | Z)` (`"fenrir_ieks"`) at the frozen linearisation `surrogate`, by L-BFGS-B from the current
    value, `phi_u` held. Returns it and the filter passes spent."""
    phi = jnp.asarray(phi, dtype=float)
    vg = (cal.frozen_data_likelihood_value_and_grad if criterion == "fenrir_ieks"
          else cal.information_value_and_grad[criterion])

    def negative(x):
        v, g = vg(jnp.asarray(x[0]), phi, surrogate)
        return -float(v), onp.array([-float(g)])

    res = lbfgs(negative, [float(phi[0])], [cal.bounds[0]])
    return float(res.x[0]), PASSES_PER_EVALUATION[criterion] * res.n_evals


def _hyperparameters(cal, phi, surrogate, criterion, free):
    """One cycle's hyperparameters at the frozen linearisation `surrogate`, over the coordinates `free`.

    Returns `(phi, -ell_OBS at phi, filter passes spent)`.
    """
    if criterion == "obs" or 0 not in free:
        res = cal.fit_hyperparameters(surrogate, phi, free)
        return res.x, res.value, res.n_evals
    start = phi.copy()
    start[0], passes = _sigma_by_gradient(cal, start, surrogate, criterion)
    res = cal.fit_hyperparameters(surrogate, start, [i for i in free if i != 0])
    return res.x, res.value, passes + res.n_evals


@dataclass(frozen=True)
class Fit:
    """The result of `fit`: hyperparameters, trajectory, cost in filter passes, cycles, and whether a frozen
    pass was unstable (then `phi` is the unstable cycle's and `step` is NaN)."""

    phi: onp.ndarray
    traj: Any
    passes: int
    cycles: int
    unstable: bool
    err_max: float
    step: float = float("nan")


def fit(cal, phi, traj, criterion="obs", free=(0, 1, 2)) -> Fit:
    """The IEKS alternation from `(phi, traj)`, guarded: the hyperparameters at the frozen linearisation,
    then a re-linearisation at the smoothed mean, until `ell_OBS` settles."""
    passes, err_max = 0, 0.0
    phi, free = onp.asarray(phi, float), list(free)
    nll_prev, step = None, float("inf")
    for cycle in range(1, N_MAX + 1):
        surrogate = cal.linearize(jnp.asarray(traj))
        phi_new, nll, spent = _hyperparameters(cal, phi, surrogate, criterion, free)
        err = guard_affine(cal, phi_new, traj)
        passes += spent + 1
        err_max = max(err_max, err)
        if err > GUARD:
            return Fit(phi_new, traj, passes, cycle, unstable=True, err_max=err_max)
        phi = phi_new
        traj, step = cal.relinearize(jnp.asarray(phi), jnp.asarray(traj))
        passes += 1
        if nll_prev is not None and abs(nll - nll_prev) < TAU_L:
            break
        nll_prev = nll
    return Fit(phi, onp.asarray(traj), passes, cycle, unstable=False, err_max=err_max, step=float(step))


def prolong(traj, factor=2, *, from_t0=False):
    """A trajectory on the grid `factor` times finer, by linear interpolation in time.

    `traj` holds nodes 1..N (a linearisation: node 0 is never read, and the finer grid's first node takes
    the coarse node 1), or nodes 0..N with `from_t0`.
    """
    traj = onp.asarray(traj)
    first = 0 if from_t0 else 1
    coarse = onp.arange(first, first + len(traj), dtype=float)
    fine = onp.arange(first, factor * coarse[-1] + 1, dtype=float) / factor
    return onp.stack([onp.interp(fine, coarse, traj[:, j]) for j in range(traj.shape[1])], axis=1)


def start_point(cal, fractions=START):
    """The point at `fractions` of `cal`'s boxes."""
    return onp.array([lo + f * (hi - lo) for (lo, hi), f in zip(cal.bounds, fractions, strict=True)])


# ------------------------------------------------------------------ the force estimate
class ForceAtObs(NamedTuple):
    """The force mean at the observation times, its rms posterior sd at the interval midpoints, and
    `ell_OBS`."""

    u_obs: onp.ndarray
    sd: float
    ll_obs: float


def force_at_obs(cal, phi, traj) -> ForceAtObs:
    post = cal.posterior(jnp.asarray(phi), jnp.asarray(traj))
    g = cal.grid
    u_obs = onp.asarray(post["m_s"])[g.obs_nodes, g.u_col]
    sd = float(onp.sqrt(onp.mean(onp.asarray(post["var"])[g.mid_idx, g.u_col])))
    return ForceAtObs(u_obs, sd, float(post["ll_obs"]))


def ode_evidence_pick(cal, phi):
    """The `log sigma_x^2` with the largest residual evidence `ell_ODE` on a `PICK_GRID`-point grid
    over the window, at the linearisation of the EK1 pass at `phi`, `phi_u` held. This is the ODE filters'
    default calibration, used as a starting point."""
    _, traj = guard_ek1(cal, phi)
    surrogate = cal.linearize(jnp.asarray(traj))
    grid = onp.linspace(*cal.bounds[0], PICK_GRID)
    phi = jnp.asarray(onp.asarray(phi, float))
    lls = onp.array([float(cal.ode_evidence(phi.at[0].set(s), surrogate)) for s in grid])
    if not onp.isfinite(lls).any():
        return float(phi[0])
    return float(grid[int(onp.argmax(onp.where(onp.isfinite(lls), lls, -onp.inf)))])


# ------------------------------------------------------------------ the grid loop
def warm_path(kappas, cold: Callable, warm: Callable, failed: Callable) -> Iterator:
    """Yield `(kappa, previous, outcome)` along the grid ladder `kappas`.

    Each grid starts warm from the previous grid's outcome (`warm(kappa, previous)`), and cold on the first
    grid and after a failure (`cold(kappa)`, `previous` None). The caller may stop early: the grids are
    fitted lazily, one per step.
    """
    previous = None
    for kappa in kappas:
        outcome = cold(kappa) if previous is None else warm(kappa, previous)
        yield kappa, previous, outcome
        previous = None if failed(outcome) else outcome


@dataclass(frozen=True)
class GridFit:
    """One grid of the ladder: its calibration, how it started, and its fit (None when the cold start was
    unstable). `passes` counts the grid's filter passes, the cold start's EK1 pass included."""

    cal: Calibration
    warm: bool
    passes: int
    t0: float
    fit: Fit | None = None
    start_err: float | None = None

    @property
    def failed(self) -> bool:
        return self.fit is None or self.fit.unstable

    @cached_property
    def consistency(self) -> float:
        return self.cal.consistency(self.fit.phi, self.fit.traj)

    def log_entry(self) -> dict:
        """The grid's line of the algorithm's per-grid log."""
        entry = {"kappa": self.cal.kappa}
        if self.start_err is not None:
            entry["start_err"] = self.start_err
        if self.fit is None:
            return {**entry, "stop": "start unstable"}
        entry.update(cycles=self.fit.cycles, unstable=self.fit.unstable, err_max=self.fit.err_max, warm=self.warm)
        return {**entry, "stop": "fit unstable"} if self.fit.unstable else entry


def _cold_start(cal, start=None):
    """`(phi^(0), the EK1 pass's predicted means, its largest local error)`.

    `start`: fractions of the boxes instead of `START`, or `"ode"`: `START` with `log sigma_x^2` at the
    `ell_ODE` pick (`ode_evidence_pick`).
    """
    phi0 = start_point(cal, START if start in (None, "ode") else start)
    if start == "ode":
        phi0[0] = ode_evidence_pick(cal, phi0)
    err, traj = guard_ek1(cal, phi0)
    return phi0, traj, err


def cold_grid(cal, criterion, start=None) -> GridFit:
    """The fit on `cal`'s grid from `phi^(0)` and one EK1 pass, unless that pass is unstable."""
    t0 = time.time()
    phi0, traj0, err = _cold_start(cal, start)
    if err > GUARD:
        return GridFit(cal, warm=False, passes=1, t0=t0, start_err=err)
    result = fit(cal, phi0, traj0, criterion)
    return GridFit(cal, warm=False, passes=1 + result.passes, t0=t0, fit=result, start_err=err)


def warm_grid(cal, previous: GridFit, criterion, force_start=None) -> GridFit:
    """The fit on `cal`'s grid from the previous grid's fit. With `force_start` (fractions of the boxes), the
    force hyperparameters restart from there, and only the linearisation and sigma_x^2 are warm."""
    t0 = time.time()
    phi = onp.array(previous.fit.phi)
    if force_start is not None:
        phi[1:] = start_point(cal, force_start)[1:]
    result = fit(cal, phi, prolong(previous.fit.traj), criterion)
    return GridFit(cal, warm=True, passes=result.passes, t0=t0, fit=result)


def _force_moved(previous: GridFit, current: GridFit) -> float:
    """How far the force mean at the observation times moved between two grids' fits, in posterior sds."""
    u_prev = force_at_obs(previous.cal, previous.fit.phi, previous.fit.traj).u_obs
    u_cur, sd, _ = force_at_obs(current.cal, current.fit.phi, current.fit.traj)
    return float(onp.sqrt(onp.mean((u_cur - u_prev) ** 2)) / sd)


def check_kappas(kappas):
    """`kappas` must double from grid to grid, from an even density, and hold at least two grids: a grid is
    accepted against the previous one."""
    kappas = tuple(int(k) for k in kappas)
    if len(kappas) < 2 or kappas[0] % 2 or any(b != 2 * a for a, b in itertools.pairwise(kappas)):
        raise ValueError(f"kappas must be at least two even grid densities, each double the last; got {kappas}")
    return kappas


class Posterior(NamedTuple):
    """The smoothed posterior on the nodes `ts` of the reported grid, in physical units: the mean and sd of the
    state `x`, shape `(N + 1, d)`, and of the force, shape `(N + 1,)`."""

    ts: onp.ndarray
    state_mean: onp.ndarray
    state_sd: onp.ndarray
    force_mean: onp.ndarray
    force_sd: onp.ndarray


@dataclass(frozen=True)
class Calibrated:
    """The result of `calibrate`: whether a grid was `accepted`, the `reported` grid's fit (the accepted grid,
    else the last warm grid that was fitted; None if there is none), the cost in filter passes weighted by
    their grid density, and the per-grid `log`.

    The fit's hyperparameters, in physical units: the state diffusion `sigma_x2`, the force's marginal std
    `force_std` and its length scale `length_scale`. `phi` holds their logs as fitted, `(log sigma_x^2,
    log xi_u, log ell_u)` with `force_std = force_scale sqrt(xi_u)`, and `traj` the converged linearisation
    (nodes 1..N, model units). It holds jitted functions, so `posterior()` is what to keep.
    """

    problem: Any
    criterion: str
    accepted: bool
    reported: GridFit | None
    cost: int
    log: list

    @property
    def failed(self) -> bool:
        """Whether no warm grid was fitted, so there is no fit to report."""
        return self.reported is None

    def _fit(self) -> GridFit:
        if self.reported is None:
            raise ValueError("no grid was fitted stably: see `log` for each grid's stop")
        return self.reported

    @property
    def calibration(self) -> Calibration:
        return self._fit().cal

    @property
    def kappa(self) -> int:
        return self._fit().cal.kappa

    @property
    def phi(self) -> onp.ndarray:
        return self._fit().fit.phi

    @property
    def traj(self):
        return self._fit().fit.traj

    @property
    def consistency(self) -> float:
        """The ODE-consistency statistic `r_kappa` of the fit (about 1 when the fit reproduces the data)."""
        return self._fit().consistency

    @property
    def sigma_x2(self) -> float:
        return float(onp.exp(self.phi[0]))

    @property
    def force_std(self) -> float:
        return float(self.problem.force_scale * onp.exp(0.5 * self.phi[1]))

    @property
    def length_scale(self) -> float:
        return float(onp.exp(self.phi[2]))

    def posterior(self) -> Posterior:
        """The smoothed posterior of the fit on its grid (`Posterior`)."""
        cal = self.calibration
        post = cal.posterior(jnp.asarray(self.phi), jnp.asarray(self.traj))
        m_s, sd = onp.asarray(post["m_s"]), onp.sqrt(onp.asarray(post["var"]))
        d, u, scale = self.problem.d_x, cal.grid.u_col, self.problem.force_scale
        return Posterior(cal.grid.ts, m_s[:, :d], sd[:, :d], scale * m_s[:, u], scale * sd[:, u])


def calibrate(problem, criterion="mi", *, kappas=KAPPAS, start=None, reset_force=True) -> Calibrated:
    """The paper's algorithm on the `LatentForceProblem` `problem`, the diffusion selected by `criterion`
    (`CRITERIA`; `"mi"`, the mutual information `I(U; Y | Z)`, by default) along the grid densities `kappas`.

    `start`: the starting point (fractions of the boxes instead of `START`, or `"ode"`: `START` with
    `log sigma_x^2` at the `ell_ODE` pick, `ode_evidence_pick`). `reset_force` (the default): a warm grid
    keeps the previous grid's linearisation and sigma_x^2 but restarts the force hyperparameters from the
    start. Without it, a warm grid starts from the whole previous fit.
    """
    if criterion not in CRITERIA:
        raise ValueError(f"criterion must be one of {CRITERIA}, got {criterion!r}")
    kappas = check_kappas(kappas)
    force_start = (START if start in (None, "ode") else start) if reset_force else None
    ladder = warm_path(kappas,
                       cold=lambda kappa: cold_grid(Calibration(problem, kappa), criterion, start),
                       warm=lambda kappa, prev: warm_grid(Calibration(problem, kappa), prev, criterion, force_start),
                       failed=lambda g: g.failed)
    cost, log, accepted, last = 0, [], None, None
    for kappa, previous, current in ladder:
        cost += current.passes * kappa
        entry = current.log_entry()
        if not current.failed:
            entry["cons"] = current.consistency
            if previous is not None:
                entry["du_sd"] = moved = _force_moved(previous, current)
                last = current
                if moved < TAU_U and current.consistency <= TAU_R:
                    accepted = current
                else:
                    entry["stop"] = "moved" if moved >= TAU_U else "inconsistent"
        log.append(entry)
        if accepted is not None:
            break
    return Calibrated(problem=problem, criterion=criterion, accepted=accepted is not None,
                      reported=accepted or last, cost=cost, log=log)
