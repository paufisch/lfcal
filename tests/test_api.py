"""The public API: `lfcal.LatentForceProblem` and `lfcal.calibrate` on a problem built by hand.

A problem built from the ODE, the data and the search box of a benchmark system gives the paper's fit: on
Duffing (second order) the same numbers as the benchmark system itself, on Lorenz-96 (first order) the
recorded result of the comparison with the default criterion. Smaller tests build problems of both orders
with every observed derivative, and check that invalid input is refused.
"""

import jax.numpy as jnp
import numpy as onp
import pytest

import lfcal
from lfcal.algorithm import start_point
from lfcal.calibration import Calibration

FIT_RTOL = 1e-3


def _duffing_accel(K=64):
    """The benchmark system duffing_accel at draw 0 with `K` observations, and the same problem by hand."""
    from lfcal.paper import benchmark, systems

    system = systems.build("duffing_accel", score_at="midpoints", n_obs=K, **benchmark.system_kw("duffing_accel", 0, K))
    t, y = system.observations()
    box = systems.window(system, systems.SYSTEMS["duffing_accel"][1])
    problem = lfcal.LatentForceProblem(vector_field=systems.duffing_field, order=2, x0=[0.0], times=t, data=y,
                                       obs_derivative=2, noise_std=system.noise_stds, force_scale=systems.DUFFING_STD,
                                       length_scale=systems.DUFFING_ELL, diffusion_box=(box[0], box[-1]))
    return system, problem


def test_problem_by_hand_is_the_paper_path():
    system, problem = _duffing_accel()
    assert problem.bounds == system.bounds
    mine, paper = lfcal.calibrate(problem, "obs"), lfcal.calibrate(system, "obs")
    assert (mine.kappa, mine.accepted, len(mine.log)) == (paper.kappa, paper.accepted, len(paper.log))
    onp.testing.assert_allclose(mine.phi, paper.phi, rtol=1e-12, atol=0)
    onp.testing.assert_allclose(onp.asarray(mine.traj), onp.asarray(paper.traj), rtol=1e-12, atol=1e-14)
    onp.testing.assert_allclose(mine.consistency, paper.consistency, rtol=1e-12)

    post = mine.posterior()
    N = mine.kappa * problem.n_obs
    assert post.ts.shape == (N + 1,) and post.state_mean.shape == (N + 1, 1) and post.force_mean.shape == (N + 1,)
    assert onp.isclose(mine.force_std, problem.force_scale * onp.exp(0.5 * mine.phi[1]))


def _lorenz96_da(draw=0):
    """The benchmark system lorenz96_da at `draw` with K0 observations, and the same first-order problem by
    hand: eight components, every other one observed, the force on the first."""
    from lfcal.paper import benchmark, systems

    K0 = benchmark.DESIGNS["lorenz96_da"].K0
    system = systems.build("lorenz96_da", score_at="midpoints", n_obs=K0,
                           **benchmark.system_kw("lorenz96_da", draw, K0))
    t, y = system.observations()
    box = systems.window(system, systems.SYSTEMS["lorenz96_da"][1])
    problem = lfcal.LatentForceProblem(vector_field=systems.lorenz96_field, x0=systems.L96_X0, times=t, data=y,
                                       obs_indices=systems.L96_OBS_SITES, noise_std=system.noise_stds,
                                       force_scale=systems.L96_STD, length_scale=systems.L96_ELL,
                                       diffusion_box=(box[0], box[-1]))
    return system, problem


def test_first_order_problem_by_hand():
    """Lorenz-96 (first order, eight components, half of them observed) by hand, end to end with the default
    criterion I(U; Y | Z): the recorded fit of the comparison."""
    import objectives

    system, problem = _lorenz96_da()
    assert problem.bounds == system.bounds and problem.p0_jitter == system.p0_jitter
    fit = lfcal.calibrate(problem)
    assert fit.criterion == "mi" and not fit.failed
    assert fit.posterior().state_mean.shape == (fit.kappa * problem.n_obs + 1, 8)

    rec = objectives.load("lorenz96_da", 0, "mi")
    if rec is None:
        pytest.skip("no recorded comparison entry for mi on lorenz96_da, draw 0")
    assert (fit.kappa, fit.accepted) == (rec["kappa"], rec["identified"])
    mine = {"cons": fit.consistency, "log_sigma_x2": fit.phi[0], "log_xi_u": fit.phi[1], "log_ell_u": fit.phi[2]}
    for key, value in mine.items():
        assert onp.isclose(value, rec[key], rtol=FIT_RTOL, atol=1e-9), (key, value, rec[key])


def _oscillators(order, obs_derivative):
    """Two coupled oscillators driven on the first, as a first- or second-order ODE in 2 or 4 components."""
    times = onp.linspace(0.0, 10.0, 21)[1:]
    if order == 2:
        def f(x, dx, u, *, t):
            return (-x - 0.2 * dx + 0.1 * x[::-1]).at[0].add(u)

        x0 = [1.0, 0.0]
    else:
        def f(x, u, *, t):
            return jnp.array([x[2], x[3], -x[0] + 0.1 * x[1] + u, -x[1] + 0.1 * x[0]])

        x0 = [1.0, 0.0, 0.0, 0.0]
    return lfcal.LatentForceProblem(vector_field=f, order=order, x0=x0, times=times,
                                    data=onp.full((len(times), 2), 0.1), obs_indices=(0, 1),
                                    obs_derivative=obs_derivative, noise_std=0.1, force_scale=1.0, length_scale=2.0)


@pytest.mark.parametrize(("order", "obs_derivative"), [(1, 0), (1, 1), (2, 0), (2, 1), (2, 2)])
def test_problems_of_both_orders(order, obs_derivative):
    problem = _oscillators(order, obs_derivative)
    cal = Calibration(problem, 2)
    phi = start_point(cal)
    m_pred = cal.ek1_predicted_means(jnp.asarray(phi))
    traj, _ = cal.relinearize(jnp.asarray(phi), m_pred)
    post = cal.posterior(jnp.asarray(phi), traj)
    assert onp.all(onp.isfinite(onp.asarray(post["m_s"]))) and onp.all(onp.isfinite(onp.asarray(post["var"])))
    assert onp.isfinite(cal.consistency(phi, traj))
    y = problem.observe(problem.y0, jnp.asarray(0.5), 0.0)
    assert y.shape == (2,)


def test_invalid_input_is_refused():
    def f(x, u, *, t):
        return -x + u

    good = {"vector_field": f, "x0": [0.0], "times": [1.0, 2.0, 3.0], "noise_std": 0.1, "force_scale": 1.0,
            "length_scale": 1.0}
    lfcal.LatentForceProblem(**good)
    for bad in ({"times": [1.0, 2.0, 3.5]}, {"times": [0.0, 1.0, 2.0]}, {"data": onp.zeros((2, 1))},
                {"obs_derivative": 2}, {"v0": [0.0]}, {"order": 3}, {"obs_indices": (1,)}):
        with pytest.raises(ValueError):
            lfcal.LatentForceProblem(**{**good, **bad})
    problem = lfcal.LatentForceProblem(**good, data=onp.zeros(3))
    for kw in ({"kappas": (1, 2)}, {"kappas": (2, 6)}, {"kappas": (2,)}, {"criterion": "nope"}):
        with pytest.raises(ValueError):
            lfcal.calibrate(problem, **kw)


def test_information_criteria_gradient_is_finite():
    """The gradients of I(U; Y | Z) and D in log sigma_x^2 are finite across the box (Lorenz-96 draw 9,
    kappa = 4, first linearisation). ode-filters 0.8.0 and 0.8.1 return NaN at most of these points."""
    from lfcal.algorithm import guard_ek1
    from lfcal.paper import benchmark, systems

    K = benchmark.DESIGNS["lorenz96_da"].K0
    cal = Calibration(systems.build("lorenz96_da", score_at="midpoints", n_obs=K,
                                    **benchmark.system_kw("lorenz96_da", 9, K)), 4)
    phi0 = start_point(cal)
    _, traj = guard_ek1(cal, phi0)
    surrogate = cal.linearize(jnp.asarray(traj))
    for key in ("mi", "di"):
        for s in onp.r_[onp.linspace(*cal.bounds[0], 9), phi0[0]]:
            value, grad = cal.information_value_and_grad[key](jnp.asarray(s), jnp.asarray(phi0), surrogate)
            assert onp.isfinite(float(value)) and onp.isfinite(float(grad)), (key, float(s))
