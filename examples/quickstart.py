"""Quickstart: infer the torque that drives a damped pendulum from noisy readings of its angle.

The pendulum `theta'' = -sin(theta) - 0.2 theta' + u(t)` is driven by a torque `u` we pretend not to know. We
simulate it with a known torque, observe the angle at 100 times with noise, and let `lfcal.calibrate` recover
the torque and the angle, with their uncertainty, from the data alone.

    uv run python examples/quickstart.py          # prints the fit, writes figures/quickstart.pdf
"""

from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as onp
from tueplots import bundles

import lfcal


def pendulum(x, dx, u, *, t):
    """The ODE as lfcal takes it: `theta'' = f(theta, theta', u, t)`, with `x = (theta,)`."""
    return -jnp.sin(x) - 0.2 * dx + u


# ------------------------------------------------------------------ data: a torque we pretend not to know
def true_torque(t):
    return 0.5 * onp.sin(0.3 * t) + 0.3 * onp.sin(0.9 * t + 1.0)


def simulate(times, theta0=0.5, steps=50):
    """The angle at `times` under the true torque, by RK4 with `steps` steps per observation interval."""
    def rhs(y, t):
        return onp.array([y[1], -onp.sin(y[0]) - 0.2 * y[1] + true_torque(t)])

    dt, y, t, angles = (times[1] - times[0]) / steps, onp.array([theta0, 0.0]), 0.0, []
    for _ in times:
        for _ in range(steps):
            k1 = rhs(y, t)
            k2 = rhs(y + 0.5 * dt * k1, t + 0.5 * dt)
            k3 = rhs(y + 0.5 * dt * k2, t + 0.5 * dt)
            k4 = rhs(y + dt * k3, t + dt)
            y, t = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4), t + dt
        angles.append(y[0])
    return onp.array(angles)


times = onp.linspace(0.0, 40.0, 101)[1:]           # K = 100 equidistant times after t0 = 0
data = simulate(times) + 0.05 * onp.random.default_rng(0).normal(size=len(times))

# ------------------------------------------------------------------ the latent force problem
problem = lfcal.LatentForceProblem(
    vector_field=pendulum, order=2, x0=[0.5], v0=[0.0],
    times=times, data=data, noise_std=0.05,          # the angle, observed with noise std 0.05
    force_scale=0.5, length_scale=3.0,               # rough guesses of the torque's size and time scale
)
fit = lfcal.calibrate(problem)                       # the diffusion selected by I(U; Y | Z)
print(f"grid density kappa = {fit.kappa}, accepted: {fit.accepted}, consistency r = {fit.consistency:.2f}")
print(f"torque std {fit.force_std:.2f}, length scale {fit.length_scale:.2f}, state diffusion {fit.sigma_x2:.3g}")

# ------------------------------------------------------------------ the posterior against the truth
post = fit.posterior()
plt.rcParams.update(bundles.tue_ai_beamer())
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
fig, (ax_x, ax_u) = plt.subplots(2, 1, sharex=True)
for ax, mean, sd in ((ax_x, post.state_mean[:, 0], post.state_sd[:, 0]), (ax_u, post.force_mean, post.force_sd)):
    ax.fill_between(post.ts, mean - 2 * sd, mean + 2 * sd, color=colors[0], alpha=0.3, lw=0)
    ax.plot(post.ts, mean, color=colors[0], label="posterior mean, $\\pm 2$ sd")
ax_x.plot(times, data, ".", ms=2, color=colors[1], label="data")
ax_u.plot(post.ts, true_torque(post.ts), "--", color=colors[5], label="truth")     # a colour kept for the truth
ax_x.set_ylabel("angle")
ax_u.set_ylabel("torque")
ax_u.set_xlabel("time")
handles = {label: h for ax in (ax_x, ax_u) for h, label in zip(*ax.get_legend_handles_labels(), strict=True)}
fig.legend(handles.values(), handles.keys(), loc="outside lower center", ncol=len(handles))
out = Path("figures") / "quickstart.pdf"
out.parent.mkdir(exist_ok=True)
fig.savefig(out)
print(f"wrote {out}")
