# lfcal: calibrated latent force inference with probabilistic ODE solvers

Code for the paper [*Stable and Calibrated Latent Force ODE Filters*](https://arxiv.org/abs/2610.33939) by Paul Fischer, Jonathan Schmidt and Philipp Hennig.

`lfcal` estimates an unknown force that drives an ODE, together with the state, from noisy observations. It is an ODE filter with a Gauss–Markov prior on the force, whose state diffusion is chosen so that the force, not the solver's numerical slack, explains the data. The paper's benchmark is in `lfcal.paper`, and `experiments/` has one script per study.

## Quickstart

```python
import jax.numpy as jnp
import lfcal

def pendulum(x, dx, u, *, t):                   # theta'' = f(theta, theta', u, t), u the unknown torque
    return -jnp.sin(x) - 0.2 * dx + u

problem = lfcal.LatentForceProblem(
    vector_field=pendulum, order=2, x0=[0.5], v0=[0.0],
    times=times, data=angles, noise_std=0.05,   # the angle at K equidistant times after t0 = 0
    force_scale=0.5, length_scale=3.0,          # rough guesses of the torque's size and time scale
)
fit = lfcal.calibrate(problem)                  # the diffusion selected by I(U; Y | Z)
post = fit.posterior()                          # post.ts, post.force_mean, post.force_sd, post.state_mean, ...
```

`uv run python examples/quickstart.py` runs this on simulated data and plots the recovered torque.

`calibrate` refines the grid from 2 to 32 filter steps per observation interval and accepts a grid once the force estimate stops changing and the fit reproduces the data through the ODE. The criterion picks the state diffusion `sigma_x^2`; the force hyperparameters always come from the observation evidence.

| `criterion` | `sigma_x^2` maximizes |
|---|---|
| `"mi"` (default) | the mutual information `I(U; Y \| Z)` of force and data given the ODE; recovers the force on every problem of the paper |
| `"di"` | the directed information of the data about the force |
| `"fenrir_ieks"` | the evidence `log p(Y \| Z)`, by Fenrir's estimator at the IEKS linearisation |
| `"obs"` | the observation evidence, jointly with the force hyperparameters |

The fit says whether a grid was `accepted`; if not, the force is not identified. `help(lfcal.LatentForceProblem)` and `help(lfcal.Calibrated)` describe the arguments and the fit.

The assumptions are the paper's: one scalar force, a known initial value, equidistant data with Gaussian noise of known variance. The acceptance test integrates the ODE with explicit RK4, so on stiff problems no fit is ever accepted.

## Setup

Set up the environment with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

This creates `.venv` with Python 3.14 and the dependencies pinned in `uv.lock`, among them JAX and [`ode-filters`](https://github.com/paufisch/ode_filters), which does the ODE filtering. Importing `lfcal` switches JAX to double precision on the CPU. Drawing the figures needs LaTeX with `pdflatex`.

## Reproducing the paper

`results/` holds the paper's results; [`results/README.md`](results/README.md) describes the files.

```bash
make figures   # the paper's figures from results/, into figures/
make results   # recomputes results/: hours, JOBS (default 13) jobs at a time
make test      # recomputes a small part of each study and compares it with results/
```

`make results` skips recorded results, so move `results/` aside to recompute them all. The `Makefile` has a target per study, and each script in `experiments/` lists its options with `--help`.

## License

MIT, see `LICENSE`.

## Citation

```bibtex
@article{fischer2026stable,
  title   = {Stable and Calibrated Latent Force {ODE} Filters},
  author  = {Fischer, Paul and Schmidt, Jonathan and Hennig, Philipp},
  journal = {arXiv preprint arXiv:2610.33939},
  year    = {2026},
}
```
