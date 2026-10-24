"""Latent force inference with probabilistic ODE solvers, calibrated as in "Stable and Calibrated Latent Force ODE Filters".

    problem = lfcal.LatentForceProblem(vector_field=f, x0=x0, times=t, data=y, noise_std=sd,
                                       force_scale=1.0, length_scale=2.0)
    fit = lfcal.calibrate(problem)        # the diffusion selected by I(U; Y | Z), the paper's recommendation
    post = fit.posterior()                # force and state, mean and sd, on the reported grid

  problem       the latent force problem (ODE, initial value, data, the force's scales) and its grids
  algorithm     the algorithm: the fit, the grid loop, the stability guard, the ODE-consistency check
  calibration   one problem on one grid: the jitted pieces the algorithm is built from
  force         the Matern force prior and its sample paths
  moments       covariances and marginal variances of square-root factors
  paper         the paper's benchmark, its baselines and sweeps, figures and result files

Every computation here assumes double precision on the CPU; importing the package switches JAX to it.
"""

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

from lfcal.algorithm import CRITERIA, Calibrated, Posterior, calibrate
from lfcal.force import MaternForce
from lfcal.problem import LatentForceProblem

__all__ = ["CRITERIA", "Calibrated", "LatentForceProblem", "MaternForce", "Posterior", "calibrate"]
