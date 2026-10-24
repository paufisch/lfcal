"""The paper's benchmark and everything the experiments share, on top of the method in `lfcal`.

  systems       the benchmark systems: latent force problems with a known truth, and their search windows
  benchmark     the comparison: designs, draws, methods, the algorithm's records, and the scoring of a dataset
  scores        a posterior's scores against the truth, and the chi^2 interval of a calibrated one
  classical     the classical state-space LFM with an RK4 discretisation
  iterated      the IEKS (residual re-linearised at the smoothed mean) and the sigma_x continuation
  information   I(U; Z), I(U; Y | Z) and the directed information on the IEKS-converged linearisation
  plotting      figure style, palette, labels and shared drawing helpers
  io            result files: CSV tables and .npz arrays
"""
