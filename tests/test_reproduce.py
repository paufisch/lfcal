"""The code reproduces the recorded results: one small cell of each pipeline, recomputed and compared
with `results/`. Nothing is written to `results/`. Where a result is checked, it goes through the CSV/NPZ
writer first, so the files are checked too.

Floats are compared to a relative tolerance (`RTOL`; `ATOL` for quantities that are zero up to rounding,
such as the last IEKS step): their last bits depend on the CPU, the BLAS and XLA's compilation. A time
series is compared relative to its own scale. Everything discrete (flags, picks, counts, labels) must
agree exactly. The numbers of a record that is not `ok` (a diverged filter, not plotted) are not
reproducible, so only its fields are compared.

The fits (the algorithm, the classical LFM) are compared to `FIT_RTOL`, and the record of their path
(`PATH_FIELDS`: how many passes it took, how far its last cycle moved) is not compared. Records written before
`ode-filters` 0.8 measure the force's length scale as a squared time and are converted on reading (see
`lfcal.paper.io`); L-BFGS-B still runs in the coordinates of 0.7 (`lfcal.calibration.LBFGS_SCALE`). The
Lorenz-96 datasets are the recorded ones (`test_lorenz96_recorded_data`).

A test whose recorded results are missing is skipped.
"""

import dataclasses
import numbers

import numpy as onp
import pytest

RTOL, ATOL = 1e-5, 1e-9
FIT_RTOL = 1e-3
#: a fit's record of its own path rather than its result
PATH_FIELDS = ("cost", "cost_cum", "passes", "step", "du_sd")


def _is_float(v):
    return isinstance(v, numbers.Real) and not isinstance(v, (bool, int))


def assert_close(a, b, where="result", numbers_too=True, rtol=RTOL):
    """`a` (recomputed) agrees with `b` (recorded): same structure, discrete leaves equal, floats
    to `rtol`/`ATOL` (NaN equal to NaN) unless `numbers_too` is off."""
    if isinstance(b, dict):
        assert a.keys() == b.keys(), f"{where}: fields {sorted(a.keys() ^ b.keys())} differ"
        numbers_too = numbers_too and b.get("ok", True) is not False
        for k in b:
            assert_close(a[k], b[k], f"{where}.{k}", numbers_too, rtol)
    elif isinstance(b, list) and len(b) > 1 and all(_is_float(v) for v in b):
        assert len(a) == len(b), f"{where}: length {len(a)} != {len(b)}"
        b = onp.asarray(b)
        scale = onp.nanmax(onp.abs(b)) if onp.isfinite(b).any() else 0.0
        assert not numbers_too or onp.allclose(a, b, rtol=rtol, atol=ATOL + rtol * scale, equal_nan=True), where
    elif isinstance(b, list):
        assert len(a) == len(b), f"{where}: length {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            assert_close(x, y, f"{where}[{i}]", numbers_too, rtol)
    elif _is_float(b) and _is_float(a):
        assert not numbers_too or onp.isclose(a, b, rtol=rtol, atol=ATOL, equal_nan=True), f"{where}: {a!r} != {b!r}"
    else:
        assert a == b, f"{where}: {a!r} != {b!r}"


def recorded(load, *args):
    """The recorded result `load(*args)`, or skip the test when `results/` does not hold it."""
    try:
        out = load(*args)
    except FileNotFoundError:
        out = None
    if not out:
        pytest.skip(f"no recorded results for {load.__module__}.{load.__name__}{args}")
    return out


def test_sigma_x_sweep_duffing_kappa_2(tmp_path):
    import sigma_x_sweep

    rec = recorded(sigma_x_sweep.load, "duffing")
    sigma_x_sweep.save("duffing", sigma_x_sweep.compute("duffing", kappas=(2,)), tmp_path)
    new = sigma_x_sweep.load("duffing", tmp_path)
    assert_close(new["sigmas"], rec["sigmas"], "sigmas")
    assert_close(new["sweep"]["2"], rec["sweep"]["2"], "sweep")
    assert_close(new["picks"]["2"], rec["picks"]["2"], "picks")


def test_failure_directions(tmp_path):
    import failure_directions
    import force_information

    recorded(force_information.load, "duffing")                 # the sigma_x^2 of the middle column
    rec = recorded(failure_directions.load)
    failure_directions.save(failure_directions.compute(), tmp_path)
    assert_close(failure_directions.load(tmp_path), rec)


def test_force_information_duffing_kappa_2(tmp_path):
    import force_information

    rec = recorded(force_information.load, "duffing")
    force_information.save("duffing", {"2": force_information.compute("duffing", 2)}, tmp_path)
    assert_close(force_information.load("duffing", tmp_path)["2"], rec["2"])


# ---------------------------------------------------------------- the comparison and its offshoots
# Whole entries of the comparison (the data of Fig. 4) and of the robustness study, recomputed
# through the same `benchmark.run_method`, record and scoring as every method's: the algorithm with two
# criteria, and the classical LFM with and without its process noise.
def result(entry):
    """An entry without its wall-clock time, which no run reproduces, and its fit's path (`PATH_FIELDS`)."""
    return {k: v for k, v in entry.items() if k not in ("wall_s", *PATH_FIELDS)}


@pytest.mark.parametrize(("name", "method"), [("duffing_accel", "obs"), ("duffing_accel", "mi"),
                                              ("duffing_accel", "classical_rk4"), ("lorenz96_da", "classical_rk4")])
def test_objectives_entry(tmp_path, name, method):
    import objectives

    from lfcal.paper import benchmark

    draw = 0
    rec = recorded(objectives.load, name, draw, method)
    K0 = benchmark.DESIGNS[name].K0
    entry = benchmark.run_method(name, K0, method, **benchmark.system_kw(name, draw, K0))
    objectives.save(name, draw, method, entry, tmp_path)
    assert_close(result(objectives.load(name, draw, method, tmp_path)), result(rec), rtol=FIT_RTOL)


def test_robustness_classical_entry(tmp_path):
    import robustness

    from lfcal.paper import benchmark

    name, kind, draw = "sir", "matern", 0
    rec = recorded(robustness.load, name, kind, draw, benchmark.CLASSICAL)
    entry = benchmark.run_method(name, robustness.k0(name), benchmark.CLASSICAL,
                                 **robustness.system_kw(name, kind, draw))
    robustness.save(name, kind, draw, benchmark.CLASSICAL, entry, tmp_path)
    assert_close(result(robustness.load(name, kind, draw, benchmark.CLASSICAL, tmp_path)), result(rec),
                 rtol=FIT_RTOL)


def test_posterior_marginals(tmp_path):
    """The stored posterior of a reported fit, the input of the failure figures."""
    import failures

    from lfcal.paper import benchmark
    from lfcal.paper.io import write_npz

    name, draw = "duffing_accel", 0
    if not failures.posterior_path(name, benchmark.CLASSICAL, draw).exists():
        pytest.skip("no recorded posterior")
    K0 = benchmark.DESIGNS[name].K0
    write_npz(tmp_path / "posterior.npz",
              benchmark.posterior_marginals(name, K0, benchmark.CLASSICAL, **benchmark.system_kw(name, draw, K0)))
    new = benchmark.read_posterior(tmp_path / "posterior.npz")
    rec = benchmark.read_posterior(failures.posterior_path(name, benchmark.CLASSICAL, draw))
    assert_close({k: onp.asarray(v).tolist() for k, v in new.items()},
                 {k: onp.asarray(v).tolist() for k, v in rec.items()})


#: the last observation (t = 10) of the recorded Lorenz-96 datasets: the comparison's draws 0 and 9, the
#: sinusoid's draw 3 of the robustness study, and the K = 200 system of the sweeps
L96_LAST = {
    ("lorenz96_da", 0, "matern"): (-0.6873155233556968, 8.247378092039673, -4.171401597655513, 0.04576482261478221),
    ("lorenz96_da", 9, "matern"): (7.718498059650104, -0.0944600437397688, 1.5426240039240013, 7.704607885821766),
    ("lorenz96_da", 3, "sine"): (1.7900369102675424, 2.9744630536325056, 3.9957842002332065, 5.949429292221585),
    ("lorenz96", 0, "matern"): (-0.9220158247435719, 7.31118474687294, -4.848887358896876, 1.1068834231027926),
}


def _lorenz96(name, draw, kind):
    from lfcal.paper import benchmark, systems

    K = 200 if name == "lorenz96" else benchmark.DESIGNS[name].K0
    kw = {} if name == "lorenz96" else benchmark.system_kw(name, draw, K)
    return systems.build(name, n_obs=K, force_kind=kind, **kw)


@pytest.mark.parametrize(("name", "draw", "kind"), list(L96_LAST))
def test_lorenz96_recorded_data(name, draw, kind):
    """The Lorenz-96 data at t = 10 are the recorded ones. They come from stored states (`systems.L96_TRUTH`),
    so this holds on every platform."""
    _, data = _lorenz96(name, draw, kind).observations()
    assert onp.allclose(data[-1], L96_LAST[(name, draw, kind)], rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize(("name", "draw", "kind"), list(L96_LAST))
def test_lorenz96_recorded_truth_is_the_trajectory_of_its_force(name, draw, kind):
    """The stored states belong to their dataset's force and initial state: up to t = 1, before chaos
    amplifies the solvers' difference, the truth computed without them agrees to 1e-6."""
    system = _lorenz96(name, draw, kind)
    t = system.obs_times[system.obs_times <= 1.0]
    free = dataclasses.replace(system, truth_at_obs=None)
    assert system.truth_at_obs is not None
    assert onp.array_equal(system.solve(t), system.truth_at_obs[: len(t)])
    assert onp.abs(free.solve(t) - system.solve(t)).max() < 1e-6
