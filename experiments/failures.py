"""The failure figures: the posterior of each wrong fit, against the algorithm with I(U;Y|Z) and the classical LFM.

A fit is wrong when its force error exceeds `benchmark.WRONG` times the best method's on its dataset. The
figures, by their name in `FIGURES`:

  failures_by_method    per method with several wrong fits on the 60 datasets of `objectives`, those fits, one
                        column each (`failures_by_method_{method}.pdf`), and the methods with a single wrong fit
                        together (`failures_by_method_singles.pdf`)
  robustness_examples   per condition of `robustness`, a dataset on which every method works

The figures draw recorded posteriors only. `--compute` fits every posterior they need that is not recorded.
Unlike the other scripts it takes no systems: its jobs follow from the recorded results of `objectives` and
`robustness`.

    uv run python experiments/failures.py --compute --jobs 8   # results/{posteriors,failures_robustness}/
    uv run python experiments/failures.py --figure             # both figures
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import objectives
import robustness

from lfcal.paper import benchmark, plotting
from lfcal.paper.benchmark import CLASSICAL, COMPARED, WRONG
from lfcal.paper.io import write_npz
from lfcal.paper.plotting import METHOD_LEGEND, METHOD_TICKS, POSTERIOR_BANDS, POSTERIOR_COLOURS

RESULTS = Path(__file__).resolve().parents[1] / "results"
#: the posteriors of the wrong fits of `objectives` and of their references, and those of `robustness_examples`
POSTERIORS, ROBUSTNESS = RESULTS / "posteriors", RESULTS / "failures_robustness"
#: drawn with every failure: the algorithm with I(U;Y|Z) (with its band) and the classical LFM
REFERENCE = ("mi", CLASSICAL)
#: drawn on every example of `robustness_examples`, in this order
EXAMPLE_METHODS = ("mi", "obs", CLASSICAL)


def posterior_path(name, method, draw):
    return POSTERIORS / f"{name}_s{draw}_{method}.npz"


def robustness_path(name, kind, draw, method):
    return ROBUSTNESS / f"{name}_{kind}_s{draw}_{method}.npz"


def store(out, posterior):
    """Write a posterior (`benchmark.posterior_marginals`) to `out` and print its summary line."""
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    write_npz(out, posterior)
    print(f"{Path(out).stem}: kappa {posterior['kappa']} rel_u {posterior['rel_u']:.3f}", flush=True)


# -------------------------------------------------------------------- the wrong fits
def wrong_fits(data=None):
    """`[(system, draw, method, error ratio)]` of every wrong fit of `objectives`."""
    data = data or objectives.collect()
    return [(name, r["draw"], m, r[m]["ratio"]) for name, recs in data.items() for r in recs for m in COMPARED
            if r[m]["ratio"] > WRONG]


def _robustness_datasets():
    """`(name, kind, draw, {method: error ratio})` of every dataset of `robustness` with every method recorded."""
    for name, kind, draw in robustness.datasets():
        entries = {m: robustness.load(name, kind, draw, m) for m in COMPARED}
        if all(e is not None for e in entries.values()):
            yield name, kind, draw, benchmark.error_ratios(entries)[0]


def robustness_examples():
    """`[(condition, system, kind, draw)]`: per condition of `robustness`, the dataset on which every method is
    within `WRONG` times the best and I's force error is closest to its median over such datasets. Of an even
    number of datasets the two middle ones are equally close; of those, the one on which the methods agree best
    (the smaller largest error ratio)."""
    candidates = {}
    for name, kind, draw, ratios in _robustness_datasets():
        if all(r <= WRONG for r in ratios.values()):
            error_mi = benchmark.force_error(robustness.load(name, kind, draw, "mi"))
            candidates.setdefault(robustness.condition(name, kind), []).append(
                (name, kind, draw, error_mi, max(ratios.values())))
    out = []
    for cond in robustness.CONDITIONS:
        if cases := sorted(candidates.get(cond, []), key=lambda c: c[3]):
            middle = cases[(len(cases) - 1) // 2: len(cases) // 2 + 1]
            name, kind, draw, *_ = min(middle, key=lambda c: c[4])
            out.append((cond, name, kind, draw))
    return out


# -------------------------------------------------------------------- computing the posteriors
def _benchmark_job(out, name, draw, method):
    K0 = benchmark.DESIGNS[name].K0
    return out, name, K0, method, benchmark.system_kw(name, draw, K0)


def _robustness_job(name, kind, draw, method):
    return (robustness_path(name, kind, draw, method), name, robustness.k0(name), method,
            robustness.system_kw(name, kind, draw))


def missing_posteriors():
    """Every posterior the figures need that is not recorded, as `(path, system, K, method, factory arguments)`,
    the slow FitzHugh--Nagumo fits first."""
    fails = wrong_fits()
    pairs = [(n, d, m) for n, d, failing, _ in fails for m in (failing, *REFERENCE)]
    jobs = [_benchmark_job(posterior_path(n, m, d), n, d, m) for n, d, m in pairs]        # failures_by_method
    jobs += [_robustness_job(n, kind, draw, m) for _, n, kind, draw in robustness_examples() for m in EXAMPLE_METHODS]
    missing = {job[0]: job for job in jobs if not job[0].exists()}
    return sorted(missing.values(), key=lambda job: (job[1] != "fhn_rough", str(job[0])))


def _fit_posterior(job):
    out, name, K, method, system_kw = job
    store(out, benchmark.posterior_marginals(name, K, method, **system_kw))


def run_parallel(fn, jobs, processes=1):
    """`fn` on every job, in `processes` worker processes (spawned, not forked: JAX's threads do not survive a
    fork)."""
    if processes > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(processes, mp_context=get_context("spawn")) as pool:
            list(pool.map(fn, jobs))
    else:
        for job in jobs:
            fn(job)


def compute(processes=1):
    """Fit every posterior the figures need that is not recorded."""
    jobs = missing_posteriors()
    print(f"failures: {len(jobs)} posteriors to fit", flush=True)
    run_parallel(_fit_posterior, jobs, processes)


# -------------------------------------------------------------------- figures
def _close_columns(fig):
    """Pull the columns of a many-column posterior figure together: they share their axes, and the bundle's
    padding would leave each narrow column too little width."""
    fig.get_layout_engine().set(w_pad=0.01, wspace=0.0)


def _posterior_columns_axes(axes):
    """Labels and ticks of a two-row grid of posterior columns (force on top, scored state below)."""
    axes[0, 0].set_ylabel("force $u$")
    axes[1, 0].set_ylabel("state $g$")                       # the scored state component (Sec. Metrics)
    for ax in axes[-1]:
        ax.set_xlabel("$t/T$")
        ax.set_xticks([0.0, 0.5, 1.0], ["0", "0.5", "1"])
    fig = axes[0, 0].figure
    fig.align_ylabels(axes[:, 0])


def failure_grid(rows, out):
    """Wrong fits `rows` (`[(system, draw, method, ratio)]`; the method may differ between them), one column
    each: force (top) and scored state with its data (bottom), in the style of Fig. `failure-directions`. I's
    band is filled blue, the failing method's outlined purple whatever the method, with its mean on top, the
    classical LFM's dotted navy, and the truth dashed (`plotting.POSTERIOR_COLOURS`). A single column is one text
    column wide."""
    blue, purple, navy = (POSTERIOR_COLOURS[m] for m in ("mi", "obs", CLASSICAL))
    methods = {m for _, _, m, _ in rows}
    mixed, single = len(methods) > 1, len(rows) == 1
    fig, axes = plotting.subplots(2, len(rows), squeeze=False, sharex="col", column="half" if single else "full",
                                  height_to_width_ratio=0.4 if single else 0.75 if len(rows) > 2 else 0.45)
    for c, (name, draw, method, ratio) in enumerate(rows):
        fits = {m: benchmark.read_posterior(posterior_path(name, m, draw)) for m in {method, *REFERENCE}}
        ref, failing, lfm = fits["mi"], fits[method], fits[CLASSICAL]
        for r, ch in enumerate(("u", "x")):
            ax = axes[r, c]
            mean, sd = f"{ch}_mean", f"{ch}_sd"
            plotting.truth_and_data(ax, ref, ch, name)
            if method != "mi":
                plotting.posterior_band(ax, ref["ts"], ref[mean], ref[sd], blue, "filled")
                ax.plot(ref["ts"], ref[mean], color=blue, lw=plotting.MEAN_LW, zorder=3)
            plotting.posterior_band(ax, failing["ts"], failing[mean], failing[sd], purple, "outlined", zorder=4)
            ax.plot(failing["ts"], failing[mean], color=purple, lw=1.0, zorder=5)
            if method != CLASSICAL:
                ax.plot(lfm["ts"], lfm[mean], color=navy, lw=0.7, zorder=3)
                plotting.posterior_band(ax, lfm["ts"], lfm[mean], lfm[sd], navy, "dotted", zorder=3)
            plotting.posterior_limits(ax, ref, ch)
        which = f": {METHOD_TICKS[method]}" if mixed else ""
        axes[0, c].set_title(f"{plotting.SYSTEM_SHORT[name]}, draw {draw + 1}{which}\nerror ratio {ratio:.2f}")
    _posterior_columns_axes(axes)
    if not single:
        _close_columns(fig)
    handles = [plotting.posterior_handle(purple, "outlined")]
    labels = ["failing method" if mixed else METHOD_LEGEND[next(iter(methods))]]
    if "mi" not in methods:
        handles.insert(0, plotting.posterior_handle(blue, "filled"))
        labels.insert(0, METHOD_LEGEND["mi"])
    if methods != {CLASSICAL}:
        handles.append(plotting.posterior_handle(navy, "dotted"))
        labels.append(METHOD_LEGEND[CLASSICAL])
    handles += [plotting.truth_handle(), plotting.data_handle(alpha=0.6)]
    labels += ["truth", "data"]
    plotting.shared_legend(fig, handles=handles, labels=labels, ncol=2 if single else len(labels))
    print(f"wrote {plotting.save(fig, out)}  ({len(rows)} wrong fits)")


def _fits_recorded(rows):
    return all(posterior_path(name, m, draw).exists() for name, draw, method, _ in rows for m in {method, *REFERENCE})


def figures_by_method():
    """Per method with several wrong fits on the 60 datasets, those fits (`failure_grid`,
    `figures/failures_by_method_{method}.pdf`), and the single wrong fits of the methods that have one, in one
    figure (`failures_by_method_singles.pdf`)."""
    fails = wrong_fits()
    by_method = {m: [f for f in fails if f[2] == m] for m in COMPARED}
    singles = [rows[0] for rows in by_method.values() if len(rows) == 1]
    for name, rows in [*((f"failures_by_method_{m}", rows) for m, rows in by_method.items() if len(rows) != 1),
                       ("failures_by_method_singles", singles)]:
        if not rows:
            print(f"failures: no wrong fits for {name}; figure skipped")
        elif not _fits_recorded(rows):
            print(f"failures: posteriors missing for {name} (`failures.py --compute`); figure skipped")
        else:
            failure_grid(rows, name)


def figure_robustness_examples():
    """One column per condition of `robustness`, on the dataset of `robustness_examples`: force (top) and scored
    state with its data (bottom), in the style of Fig. `failure-directions`. I's band is filled blue, ell_OBS's
    outlined purple, the classical LFM's dotted navy (`plotting.POSTERIOR_COLOURS`), I's mean on top, and the
    truth dashed (`figures/robustness_examples.pdf`)."""
    examples = robustness_examples()
    if not all(robustness_path(n, k, d, m).exists() for _, n, k, d in examples for m in EXAMPLE_METHODS):
        print("failures: example posteriors missing (`failures.py --compute`); robustness_examples skipped")
        return
    fig, axes = plotting.subplots(2, len(examples), squeeze=False, sharex="col", height_to_width_ratio=0.75)
    for c, (cond, name, kind, draw) in enumerate(examples):
        fits = {m: benchmark.read_posterior(robustness_path(name, kind, draw, m)) for m in EXAMPLE_METHODS}
        for r, ch in enumerate(("u", "x")):
            ax = axes[r, c]
            plotting.truth_and_data(ax, fits["mi"], ch, name)
            for m in EXAMPLE_METHODS:
                f, band = fits[m], POSTERIOR_BANDS[m]
                plotting.posterior_band(ax, f["ts"], f[f"{ch}_mean"], f[f"{ch}_sd"], POSTERIOR_COLOURS[m], band,
                                        zorder=2 if band == "filled" else 3)
                ax.plot(f["ts"], f[f"{ch}_mean"], color=POSTERIOR_COLOURS[m], lw=plotting.MEAN_LW,
                        zorder=4 if m == "mi" else 3)
            plotting.posterior_limits(ax, fits["mi"], ch)
        system = "" if cond == "sir" else f"{plotting.SYSTEM_SHORT[name]}, "        # SIR is its own system
        axes[0, c].set_title(f"{robustness.CONDITIONS[cond]}\n{system}draw {draw + 1}")
    _posterior_columns_axes(axes)
    _close_columns(fig)
    handles = [plotting.posterior_handle(POSTERIOR_COLOURS[m], POSTERIOR_BANDS[m]) for m in EXAMPLE_METHODS]
    handles += [plotting.truth_handle(), plotting.data_handle(alpha=0.6)]
    labels = [METHOD_LEGEND[m] for m in EXAMPLE_METHODS] + ["truth", "data"]
    plotting.shared_legend(fig, handles=handles, labels=labels, ncol=len(labels))
    print(f"wrote {plotting.save(fig, 'robustness_examples')}  {[(c, n, d + 1) for c, n, _, d in examples]}")


#: the figures this script draws, by name (`--figure NAME`)
FIGURES = {"failures_by_method": figures_by_method, "robustness_examples": figure_robustness_examples}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--compute", action="store_true", help="fit every posterior the figures need")
    parser.add_argument("--jobs", type=int, default=1, help="parallel processes for the fits")
    parser.add_argument("--figure", nargs="*", choices=sorted(FIGURES), default=None,
                        help="draw these figures from results/ (all without names)")
    args = parser.parse_args()
    if args.compute:
        compute(args.jobs)
    if args.figure is not None or not args.compute:
        for name in args.figure or FIGURES:
            FIGURES[name]()


if __name__ == "__main__":
    main()
