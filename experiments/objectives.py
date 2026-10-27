"""Fig. 4 of the paper: the sigma_x^2 objectives compared over many datasets, the whole algorithm at one design.

On each benchmark system at its K0 and draws 0-19 (seeds and nested data of `benchmark.system_kw`), every method
of `benchmark.COMPARED` reports its fit: the algorithm with the criterion ell_OBS (`obs`), I (`mi`), D
(`di`) and `log p(Y | Z)` at the IEKS (`fenrir_ieks`), and the classical LFM with RK4
(`classical_rk4`). The methods are compared with the best method on each dataset (a Dolan--More performance
profile): a fit is wrong when its force error exceeds `WRONG` times the best.

    uv run python experiments/objectives.py fhn_rough --draw 7 --methods obs   # results/objectives/
    uv run python experiments/objectives.py --figure                           # figures/comparison.pdf

`make results-objectives` runs every (system, draw, method) job, `JOBS` at a time, single-threaded.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as onp
from matplotlib.ticker import FixedLocator, FuncFormatter

from lfcal.paper import plotting
from lfcal.paper.benchmark import (
    ALGORITHMS,
    COMPARED,
    DESIGNS,
    DRAWS,
    WRONG,
    chi2_band,
    error_ratios,
    run_method,
    summary,
    system_kw,
)
from lfcal.paper.io import read_csv, write_csv
from lfcal.paper.plotting import (
    DASHED,
    DOTTED,
    METHOD_COLOURS,
    METHOD_MARKERS,
    SYSTEM_NAMES,
    hline,
    jitter,
    log_axis,
    marker_scale,
    method_axis,
    outline_wrong,
    shade_band,
    subplots,
)

RESULTS = Path(__file__).resolve().parents[1] / "results" / "objectives"


# ------------------------------------------------------------------ results
def path(name, draw, method, root=RESULTS):
    """`{name}_s{draw}_{method}.csv`: one row, the method's entry."""
    return Path(root) / f"{name}_s{draw}_{method}.csv"


def load(name, draw, method, root=RESULTS):
    """A method's entry; None if not recorded."""
    p = path(name, draw, method, root)
    return read_csv(p)[0] if p.exists() else None


def save(name, draw, method, entry, root=RESULTS):
    Path(root).mkdir(parents=True, exist_ok=True)
    write_csv(path(name, draw, method, root), [entry])


# ------------------------------------------------------------------ computation
def run(name, draw, methods):
    """Every missing entry of `methods` on one dataset, each recorded when computed."""
    K0 = DESIGNS[name].K0
    for method in methods:
        if load(name, draw, method) is not None:
            continue
        entry = run_method(name, K0, method, **system_kw(name, draw, K0))
        save(name, draw, method, entry)
        print(f"{name} s{draw} {method}: {summary(entry)}", flush=True)


# ------------------------------------------------------------------ evaluation
def _method_record(entry, ratio, log_theta_true):
    """One method's fields of a dataset's record. The force hyperparameters are log10 ratios to the truth."""
    return {"log10_xi": (entry.get("log_xi_u", onp.nan) - log_theta_true[0]) / onp.log(10),
            "log10_ell": (entry.get("log_ell_u", onp.nan) - log_theta_true[1]) / onp.log(10),
            "ratio": ratio, "chi2_u": entry.get("chi2_u", onp.nan)}


def _dataset_record(draw, entries):
    """One dataset's record (see `collect`)."""
    ratios, _ = error_ratios(entries)
    ref = next(e for m, e in entries.items() if m in ALGORITHMS and "dlog_xi_u" in e)
    log_theta_true = (ref["log_xi_u"] - ref["dlog_xi_u"], ref["log_ell_u"] - ref["dlog_ell_u"])
    return {"draw": draw, **{m: _method_record(e, ratios[m], log_theta_true) for m, e in entries.items()}}


def collect(names=tuple(DESIGNS), draws=DRAWS):
    """Per system, a record per dataset with every method recorded.

    A record holds per method `ratio` (its force error over the best method's), `log10_xi` and `log10_ell` (the
    fitted force variance and length scale over the truth) and `chi2_u`.
    """
    def complete(name, draw):
        entries = {m: load(name, draw, m) for m in COMPARED}
        return _dataset_record(draw, entries) if all(e is not None for e in entries.values()) else None

    return {name: [rec for rec in (complete(name, draw) for draw in draws) if rec is not None] for name in names}


# ------------------------------------------------------------------ the figure
#: the error ratio axis' ticks
RATIO_TICKS = (1, 1.5, 2, 3, 5, 10)
#: the Matern smoothness nu = q_u + 1/2 of the force prior (q_u = 2), and the cut of the roughness axis (log10
#: over the truth) below which a fit's force is switched off or far too smooth
NU = 2.5
ROUGH_CUT = -2.5
CHI2_LIM = (0.1, 100.0)


def save_figure(fig, name):
    print(f"wrote {plotting.save(fig, name)}")


def _complete(data, what="figure"):
    """The comparison data (`collect` unless given) and the systems with complete datasets, or None after
    saying that the `what` is skipped."""
    data = data or collect()
    names = [n for n in DESIGNS if data.get(n)]
    if not names:
        print(f"objectives: no complete datasets in {RESULTS}; {what} skipped")
        return None
    return data, names


def roughness(recs, m):
    """log10 of the force roughness sigma_u^2 / ell_u^(2 nu) over the truth, per dataset.

    This is the microergodic parameter of the Matern prior, the combination of its hyperparameters the data
    identify (Zhang 2004), which sets the prior's power at high frequencies.
    """
    return onp.array([r[m]["log10_xi"] - 2 * NU * r[m]["log10_ell"] for r in recs])


def any_rough_cut(rough):
    """Whether any fit of `{name: {method: roughness}}` lies below the cut."""
    return any((v < ROUGH_CUT).any() for d in rough.values() for v in d.values())


def force_errors(recs, m):
    """Per dataset, method `m`'s force error over the best method's."""
    return onp.array([r[m]["ratio"] for r in recs])


def error_range(values):
    """`(bottom, top)` of an error ratio axis over the finite `values`."""
    v = onp.asarray([x for x in values if onp.isfinite(x)])
    return 0.95, 1.15 * min(max(2.0, v.max()), 10.0)


def _strip_panels(axes, x, recs, m, rough, top):
    """Method `m`'s dots at `x` in the three panels, its wrong fits outlined in each: the force error ratio, the
    force chi^2, and the roughness ratio (below the cut, in the strip under it)."""
    style = {"s": 6 * marker_scale(m) ** 2, "color": METHOD_COLOURS[m], "marker": METHOD_MARKERS[m],
             "linewidths": 0.6, "zorder": 3}
    ys = [onp.minimum(force_errors(recs, m), top), onp.clip([r[m]["chi2_u"] for r in recs], *CHI2_LIM),
          onp.where(rough < ROUGH_CUT, ROUGH_CUT - 0.5, rough)]
    for ax, y in zip(axes, ys, strict=True):
        ax.scatter(x, y, **style)
    wrong = force_errors(recs, m) > WRONG
    if wrong.any():
        for ax, y in zip(axes, ys, strict=True):
            outline_wrong(ax, x[wrong], y[wrong], style)


def _reference_lines(axes, cut, name):
    """The panels' reference marks: dashed, the wrong-fit threshold and chi^2 = 1; grey, the chi^2 interval of
    system `name` (`benchmark.chi2_band`); solid, the true roughness; dotted, the roughness cut."""
    shade_band(axes[1], *chi2_band(name))
    hline(axes[0], WRONG, ls=DASHED)
    hline(axes[1], 1.0, ls=DASHED)
    hline(axes[2], 0.0)
    if cut:
        hline(axes[2], ROUGH_CUT, ls=DOTTED, lw=0.6)


def _limits(vals, pad=0.06, include=()):
    """`(lo, hi)` over the finite `vals` and `include`, with `pad` of the span (at least 0.3) on each side."""
    v = onp.concatenate([onp.asarray(vals, dtype=float).ravel(), onp.asarray(include, dtype=float)])
    v = v[onp.isfinite(v)]
    margin = pad * max(v.max() - v.min(), 0.3)
    return v.min() - margin, v.max() + margin


def _error_axis(ax, bottom, top):
    log_axis(ax.yaxis, bottom, top * 1.05, RATIO_TICKS)
    ax.set_ylabel("error ratio")


def _chi2_axis(ax, chi2):
    """The force chi^2 axis, fitted to the plotted values `chi2` (and the bands' edges)."""
    lo, hi = (10 ** v for v in _limits(onp.log10(onp.clip(chi2, *CHI2_LIM)), include=(0.0,)))
    decades = onp.log10(hi / lo) > 2.5
    log_axis(ax.yaxis, lo, hi, (0.1, 1.0, 10.0, 100.0) if decades else (0.1, 0.3, 1.0, 3.0, 10.0, 30.0))
    ax.set_ylabel(r"force $\chi^2$")


def _roughness_axis(ax, rough):
    """The roughness ratio axis, symmetric about the truth, with a strip below the cut if a fit lies there."""
    vals = [v for d in rough.values() for v in d.values()]
    cut = any_rough_cut(rough)
    span = max(1.0, max(onp.abs(v[v >= ROUGH_CUT]).max() for v in vals) + 0.3)
    strip = ROUGH_CUT - 0.5
    ax.set_ylim(strip - 0.4 if cut else -span, span)
    ticks = [t for t in (-2.0, -1.0, 0.0, 1.0, 2.0) if -span <= t <= span]
    ax.yaxis.set_major_locator(FixedLocator(([strip] if cut else []) + ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: r"$\ll 1$" if v < ROUGH_CUT else f"{10 ** v:g}"))
    ax.set_ylabel("roughness ratio")


def figure_comparison(data=None):
    """A column per system, a dot per dataset and method, the wrong fits outlined.

    Rows: the force error over the best method's, the force chi^2 with its 95 % interval under a calibrated
    posterior with errors correlated as the true force prior (grey, `benchmark.chi2_band`), and the force
    roughness over the truth (`roughness`).
    """
    if (complete := _complete(data)) is None:
        return
    data, names = complete
    bottom, top = error_range([v for recs in data.values() for m in COMPARED for v in force_errors(recs, m)])
    rough = {name: {m: roughness(data[name], m) for m in COMPARED} for name in names}
    cut = any_rough_cut(rough)
    fig, axes = subplots(3, len(names), squeeze=False, sharex=True, sharey="row", height_to_width_ratio=0.4)
    rng = onp.random.default_rng(0)
    for c, name in enumerate(names):
        recs = data[name]
        for i, m in enumerate(COMPARED):
            _strip_panels(axes[:, c], jitter(i, len(recs), rng), recs, m, rough[name][m], top)
        _reference_lines(axes[:, c], cut, name)
        axes[0, c].set_title(SYSTEM_NAMES[name])
        method_axis(axes[-1, c], COMPARED)
    chi2 = [r[m]["chi2_u"] for name in names for r in data[name] for m in COMPARED]
    chi2 += [b for name in names for b in chi2_band(name)]
    _error_axis(axes[0, 0], bottom, top)
    _chi2_axis(axes[1, 0], chi2)
    _roughness_axis(axes[2, 0], rough)
    fig.align_ylabels(axes[:, 0])
    save_figure(fig, "comparison")


#: the figure, by name (`--figure NAME`)
FIGURES = {"comparison": figure_comparison}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("systems", nargs="*", choices=sorted(DESIGNS), default=[],
                        help="compute the missing entries of these systems")
    parser.add_argument("--draw", type=int, nargs="+", default=list(DRAWS))
    parser.add_argument("--methods", nargs="+", choices=COMPARED, default=list(COMPARED))
    parser.add_argument("--figure", nargs="*", choices=list(FIGURES), default=None,
                        help="draw the figure from results/")
    args = parser.parse_args()
    for name in args.systems:
        for draw in args.draw:
            run(name, draw, args.methods)
    if args.figure is not None or not args.systems:
        for figure_name in args.figure or FIGURES:
            FIGURES[figure_name]()


if __name__ == "__main__":
    main()
