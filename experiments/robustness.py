"""Fig. `fig:robustness`: the sigma_x^2 objectives beyond the benchmark of `objectives`.

Two sets of datasets, each run with the methods of the comparison at the system's K0 (the whole algorithm):

  misspecified   the benchmark systems with a truth force that is not a draw of the model's Matern-5/2 prior:
                 a rougher Matern-3/2 draw, a sinusoid, or a piecewise-constant force (`systems.FORCE_KINDS`),
                 `MISSPECIFIED_DRAWS` draws each
  sir            the SIR model, whose force enters multiplicatively through the transmission rate
                 (`systems.sir`), `SIR_DRAWS` draws of a Matern-5/2 force

A fit is wrong when its force error exceeds `benchmark.WRONG` times the best method's on its dataset.

    uv run python experiments/robustness.py sir --draw 3 --methods mi     # results/robustness/
    uv run python experiments/robustness.py --figure                      # figures/robustness.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as onp
import objectives

from lfcal.paper import benchmark, plotting
from lfcal.paper.benchmark import COMPARED, WRONG
from lfcal.paper.io import read_csv, write_csv

RESULTS = Path(__file__).resolve().parents[1] / "results" / "robustness"
MISSPECIFIED = ("matern32", "sine", "step")
MISSPECIFIED_DRAWS = tuple(range(10))
SIR_K0, SIR_DRAWS = 150, tuple(range(20))
#: the figure's conditions: each misspecified force pooled over the systems, and SIR
CONDITIONS = {"matern32": "Matérn-3/2", "sine": "sinusoid", "step": "piecewise constant", "sir": "SIR"}


def datasets():
    """Every `(name, kind, draw)`."""
    misspecified = [(name, kind, draw) for kind in MISSPECIFIED for name in benchmark.DESIGNS
                    for draw in MISSPECIFIED_DRAWS]
    return misspecified + [("sir", "matern", draw) for draw in SIR_DRAWS]


def condition(name, kind):
    """The dataset's condition of `CONDITIONS`."""
    return "sir" if name == "sir" else kind


def k0(name):
    return SIR_K0 if name == "sir" else benchmark.DESIGNS[name].K0


def system_kw(name, kind, draw):
    """The factory arguments of a dataset: its draw's seeds (and nesting, on a benchmark system) and its force."""
    kw = benchmark.draw_seeds(draw) if name == "sir" else benchmark.system_kw(name, draw, k0(name))
    return {**kw, "force_kind": kind}


def path(name, kind, draw, method, root=RESULTS):
    return Path(root) / f"{name}_{kind}_s{draw}_{method}.csv"


def load(name, kind, draw, method, root=RESULTS):
    """One method's entry on one dataset; None if not recorded."""
    p = path(name, kind, draw, method, root)
    return read_csv(p)[0] if p.exists() else None


def save(name, kind, draw, method, entry, root=RESULTS):
    Path(root).mkdir(parents=True, exist_ok=True)
    write_csv(path(name, kind, draw, method, root), [entry])


def run(name, kind, draw, method):
    """One method's entry on one dataset, unless recorded."""
    if load(name, kind, draw, method) is not None:
        return
    entry = benchmark.run_method(name, k0(name), method, **system_kw(name, kind, draw))
    save(name, kind, draw, method, entry)
    print(f"{name} {kind} s{draw} {method}: {benchmark.summary(entry)}", flush=True)


def collect():
    """`{(name, kind): [per dataset {method: record}]}` over the datasets with every method recorded.

    A method's record holds its force error over the best method's (`ratio`) and its force chi^2 (`chi2_u`).
    """
    out = {}
    for name, kind, draw in datasets():
        entries = {m: load(name, kind, draw, m) for m in COMPARED}
        if any(e is None for e in entries.values()):
            continue
        ratios, _ = benchmark.error_ratios(entries)
        out.setdefault((name, kind), []).append({m: {"ratio": ratios[m], "chi2_u": e.get("chi2_u", onp.nan)}
                                                 for m, e in entries.items()})
    return out


def _groups(data):
    """The datasets per condition of `CONDITIONS`."""
    return {c: [r for (name, kind), recs in data.items() for r in recs if condition(name, kind) == c]
            for c in CONDITIONS}


def figure():
    """One column per condition, one dot per dataset and method: the force error over the best method's (top)
    and the force chi^2 (bottom) with the 95 % interval of a calibrated posterior of each pooled system
    (`benchmark.chi2_band`, overlaid). The wrong fits are outlined in both rows, as in the comparison figure."""
    data = collect()
    groups = {c: recs for c, recs in _groups(data).items() if recs}
    if not groups:
        print(f"robustness: no results in {RESULTS}; figure skipped")
        return
    bottom, top = objectives.error_range([r[m]["ratio"] for recs in groups.values() for r in recs for m in COMPARED])
    fig, axes = plotting.subplots(2, len(groups), squeeze=False, sharex=True, sharey="row",
                                  height_to_width_ratio=0.65)
    rng = onp.random.default_rng(0)
    for c, (cond, recs) in enumerate(groups.items()):
        for i, m in enumerate(COMPARED):
            x = plotting.jitter(i, len(recs), rng)
            style = {"s": 5 * plotting.marker_scale(m) ** 2, "color": plotting.METHOD_COLOURS[m],
                     "marker": plotting.METHOD_MARKERS[m], "linewidths": 0.6, "zorder": 3}
            ys = (onp.minimum([r[m]["ratio"] for r in recs], top),
                  onp.clip([r[m]["chi2_u"] for r in recs], *objectives.CHI2_LIM))
            wrong = onp.array([r[m]["ratio"] for r in recs]) > WRONG
            for ax, y in zip(axes[:, c], ys, strict=True):
                ax.scatter(x, y, **style)
                if wrong.any():
                    plotting.outline_wrong(ax, x[wrong], y[wrong], style)
        plotting.hline(axes[0, c], WRONG, ls=plotting.DASHED)
        plotting.hline(axes[1, c], 1.0, ls=plotting.DASHED)
        names = sorted({name for (name, kind) in data if condition(name, kind) == cond})
        for name in names:
            plotting.shade_band(axes[1, c], *benchmark.chi2_band(name, k0(name)), alpha=0.15 / len(names) ** 0.5)
        axes[0, c].set_title(f"{CONDITIONS[cond]} ($n={len(recs)}$)")
        plotting.method_axis(axes[1, c], COMPARED, rotation=90)
    plotting.log_axis(axes[0, 0].yaxis, bottom, top * 1.05, objectives.RATIO_TICKS)
    axes[0, 0].set_ylabel("error ratio")
    plotting.log_axis(axes[1, 0].yaxis, objectives.CHI2_LIM[0], objectives.CHI2_LIM[1] * 1.3, (0.1, 1.0, 10.0, 100.0))
    axes[1, 0].set_ylabel(r"force $\chi^2$")
    fig.align_ylabels(axes[:, 0])
    print(f"wrote {plotting.save(fig, 'robustness')}")


#: the figure, by name (`--figure NAME`)
FIGURES = {"robustness": figure}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("systems", nargs="*", choices=[*benchmark.DESIGNS, "sir"], default=[])
    parser.add_argument("--kind", nargs="+", default=None, help="default: every kind of the system (`datasets`)")
    parser.add_argument("--draw", type=int, nargs="+", default=None)
    parser.add_argument("--methods", nargs="+", choices=COMPARED, default=list(COMPARED))
    parser.add_argument("--figure", nargs="*", choices=sorted(FIGURES), default=None,
                        help="draw these figures from results/ (all without names)")
    args = parser.parse_args()
    selected = [(name, kind, draw) for name, kind, draw in datasets() if name in args.systems
                and (args.kind is None or kind in args.kind) and (args.draw is None or draw in args.draw)]
    for name, kind, draw in selected:
        for m in args.methods:
            run(name, kind, draw, m)
    if args.figure is not None or not args.systems:
        for name in args.figure or FIGURES:
            FIGURES[name]()


if __name__ == "__main__":
    main()
