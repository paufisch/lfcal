"""Fig. `fig:force-information`: the force error and the data's information about the force given the ODE,
`I(U; Y | Z)`, against sigma_x^2, one line per grid density kappa = 2 ... 128.

Per system and kappa, with `theta_u` at the truth and scored at the observation times, every sigma_x of the
system's sweep window runs on the IEKS-converged linearisation, by sigma_x continuation
(`lfcal.paper.information.scan_cell`). The picks are the maximisers of `I(U; Y | Z)` (diamonds), of the directed
information `D` (triangles) and of `log p(Y | Z)` (filled plus), and the largest `ell_OBS` (stars).

The systems are FitzHugh--Nagumo, Duffing and Lorenz-96 (`PLOT_SYSTEMS`), and the benchmark problems `CASES`
on which `ell_OBS` fails although the force hyperparameters are fitted. The figures show Duffing at one
column's width (`force_information_duffing.pdf`) and the failing problems with every criterion's row
(`force_information_extended_failures.pdf`). The maximisers of `I(U; Y | Z)` on every system are also marked
in the figure of `sigma_x_sweep`.

    uv run python experiments/force_information.py duffing fhn lorenz96   # results/force_information/
    uv run python experiments/force_information.py --figure               # figures/force_information_*.pdf
"""

from __future__ import annotations

import argparse
import functools
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as onp
from matplotlib.ticker import FixedLocator, MaxNLocator, NullFormatter, NullLocator

from lfcal.paper import benchmark, plotting, systems
from lfcal.paper.information import scan_cell
from lfcal.paper.io import read_csv, write_csv
from lfcal.paper.iterated import STEP_TOL

RESULTS = Path(__file__).resolve().parents[1] / "results" / "force_information"
KAPPAS = (2, 4, 8, 16, 32, 64, 128)
PLOT_SYSTEMS = ("fhn", "duffing", "lorenz96")
#: benchmark problems `(system, draw)` on which ell_OBS fails although phi_u is fitted: its maximum and the
#: force error's minimum lie apart
CASES = {"fhn_rough_s0": ("fhn_rough", 0), "duffing_accel_s10": ("duffing_accel", 10),
         "lorenz96_da_s9": ("lorenz96_da", 9)}
#: every system this script runs
SYSTEMS = (*PLOT_SYSTEMS, *CASES)
#: column titles; draws are counted from one
TITLES = {**{name: plotting.SYSTEM_NAMES[name] for name in PLOT_SYSTEMS},
          **{case: f"{plotting.SYSTEM_NAMES[name]}, draw {draw + 1}" for case, (name, draw) in CASES.items()}}
#: lower `sigma_x^2` limit of the plot: below ~2e4 Lorenz-96's IEKS converges at no kappa
X_MIN = {"lorenz96": 5e4}
#: the per-kappa fields of a cell (`{name}_cells.csv`)
CELL = ("kappa", "N", "K", "n_iters", "best_rel_u")


def path(name, part="", root=RESULTS):
    """`{name}.csv` (one row per kappa and sigma_x), `{name}_picks.csv` (one row per kappa and pick),
    `{name}_cells.csv` (one row per kappa)."""
    return Path(root) / f"{name}{part}.csv"


def save(name, cells, root=RESULTS):
    """`{str(kappa): scan_cell(...)}` as three CSV tables."""
    Path(root).mkdir(parents=True, exist_ok=True)
    cs = list(cells.values())
    write_csv(path(name, root=root), [{"kappa": c["kappa"], **r} for c in cs for r in c["rows"]])
    write_csv(path(name, "_picks", root), [{"kappa": c["kappa"], "pick": pick, **r}
                                           for c in cs for pick, r in c["picks"].items()])
    write_csv(path(name, "_cells", root), [{f: c[f] for f in CELL} for c in cs])


def load(name, root=RESULTS):
    """Inverse of `save`; `{}` before the first cell."""
    if not path(name, "_cells", root).exists():
        return {}
    cells = {str(c["kappa"]): dict(c, rows=[], picks={}) for c in read_csv(path(name, "_cells", root))}
    for r in read_csv(path(name, root=root)):
        cells[str(r.pop("kappa"))]["rows"].append(r)
    for r in read_csv(path(name, "_picks", root)):
        cells[str(r.pop("kappa"))]["picks"][r.pop("pick")] = r
    return cells


# ---------------------------------------------------------------- the cells
@functools.cache
def _system_and_window(name):
    """The system of `name` and its sweep window; a benchmark problem of `CASES` at its K0."""
    if name in CASES:
        base, draw = CASES[name]
        K0 = benchmark.DESIGNS[base].K0
        system = systems.build(base, n_obs=K0, **benchmark.system_kw(base, draw, K0))
        return system, tuple(systems.sweep_window(base, system))
    system = systems.build(name)
    return system, tuple(systems.sweep_window(name, system))


def compute(name, kappa):
    """One cell: every sigma_x of the system's window on the grid of density `kappa`, and the picks."""
    system, window = _system_and_window(name)
    return scan_cell(system, kappa, onp.asarray(window), system.log_theta_u_true, systems.EK1)


def _summary(name, cell, seconds):
    picks = cell["picks"]
    n_unconverged = sum(1 for r in cell["rows"] if r["ok"] and r["step"] > STEP_TOL)
    return (f"{name:9s} kappa={cell['kappa']:3d} N={cell['N']:6d}  max I(U;Y|Z) {picks['max_info']['cost']:.2f}x "
            f"the best force error (chi2_u {picks['max_info']['chi2_u']:.2f})  max D {picks['max_di']['cost']:.2f}x "
            f"(chi2_u {picks['max_di']['chi2_u']:.2f})  ell_obs {picks['ell_obs']['cost']:.2f}x "
            f"(chi2_u {picks['ell_obs']['chi2_u']:.2f})  unconverged {n_unconverged}/{len(cell['rows'])}  "
            f"({seconds:.0f} s)")


def run(name, kappas=KAPPAS):
    """Every missing cell of one system, saved after each, so that an interrupted run resumes."""
    cells = load(name)
    for kappa in (k for k in kappas if str(k) not in cells):
        t0 = time.time()
        cells = {**cells, str(kappa): compute(name, kappa)}
        save(name, cells)
        print(_summary(name, cells[str(kappa)], time.time() - t0), flush=True)
    return cells


# ---------------------------------------------------------------- figures

#: relative force error axis: per system from just under its best to the tick above its largest plotted
#: value, never beyond `Y_TOP`
Y_TOP = 50.0
Y_TICKS = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
#: lower limit of the `ll_obs` and `logp` rows, nats below the maximum
LL_FLOOR = -40.0


def _plotted(r):
    """A sweep point is drawn if its IEKS is finite and converged."""
    return r["ok"] and r["I_Z"] > 0 and r["step"] <= STEP_TOL


def _below_max(field):
    """The value `field` of a sweep row or pick below its maximum over the cell's drawn points."""
    return lambda r, cell: r[field] - max(q[field] for q in cell["rows"] if _plotted(q))


#: the rows a figure can show: the value of a sweep row or pick (given its cell), and the y label. `D` is the
#: directed information of the data about the force given the ODE; `ll_obs` (labelled L_OBS) and `logp` are
#: drawn below their maximum over the window, which the captions say.
ROWS = {
    "rel_u": (lambda r, cell: r["rel_u"], "relative\nforce error"),
    "I_YZ": (lambda r, cell: r["I_YZ"], r"$I(U;Y\mid Z)$"),
    "D": (lambda r, cell: r["D"], r"$D(U;Y \mid Z)$"),
    "ll_obs": (_below_max("ll_obs"), r"$\mathcal{L}_{\rm OBS}$"),
    "logp": (_below_max("logp"), r"$\log p(Y\mid Z)$"),
}
#: per pick: its marker (`plotting.PICK_MARKERS`) and legend label
PICKS = {"max_info": ("max_info", r"max $I(U;Y\mid Z)$"), "max_di": ("max_di", r"max $D(U;Y \mid Z)$"),
         "ell_obs": ("ll_obs", r"$\mathcal{L}_{\rm OBS}$ pick"), "ell_ode": ("ll_ode", r"$\mathcal{L}_{\rm ODE}$ pick"),
         "max_logp": ("max_logp", r"max $\log p(Y\mid Z)$")}
#: the picks read off the sweep rows at plot time (the largest value on the sigma_x grid) rather than
#: recorded by `scan_cell`
ROW_PICKS = {"ell_ode": "ll_ode"}
#: the row of each pick's own criterion (the `ell_ODE` pick, whose criterion has no row, on the force error)
PICK_ROW = {"max_info": "I_YZ", "max_di": "D", "ell_obs": "ll_obs", "ell_ode": "rel_u", "max_logp": "logp"}


def own_rows(picks):
    """Each pick marked on its own criterion's row only (`FigureSpec.pick_rows`)."""
    return {p: (PICK_ROW[p],) for p in picks}


@dataclass(frozen=True)
class FigureSpec:
    """What a force-information figure shows: one column per system of `names`, one row per entry of `rows`
    (`ROWS`) against sigma_x^2, with the `picks` marked.

    `pick_rows`: `{pick: rows}`, the only rows a pick is marked on (a pick not in it is marked on every
    row). `column="half"`: one text column wide. `titles`, `legend`: the column titles and the picks'
    legend (else the caption describes them). `panel_labels`: (a), (b), ... in each panel's top left
    corner. `log10_x`: the axis is log10 sigma_x^2 on a linear scale instead of sigma_x^2 on a log scale.
    `force_error_top`: the force error axis' upper limit instead of the tick above the largest value.
    """

    names: tuple = PLOT_SYSTEMS
    rows: tuple = ("rel_u", "I_YZ")
    picks: tuple = ("max_info", "ell_obs")
    pick_rows: dict | None = None
    height_to_width_ratio: float = 0.53
    column: str = "full"
    titles: bool = True
    panel_labels: bool = False
    legend: bool = True
    log10_x: bool = False
    force_error_top: float | None = None

    def marks(self, pick, row):
        return self.pick_rows is None or pick not in self.pick_rows or row in self.pick_rows[pick]


def _force_error_axis(ax, cells, sigma_min):
    """Log axis from just under the best force error to the tick above the largest plotted one."""
    best = min(c["best_rel_u"] for c in cells.values())
    worst = max(r["rel_u"] for c in cells.values() for r in c["rows"] if _plotted(r) and r["sigma_x"] >= sigma_min)
    lo = max([t for t in Y_TICKS if t <= 0.8 * best], default=Y_TICKS[0])
    hi = min([t for t in Y_TICKS if t >= 1.05 * worst], default=Y_TOP)
    ax.set_yscale("log")
    ax.set_ylim(lo, hi)
    ticks = [t for t in Y_TICKS if lo <= t <= hi]
    if len(ticks) > 6:                       # keep the decades on a tall range
        ticks = [t for t in ticks if t in (0.1, 1.0, 10.0) or t in (lo, hi)]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(plotting.plain_formatter())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_formatter(NullFormatter())


def _label_headroom(ax, corner=0.2, below=0.75):
    """Raise a linear axis' top by a quarter when a curve rises above `below` of its height within the left
    `corner` of its width, where the panel label sits."""
    (x0, x1), (_, top) = ax.get_xlim(), ax.get_ylim()
    left = x0 + corner * (x1 - x0)
    in_corner = [y for line in ax.lines for x, y in zip(line.get_xdata(), line.get_ydata(), strict=True)
                 if x <= left and onp.isfinite(y)]
    if in_corner and max(in_corner) > below * top:
        ax.set_ylim(top=1.25 * top)


def _draw_system(axes, cells, spec, colour):
    """One column: per kappa, each row's curve over the drawn points and its picks."""
    to_x = onp.log10 if spec.log10_x else onp.asarray
    for kappa, cell in sorted(cells.items(), key=lambda t: int(t[0])):
        col = colour[int(kappa)]
        plotted = sorted((r for r in cell["rows"] if _plotted(r)), key=lambda r: r["sigma_x"])
        x = to_x(onp.array([r["sigma_x"] for r in plotted]))
        for ax, row in zip(axes, spec.rows, strict=True):
            value = ROWS[row][0]
            ax.plot(x, onp.array([value(r, cell) for r in plotted]), color=col, lw=1.0)
            for pick in (p for p in spec.picks if spec.marks(p, row)):
                at = (max(plotted, key=lambda r, f=ROW_PICKS[pick]: r[f]) if pick in ROW_PICKS
                      else cell["picks"][pick])
                ax.plot([to_x(at["sigma_x"])], [value(at, cell)], **plotting.pick_style(PICKS[pick][0], col), zorder=6)


def _style_system(axes, cells, spec, sigma_lim):
    """One column's axes: the sigma_x^2 scale and range, and each row's y range."""
    to_x = onp.log10 if spec.log10_x else onp.asarray
    for ax, row in zip(axes, spec.rows, strict=True):
        if spec.log10_x:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        else:
            ax.set_xscale("log")
        ax.set_xlim(*to_x(onp.array(sigma_lim)))
        if row == "rel_u":
            _force_error_axis(ax, cells, sigma_lim[0])
            if spec.force_error_top:
                ax.set_ylim(top=spec.force_error_top)
        elif row in ("ll_obs", "logp"):
            ax.set_ylim(LL_FLOOR, -0.1 * LL_FLOOR)
        else:
            ax.set_ylim(0, 1.08 * ax.get_ylim()[1])           # room for the picks at the maximum
            if spec.panel_labels:                             # headroom where a curve runs under the label
                _label_headroom(ax)
    axes[-1].set_xlabel(r"$\log_{10} \sigma_x^2$" if spec.log10_x else r"$\sigma_x^2$")


def figure(out, spec: FigureSpec):
    """The figure `spec`, one line per kappa coloured by the grid density, saved as `figures/{out}.pdf`."""
    data = {name: load(name) for name in spec.names}
    if missing := [name for name, cells in data.items() if not cells]:
        print(f"force_information: no results for {', '.join(missing)}; {out} skipped")
        return
    all_kappas = sorted({int(k) for cells in data.values() for k in cells})
    cmap, norm, colour = plotting.grid_density_colours(all_kappas)
    fig, ax = plotting.subplots(len(spec.rows), len(spec.names), sharex="col", squeeze=False,
                                height_to_width_ratio=spec.height_to_width_ratio, column=spec.column, grid=True)
    for j, name in enumerate(spec.names):
        cells = data[name]
        sigmas = [r["sigma_x"] for c in cells.values() for r in c["rows"] if _plotted(r)]
        _draw_system(ax[:, j], cells, spec, colour)
        _style_system(ax[:, j], cells, spec, (max(min(sigmas), X_MIN.get(name, 0.0)), max(sigmas)))
        if spec.titles:
            ax[0, j].set_title(TITLES[name])
    for i, row in enumerate(spec.rows):
        ax[i, 0].set_ylabel(ROWS[row][1])
    if spec.panel_labels:
        for n, a in enumerate(ax.ravel()):
            plotting.panel_label(a, n, x=0.03)
    fig.align_ylabels(ax[:, 0])
    plotting.grid_density_colorbar(fig, ax, all_kappas, cmap, norm, pad=0.015, fraction=0.04, aspect=30)
    if spec.legend:
        handles = [plotting.pick_handle(*PICKS[p]) for p in spec.picks]
        plotting.shared_legend(fig, handles=handles, ncol=len(handles))
    print(f"wrote {plotting.save(fig, out)}")


#: one column wide, against log10 sigma_x^2, with panel labels and no titles or legend
NARROW = {"column": "half", "titles": False, "panel_labels": True, "legend": False, "log10_x": True}
#: the figures by output name
SPECS = {
    "force_information_duffing": FigureSpec(
        names=("duffing",), picks=("max_info", "ell_ode"), pick_rows={"ell_ode": ("rel_u",)}, height_to_width_ratio=0.32,
        **NARROW),
    "force_information_extended_failures": FigureSpec(
        names=tuple(CASES), rows=("rel_u", "ll_obs", "logp", "D", "I_YZ"),
        picks=("ell_obs", "max_logp", "max_di", "max_info"),
        pick_rows=own_rows(("ell_obs", "max_logp", "max_di", "max_info")), height_to_width_ratio=0.45),
}
#: the figures, by name (`--figure NAME`)
FIGURES = {out: functools.partial(figure, out, spec) for out, spec in SPECS.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("systems", nargs="*", choices=SYSTEMS, default=[], help="compute these systems' cells")
    parser.add_argument("--figure", nargs="*", choices=FIGURES, metavar="NAME",
                        help="draw these figures from results/ (default: all)")
    args = parser.parse_args()
    for name in args.systems:
        run(name)
    if args.figure is not None or not args.systems:
        for name in args.figure or FIGURES:
            FIGURES[name]()


if __name__ == "__main__":
    main()
