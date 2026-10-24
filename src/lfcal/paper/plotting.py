"""The figures' style, palette, labels and shared drawing helpers.

Style: the `tueplots` AISTATS 2025 bundle, applied once on import (`STYLE`); `subplots` sizes each figure for
the full text width or one column. Figures are saved as PDF through matplotlib's pgf backend, so LaTeX sets
all text (the usetex route of the PDF backend drops the minus sign of math mode in matplotlib 3.11).

Palette: series take their colours from the Tuebingen AI Center cycle (`colors`) where they can. The
posterior and method figures of the submitted paper are a deliberate exception: they use the colour-blind
safe Okabe--Ito palette, its blue and reddish purple with a navy, and a ramp through them for ordered
quantities. Every colour the figures use is defined in this module.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, LogNorm, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator
from tueplots import bundles, figsizes
from tueplots.constants.color import rgb

from lfcal.paper.benchmark import CLASSICAL

#: repository-level output directory of the figures (src/lfcal/paper/plotting.py: three levels below the root)
FIGURES = Path(__file__).resolve().parents[3] / "figures"
#: the bundle's default panel aspect (height / width)
GOLDEN_RATIO = (5 ** 0.5 - 1) / 2
#: grid density is written kappa in the paper (k is the Kalman step index)
GRID_SYMBOL = r"\kappa"

# ------------------------------------------------------------ style

#: every figure's style: the AISTATS 2025 bundle (its figure size is set per figure by `subplots`), with
#: pdflatex, so the pgf route sets text in the bundle's LaTeX fonts (XeLaTeX, the default, would substitute
#: matplotlib's serif font, DejaVu, for the text)
STYLE = {**bundles.aistats2025(), "pgf.texsystem": "pdflatex"}
plt.rcParams.update(STYLE)

# ------------------------------------------------------------ palette

#: the tue_ai_beamer colour cycle, for every data series that has no colour of its own below
colors = [tuple(c) for c in bundles.tue_ai_beamer()["axes.prop_cycle"].by_key()["color"]]
#: neutral grey for non-data ink (reference lines, truth, annotations)
NEUTRAL = tuple(rgb.tue_ai_dark)

#: the paper's posterior colours: the Okabe--Ito blue (the posterior, I(U;Y|Z)), its reddish purple (pi,
#: ell_OBS), and a navy, the dark end of the ramp (the classical LFM)
POSTERIOR_BLUE = "#0072B2"
POSTERIOR_PURPLE = "#CC5496"
POSTERIOR_NAVY = "#00293F"
#: the opacity of a filled +/- 2 sd band
BAND_ALPHA = 0.35
#: sequential ramp for ordered quantities (grid density, error ratios): purple through blue to navy, darker
#: at larger values
RAMP = LinearSegmentedColormap.from_list("purple_blue_ramp", [(0.0, POSTERIOR_PURPLE), (0.65, POSTERIOR_BLUE),
                                                              (1.0, POSTERIOR_NAVY)])

#: sRGB (linearised) -> LMS and cube-rooted LMS -> OKLab (Ottosson 2020)
_LMS = np.array([[0.4122214708, 0.5363325363, 0.0514459929], [0.2119034982, 0.6806995451, 0.1073969566],
                 [0.0883024619, 0.2817188376, 0.6299787005]])
_OKLAB = np.array([[0.2104542553, 0.7936177850, -0.0040720468], [1.9779984951, -2.4285922050, 0.4505937099],
                   [0.0259040371, 0.7827717662, -0.8086757660]])
#: the chroma factor of the grid-density ramp (`saturate`)
SATURATION = 1.2


def saturate(color, factor):
    """`color` with its OKLab chroma times `factor`, lightness and hue kept, as far as the sRGB gamut allows."""
    c = np.asarray(to_rgb(color), dtype=float)
    L, a, b = _OKLAB @ np.cbrt(_LMS @ np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4))

    def linear(f):
        return np.linalg.solve(_LMS, np.linalg.solve(_OKLAB, np.array([L, f * a, f * b])) ** 3)

    def inside(f):
        return np.all((linear(f) >= -1e-9) & (linear(f) <= 1 + 1e-9))

    if not inside(factor):                                   # the largest factor in the gamut, by bisection
        lo, hi = 1.0, factor
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if inside(mid) else (lo, mid)
        factor = lo
    c = np.clip(linear(factor), 0.0, 1.0)
    return tuple(np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055))


#: the ramp of the grid density: `RAMP` with its chroma raised by `SATURATION`
GRID_RAMP = LinearSegmentedColormap.from_list("grid_density_ramp", [saturate(RAMP(x), SATURATION)
                                                                   for x in np.linspace(0.0, 1.0, 256)])

#: one colour per method in the comparison figures: points of `RAMP`, as the submitted ones drew them. ell_OBS
#: is its purple end, log p(Y | Z) a third of the way to blue, I(U;Y|Z) its blue node, D most of the way to
#: navy and the classical LFM its navy end. pi of the failure-directions figure has ell_OBS's purple.
METHOD_COLOURS = {m: to_rgb(c) for m, c in (
    ("obs", RAMP(0.0)), ("fenrir_ieks", RAMP(0.33)), ("mi", POSTERIOR_BLUE), ("di", RAMP(0.88)),
    (CLASSICAL, RAMP(1.0)))}
#: the posterior figures' colour and +/- 2 sd band, as the submitted paper drew them: I(U;Y|Z) filled blue,
#: ell_OBS outlined purple, as is a failing method whatever it is, and the classical LFM navy with a dotted band
POSTERIOR_COLOURS = {"mi": to_rgb(POSTERIOR_BLUE), "obs": to_rgb(POSTERIOR_PURPLE), CLASSICAL: to_rgb(POSTERIOR_NAVY)}
POSTERIOR_BANDS = {"mi": "filled", "obs": "outlined", CLASSICAL: "dotted"}
#: the width of a posterior's mean line and of its band's outline
MEAN_LW, OUTLINE_LW = 0.9, 0.5
#: the outline of a wrong fit's marker: dark on a light marker, light on a dark one
OUTLINE, OUTLINE_LIGHT = "#383838", "#C8C8C8"

# ------------------------------------------------------------ markers and line styles

#: the `ell_obs` pick is a star, the `ell_ode` pick a square, the information maximum a diamond, the
#: directed-information maximum a triangle, the maximum of log p(Y | Z) a filled plus
PICK_MARKERS = {"ll_obs": "*", "ll_ode": "s", "max_info": "D", "max_di": "^", "max_logp": "P"}
_PICK_LABELS = {"ll_obs": r"$\mathcal{L}_{\rm OBS}$", "ll_ode": r"$\mathcal{L}_{\rm ODE}$"}
#: one marker per method, so that coinciding lines stay apart; a method that selects sigma_x^2 by a criterion
#: of the force-information figures takes that criterion's pick marker
METHOD_MARKERS = {"obs": "*", "mi": "D", "di": "^", "fenrir_ieks": "P", CLASSICAL: "o"}
#: a star or a triangle reads smaller than a square of the same size
MARKER_SCALE = {"*": 1.6, "^": 1.3}
PICK_MS = 4.0
#: reference lines: dashed for a target (chi^2 = 1, the wrong-fit threshold), dotted for a cut or a box edge
DASHED, DOTTED = (0, (6, 2)), (0, (1, 1.5))
#: the truth under the posteriors: dashed
TRUTH = {"color": NEUTRAL, "lw": 0.9, "ls": "--", "zorder": 1}

# ------------------------------------------------------------ labels

#: the systems' names in titles, and short names for narrow panels
SYSTEM_NAMES = {"duffing": "Duffing", "duffing_accel": "Duffing", "fhn": "FitzHugh--Nagumo",
                "fhn_rough": "FitzHugh--Nagumo", "lorenz96": "Lorenz-96", "lorenz96_da": "Lorenz-96", "sir": "SIR"}
SYSTEM_SHORT = {"duffing_accel": "Duffing", "fhn_rough": "FHN", "lorenz96_da": "Lorenz-96", "sir": "SIR"}
#: whether each system's data are its scored state (Duffing observes the acceleration, not the plotted position)
SHOWS_DATA = {"duffing_accel": False, "fhn_rough": True, "lorenz96_da": True, "sir": True}
#: the methods in a legend and as tick labels
METHOD_LEGEND = {"obs": r"$\mathcal{L}_{\rm OBS}$", "mi": r"$I(U;Y \mid Z)$",
                 "di": r"$D(U;Y \mid Z)$",
                 "fenrir_ieks": r"$p(Y \mid Z)$ at the IEKS", CLASSICAL: "classical LFM (RK4)"}
METHOD_TICKS = {"obs": r"$\mathcal{L}_{\rm OBS}$", "mi": "$I$", "di": "$D$",
                "fenrir_ieks": r"$p(Y \mid Z)$", CLASSICAL: "LFM"}


def grid_density_label(steps: bool = True) -> str:
    """`"grid density $kappa$   ($N = kappa K$)"`, or without the step count."""
    g = GRID_SYMBOL
    return f"grid density ${g}$   ($N = {g} K$)" if steps else f"grid density ${g}$"


# ------------------------------------------------------------ figures

def subplots(nrows=1, ncols=1, *, height_to_width_ratio=GOLDEN_RATIO, size_ncols=None, column="full", grid=False,
             **kwargs):
    """`plt.subplots` for an `nrows x ncols` grid of panels, sized by the AISTATS bundle.

    Each panel is `height_to_width_ratio` as high as wide, at the full text width or one `column` (`"half"`).
    `size_ncols`: the number of panel columns that share the width, when some subplot columns are not
    panels. `grid`: grid lines on the panels. The size and the grid are read when the figure and its axes
    are created, so they are set for that alone; the rest of the style is `STYLE`.
    """
    cols = size_ncols or ncols
    size = figsizes.aistats2025_full if column == "full" else figsizes.aistats2025_half
    with plt.rc_context({**size(nrows=nrows, ncols=cols, height_to_width_ratio=height_to_width_ratio),
                         "axes.grid": grid}):
        return plt.subplots(nrows, ncols, **kwargs)


def shared_legend(fig, *, handles, labels=None, loc="outside lower center", ncol=None, **kwargs):
    """One legend for the whole figure, outside the axes."""
    labels = labels if labels is not None else [h.get_label() for h in handles]
    return fig.legend(handles, labels, loc=loc, ncol=ncol or min(len(labels), 4), frameon=False, **kwargs)


def save(fig, name) -> Path:
    """Save `figures/<name>.pdf` (typeset by LaTeX) and close the figure."""
    path = (FIGURES / name).with_suffix(".pdf")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, backend="pgf")
    plt.close(fig)
    return path


def panel_label(ax, n, x=0.015, y=0.94, va="top"):
    """`(a)`, `(b)`, ... for panel `n`, in a white box at `(x, y)` in axes coordinates."""
    ax.text(x, y, f"({chr(ord('a') + n)})", ha="left", va=va, transform=ax.transAxes, zorder=7,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0})


def hline(ax, y, ls="-", lw=0.8):
    """A neutral horizontal reference line behind the data."""
    ax.axhline(y, color=NEUTRAL, lw=lw, ls=ls, zorder=0)


def shade_band(ax, lo, hi, alpha=0.15):
    """A neutral horizontal band behind the data, e.g. a chi^2 interval."""
    ax.axhspan(lo, hi, color=NEUTRAL, alpha=alpha, lw=0, zorder=0)


def plain_formatter():
    """Tick labels as plain numbers (`0.5`, `2`, `100`) on a log axis."""
    return FuncFormatter(lambda v, _: f"{v:g}")


def log_axis(axis, lo, hi, ticks):
    """A log scale on `axis` (`ax.xaxis` or `ax.yaxis`) from `lo` to `hi`, labelled at the `ticks` in range."""
    ax = axis.axes
    (ax.set_xscale if axis is ax.xaxis else ax.set_yscale)("log")
    (ax.set_xlim if axis is ax.xaxis else ax.set_ylim)(lo, hi)
    axis.set_major_locator(FixedLocator([t for t in ticks if lo <= t <= hi]))
    axis.set_minor_locator(NullLocator())
    axis.set_major_formatter(plain_formatter())


def grid_density_colours(kappas):
    """`(cmap, LogNorm, {kappa: colour})`: `GRID_RAMP` over `kappas`."""
    norm = LogNorm(vmin=min(kappas), vmax=max(kappas))
    return GRID_RAMP, norm, {k: GRID_RAMP(norm(k)) for k in kappas}


def grid_density_colorbar(fig, axes, kappas, cmap, norm, **kwargs):
    """The colour bar of the grid densities `kappas` beside `axes`."""
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=list(np.ravel(axes)), **kwargs)
    cb.set_label(grid_density_label())
    cb.set_ticks(kappas)
    cb.set_ticklabels([str(k) for k in kappas])
    cb.ax.minorticks_off()
    return cb


# ------------------------------------------------------------ methods and picks

def marker_scale(method):
    """Linear size factor of `method`'s marker (`MARKER_SCALE`); marker areas scale with its square."""
    return MARKER_SCALE.get(METHOD_MARKERS[method], 1.0)


def method_axis(ax, methods, **kwargs):
    """The x-axis of a strip plot: one column per method, labelled by `METHOD_TICKS`."""
    ax.set_xticks(range(len(methods)), [METHOD_TICKS[m] for m in methods], **kwargs)
    ax.set_xlim(-0.6, len(methods) - 0.4)


def jitter(i, n, rng, half_width=0.22):
    """`n` positions around column `i` of a strip plot."""
    return i + rng.uniform(-half_width, half_width, n)


def outline_wrong(ax, x, y, style):
    """The markers of `style` at `(x, y)` (the wrong fits) redrawn with a thin outline, dark on a light marker
    and light on a dark one, so that it shows on both."""
    r, g, b = to_rgb(style["color"])
    edge = OUTLINE if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.25 else OUTLINE_LIGHT
    ax.scatter(x, y, **{**style, "edgecolors": edge, "linewidths": 0.6, "zorder": 4})


def pick_style(pick: str, colour) -> dict:
    """Keyword arguments of a marked criterion's pick, shared by the point and its legend entry."""
    marker = PICK_MARKERS[pick]
    return {"ls": "none", "marker": marker, "ms": PICK_MS * MARKER_SCALE.get(marker, 1.0), "mfc": colour,
            "mec": "white", "mew": 0.6}


def pick_handle(pick: str, label: str | None = None, colour=NEUTRAL):
    return Line2D([], [], label=label or _PICK_LABELS[pick] + " pick", **pick_style(pick, colour))


# ------------------------------------------------------------ posteriors

def posterior_band(ax, ts, mean, sd, colour, band, *, lw=OUTLINE_LW, alpha=BAND_ALPHA, zorder=2):
    """The +/- 2 sd band of a posterior: `"filled"`, `"outlined"`, `"dotted"` (a dotted outline) or None."""
    lo, hi = mean - 2 * sd, mean + 2 * sd
    if band == "filled":
        ax.fill_between(ts, lo, hi, color=colour, alpha=alpha, lw=0, zorder=zorder)
    elif band is not None:
        for edge in (lo, hi):
            ax.plot(ts, edge, color=colour, lw=lw, ls="-" if band == "outlined" else DOTTED, zorder=zorder)


def truth_and_data(ax, fit, ch, name, truth=TRUTH):
    """The truth of channel `ch` (force `u`, scored state `x`) of a stored posterior `fit`, and on the state
    the data, where they are that state (`SHOWS_DATA`)."""
    ax.plot(fit["t_fine"], fit[f"{ch}_true"], **truth)
    if ch == "x" and SHOWS_DATA[name]:
        ax.plot(fit["t_obs"], fit["y"], ls="none", marker="o", ms=1.0, color=NEUTRAL, alpha=0.6, zorder=0)


def posterior_limits(ax, fit, ch, pad=0.35):
    """The limits of a posterior panel: the truth's range with `pad` of it on each side, and its time span."""
    lo, hi = np.min(fit[f"{ch}_true"]), np.max(fit[f"{ch}_true"])
    ax.set_ylim(lo - pad * (hi - lo), hi + pad * (hi - lo))
    ax.set_xlim(fit["t_fine"][0], fit["t_fine"][-1])


def posterior_handle(colour, band=None):
    """The legend symbol of a posterior: its mean line over its +/- 2 sd band (`posterior_band`)."""
    line = Line2D([], [], color=colour, lw=MEAN_LW)
    if band == "filled":
        return (Patch(facecolor=colour, alpha=BAND_ALPHA), line)
    if band == "outlined":
        return (Patch(facecolor="none", edgecolor=colour, lw=OUTLINE_LW), line)
    if band == "dotted":
        return (Patch(facecolor="none", edgecolor=colour, lw=OUTLINE_LW, ls=DOTTED), line)
    return line


def truth_handle(truth=TRUTH):
    return Line2D([], [], **{**truth, "zorder": None})


def data_handle(alpha=None):
    return Line2D([], [], ls="none", marker="o", ms=2, color=NEUTRAL, alpha=alpha)


# ------------------------------------------------------------ log axes

def _line_data(ax, cap=None):
    """Finite, positive y-data of every line on the axes, optionally only that at or below `cap`."""
    ys = [np.asarray(line.get_ydata(), dtype=float) for line in ax.lines]
    v = np.concatenate(ys) if ys else np.array([1.0])
    v = v[np.isfinite(v) & (v > 0)]
    if cap is not None:
        v = v[v <= cap]
    return v if v.size else np.array([1.0])


def autoscale_log(ax, pad=0.06, cap=None):
    """A log y-range from the axes' lines (at or below `cap`), with a margin, capped."""
    v = _line_data(ax, cap)
    lo, hi = v.min(), v.max()
    margin = (hi / lo) ** pad
    lo, hi = lo / margin, hi * margin
    if cap is not None:
        hi = min(hi, cap)
    ax.set_ylim(lo, hi)


def ensure_log_ticks(ax, n_min=3):
    """At least `n_min` labelled major ticks on a log y-axis (1-2-5 or finer when a panel spans less than two
    decades)."""
    lo, hi = ax.get_ylim()
    if lo <= 0 or hi <= lo:
        return
    e0, e1 = int(np.floor(np.log10(lo))), int(np.ceil(np.log10(hi)))
    decades = [10.0 ** e for e in range(e0, e1 + 1) if lo <= 10.0 ** e <= hi]
    if len(decades) >= n_min:
        return
    for mantissas in ((1, 2, 5), (1, 1.5, 2, 3, 5, 7), (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8)):
        ticks = sorted(m * 10.0 ** e for e in range(e0 - 1, e1 + 1) for m in mantissas if lo <= m * 10.0 ** e <= hi)
        if len(ticks) >= n_min:
            break
    if len(ticks) < n_min:
        ticks = sorted({float(f"{lo * (hi / lo) ** f:.2g}") for f in (0.15, 0.5, 0.85)})
    while len(ticks) > 6 and len(ticks[::2]) >= n_min:
        ticks = ticks[::2]

    def fmt(v, _pos=None):
        if 1e-3 <= v < 1e4:
            return f"{v:g}"
        e = int(np.floor(np.log10(v)))
        m = v / 10.0 ** e
        return rf"$10^{{{e}}}$" if abs(m - 1) < 1e-9 else rf"${m:g}\times10^{{{e}}}$"

    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(fmt))
    ax.yaxis.set_minor_formatter(NullFormatter())
