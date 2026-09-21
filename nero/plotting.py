"""One figure style for everything this project plots.

Used by ``scripts/figures.py`` (the report figures) and by
``scripts/online_learning.py`` (the drift curves it writes as it runs), so the two agree
on typeface, palette and weights.

Text is typeset by LaTeX when a working ``latex`` is on PATH, which matches the
report exactly; otherwise it falls back to matplotlib's STIX fonts, which are
Times-metric-compatible and look the same at a glance. Nothing here needs LaTeX
to be installed.

``regen_paper_figs.py`` is deliberately left alone -- it carries the existing
two-column paper style, and its outputs are committed artefacts.
"""

import shutil

import matplotlib
import matplotlib.pyplot as plt

# Okabe-Ito, checked with the dataviz palette validator: adjacent-pair CVD
# separation >= 11 dE, normal-vision floor >= 19 dE, contrast >= 3:1.
BLUE, VERM, GREEN, AMBER = "#0072B2", "#D55E00", "#009E73", "#E69F00"
INK, INK_SOFT, RULE, GRID = "#1A1A1A", "#5A5A5A", "#9A9A9A", "#D8D8D8"

#: Native figure widths in inches. Figures are included at exactly these widths,
#: so nothing is rescaled and every plot carries the same text size.
W = 4.60          # rectangular figures
W_SLOPE = 4.00    # the slope chart, which needs less width
W_SQUARE = 3.50   # the equal-aspect scatter

HATCH = "////"


def latex_available():
    return shutil.which("latex") is not None and shutil.which("dvipng") is not None


def apply(latex=None):
    """Install the style. ``latex=None`` uses LaTeX only if it is available."""
    use_latex = latex_available() if latex is None else latex
    matplotlib.use("Agg")
    plt.rcParams.update({
        "text.usetex": use_latex,
        "font.family": "serif",
        "font.size": 8.0,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
        "legend.frameon": False,
        "legend.handlelength": 1.5,
        "legend.handletextpad": 0.5,
        "legend.borderaxespad": 0.3,
        "axes.linewidth": 0.5,
        "axes.edgecolor": RULE,
        "axes.labelcolor": INK,
        "axes.labelpad": 3.0,
        "text.color": INK,
        "xtick.color": RULE,
        "ytick.color": RULE,
        "xtick.labelcolor": INK_SOFT,
        "ytick.labelcolor": INK_SOFT,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.4,
        "grid.alpha": 1.0,
        "lines.solid_capstyle": "round",
        "figure.dpi": 110,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.01,
        "pdf.fonttype": 42,
    })
    if use_latex:
        plt.rcParams.update({
            "text.latex.preamble": r"\usepackage{times}\usepackage{amsmath}",
            "font.serif": ["Times"],
        })
    else:
        plt.rcParams.update({
            "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
        })
    return use_latex


def value_axis_only(ax, axis="x"):
    """Grid on the measured axis only; the category axis stays clean."""
    ax.grid(axis="y" if axis == "x" else "x", visible=False)
    ax.set_axisbelow(True)
