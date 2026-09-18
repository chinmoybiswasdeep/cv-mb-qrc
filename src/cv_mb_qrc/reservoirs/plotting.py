"""Central publication style shared by all generated scientific figures."""

from itertools import cycle

import matplotlib.pyplot as plt

OKABE_ITO = ("#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000")
MARKERS = ("o", "s", "^", "D", "P", "X", "v", "*")


def apply_publication_style():
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 7,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
        }
    )


def method_styles(methods):
    colors, markers = cycle(OKABE_ITO), cycle(MARKERS)
    styles = {}
    for method in sorted(set(methods)):
        simulator = method.startswith(("cv_", "graphix", "gaussian_classical_twin"))
        styles[method] = {
            "color": next(colors),
            "marker": next(markers),
            "linestyle": "-" if simulator else "--",
            "markerfacecolor": "none" if simulator else None,
        }
    return styles


def panel_label(axis, label):
    axis.text(-0.14, 1.05, label, transform=axis.transAxes, fontweight="bold", va="top")
