"""
phenogame.viz — Visualization for fit results.
"""

import numpy as np
import matplotlib.pyplot as plt

from .fit import FitReport
from .response import RESPONSE_FUNCTIONS


def plot_fit(report: FitReport, top_n: int = 4, figsize: tuple = (10, 6)):
    """Plot observed data with top-N fitted response functions.

    Parameters
    ----------
    report : FitReport
        Output of fit_all().
    top_n : int
        Number of top-ranked fits to overlay.
    figsize : tuple
        Figure size.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Data points
    ax.scatter(report.x_data, report.y_data, c="#222", s=24, zorder=5,
               label="Observed", edgecolors="#555", linewidths=0.5)

    # Smooth x range for fitted curves
    x_fine = np.linspace(report.x_data.min(), report.x_data.max(), 300)

    colors = ["#C87941", "#5B8FA8", "#5E8B6A", "#8B6E8F", "#A68A64", "#B85042"]
    for i, result in enumerate(report.results[:top_n]):
        rf = RESPONSE_FUNCTIONS[result.func_id]
        y_fine = rf.fn(x_fine, **result.params)
        color = colors[i % len(colors)]
        label = f"{result.func_name}  (R²={result.r_squared:.4f}, K≤{result.eml_K})"
        linewidth = 2.5 if i == 0 else 1.2
        alpha = 1.0 if i == 0 else 0.6
        ax.plot(x_fine, y_fine, color=color, linewidth=linewidth,
                alpha=alpha, label=label)

    ax.set_xlabel(RESPONSE_FUNCTIONS[report.best.func_id].x_label, fontsize=11)
    ax.set_ylabel(RESPONSE_FUNCTIONS[report.best.func_id].y_label, fontsize=11)
    ax.set_title(f"PhenoGame: Best = {report.best.func_name}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    return fig
