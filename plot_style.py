"""Shared publication-quality Matplotlib styling for report figures."""

from pathlib import Path
import os

import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.figure import Figure

_LATEX_SUPPORT = Path(__file__).resolve().parent / "latex-support"
os.environ["TEXINPUTS"] = f"{_LATEX_SUPPORT}{os.pathsep}" + os.environ.get("TEXINPUTS", "")
from matplotlib.ticker import MaxNLocator

FIGURE_SIZE = (6.4, 4.8)
SQUARE_FIGURE_SIZE = (5.8, 5.8)
LINE_WIDTH = 2.4
SAVE_DPI = 300


def configure_plots() -> None:
    """Configure Matplotlib to typeset all figure text with LaTeX."""
    mpl.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 20,
        "axes.labelsize": 22,
        "axes.linewidth": 1.6,
        "xtick.labelsize": 19,
        "ytick.labelsize": 19,
        "xtick.major.size": 7,
        "ytick.major.size": 7,
        "xtick.major.width": 1.6,
        "ytick.major.width": 1.6,
        "legend.fontsize": 18,
        "lines.linewidth": LINE_WIDTH,
        "savefig.dpi": SAVE_DPI,
        "savefig.bbox": "tight",
        "axes.grid": False,
    })


def format_axes(ax: Axes, *, equal: bool = False) -> None:
    """Use sparse major ticks and prominent axes suitable for a LaTeX report."""
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.tick_params(which="major", direction="out", width=1.6, length=7)
    for spine in ax.spines.values():
        spine.set_linewidth(1.6)
    if equal:
        ax.set_aspect("equal", adjustable="box")


def save_figure(fig: Figure, destination: Path) -> None:
    """Save a tightly cropped, high-resolution PNG."""
    fig.savefig(destination, dpi=SAVE_DPI, bbox_inches="tight", facecolor="white")
