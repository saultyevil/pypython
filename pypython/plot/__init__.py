#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The basic/universal plotting functions of pypython.

The module includes functions for normalizing the style, as well as ways
to finish the plot. Included in pypython are functions to plot the
spectrum files and the wind save tables.
"""

import numpy as np
from matplotlib import pyplot as plt

from pypython import check_sorted_array_ascending, get_array_index
from pypython.error import InvalidParameter

# Generic plotting function ----------------------------------------------------


def plot(x,
         y,
         xmin=None,
         xmax=None,
         xlabel=None,
         ylabel=None,
         scale="logy",
         fig=None,
         ax=None,
         label=None,
         alpha=1.0,
         display=False):
    """Wrapper function around plotting a simple line graph.

    Parameters
    ----------

    Returns
    -------
    """
    normalize_figure_style()

    # It doesn't make sense to provide only fig and not ax, or ax and not fig
    # so at this point we will throw an error message and return

    if fig and not ax:
        raise InvalidParameter("fig has been provided, but ax has not. Both are required.")
    elif not fig and ax:
        raise InvalidParameter("fig has not been provided, but ax has. Both are required.")
    elif not fig and not ax:
        fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    x, y = get_xy_subset(x, y, xmin, xmax)
    ax.plot(x, y, label=label, alpha=alpha)

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    ax = set_axes_scales(ax, scale)

    if display:
        plt.show()
    else:
        plt.close()

    return fig, ax


# Helper functions -------------------------------------------------------------


def _check_axes_scale_string(scale):
    """Check that the axes scales passed are recognised.

    Parameters
    ----------
    scale: str
        The scaling of the axes to check.
    """

    if scale not in ["logx", "logy", "linlin", "loglog"]:
        raise ValueError(f"{scale} is an unknown axes scale choice, allowed: logx, logy, linlin, loglog")


def finish_figure(fig, title=None, hspace=None, wspace=None):
    """Add finishing touches to a figure.

    This function can be used to add a title or adjust the spacing between
    subplot panels. The subplots will also be given a tight layout.

    Parameters
    ----------
    fig:
        The figure object to update.
    title: str
        The title of the figure. Underscores are automatically modified so
        LaTeX doesn't complain.
    hspace: float
        The amount of vertical space between subplots.
    wspace: float
        The amount of horizontal space between subplots.
    """

    if title:
        fig.suptitle(title.replace("_", r"\_"))

    fig.tight_layout(rect=[0.015, 0.015, 0.985, 0.985])

    if hspace:
        fig.subplots_adjust(hspace=hspace)
    if wspace:
        fig.subplots_adjust(wspace=wspace)

    return fig


def get_xy_subset(x, y, xmin, xmax):
    """Get a subset of values from two array given xmin and xmax.

    The array must be sorted in ascending or descending order.

    Parameters
    ----------
    x: np.ndarray
        The first array to get the subset from, set by xmin and xmax.
    y: np.ndarray
        The second array to get the subset from.
    xmin: float
        The minimum x value
    xmax: float
        The maximum x value

    Returns
    -------
    x, y: np.ndarray
        The subset arrays.
    """
    assert len(x) == len(y)

    # The array has to be indexed differently depending on if it is ascending
    # or descending

    if check_sorted_array_ascending(x):
        if xmin:
            idx = get_array_index(x, xmin)
            x = x[idx:]
            y = y[idx:]
        if xmax:
            idx = get_array_index(x, xmax)
            x = x[:idx]
            y = y[:idx]
    else:
        if xmin:
            idx = get_array_index(x, xmin)
            x = x[:idx]
            y = y[:idx]
        if xmax:
            idx = get_array_index(x, xmax)
            x = x[idx:]
            y = y[idx:]

    return x, y


def normalize_figure_style():
    """Set default pypython matplotlib parameters."""

    parameters = {
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}",
        "font.serif": "cm",
        "font.size": 18,
        "legend.fontsize": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 16,
        "axes.linewidth": 2,
        "lines.linewidth": 2.2,
        "xtick.bottom": True,
        "xtick.minor.visible": True,
        "xtick.direction": "out",
        "xtick.major.width": 1.5,
        "xtick.minor.width": 1,
        "xtick.major.size": 4,
        "xtick.minor.size": 3,
        "ytick.left": True,
        "ytick.minor.visible": True,
        "ytick.direction": "out",
        "ytick.major.width": 1.5,
        "ytick.minor.width": 1,
        "ytick.major.size": 4,
        "ytick.minor.size": 3,
        "savefig.dpi": 300,
        "pcolor.shading": "auto"
    }

    plt.rcParams.update(parameters)

    return parameters


def remove_extra_axes(fig, ax, n_wanted, n_panel):
    """Remove additional axes which are included in a plot.

    This should be used if you have 4 x 2 = 8 panels but only want to use 7 of
    the panels, in this case the 8th panel will be removed.

    Parameters
    ----------
    fig: plt.Figure
        The Figure object to modify.
    ax: np.ndarray of plt.Axes
        The Axes objects to modify.
    n_wanted: int
        The actual number of plots/panels which are wanted.
    n_panel: int
        The number of panels which are currently in the Figure and Axes objects.

    Returns
    -------
    fig: plt.Figure
        The modified Figure.
    ax: plt.Axes
        The modified Axes.
    """

    if type(ax) != np.ndarray:
        return fig, ax
    elif len(ax) == 1:
        return fig, ax

    # Flatten the axes array to make life easier with indexing

    shape = ax.shape
    ax = ax.flatten()

    if n_panel > n_wanted:
        for i in range(n_wanted, n_panel):
            fig.delaxes(ax[i])

    # Return ax to the shape it was passed as

    ax = np.reshape(ax, (shape[0], shape[1]))

    return fig, ax


def set_axes_scales(ax, scale):
    """Set the scale for axes.

    Parameters
    ----------
    ax: plt.Axes
        The matplotlib Axes to update.
    scale: str
        The axes scaling to use.
    """
    _check_axes_scale_string(scale)

    if scale == "logx" or scale == "loglog":
        ax.set_xscale("log")
    if scale == "logy" or scale == "loglog":
        ax.set_yscale("log")

    return ax


def subplot_dims(n_plots):
    """Get the number of rows and columns for the give number of plots.

    Returns how many rows and columns should be used to have the correct
    number of figures available. This doesn't return anything larger than
    3 columns, but the number of rows can be large.

    Parameters
    ----------
    n_plots: int
        The number of subplots which will be plotted

    Returns
    -------
    dims: Tuple[int, int]
        The dimensions of the subplots returned as (nrows, ncols)
    """
    if n_plots > 9:
        n_cols = 3
        n_rows = (1 + n_plots) // n_cols
    elif n_plots < 2:
        n_rows = n_cols = 1
    else:
        n_cols = 2
        n_rows = (1 + n_plots) // n_cols

    return n_rows, n_cols


# This is placed here due to a circular dependency -----------------------------

from pypython.plot import misc, spectrum, wind

set_style = set_figure_style = normalize_figure_style
