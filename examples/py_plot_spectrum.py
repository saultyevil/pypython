#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script will create various figures for the spectrum generated by a model.
It creates the following figures:
    - A single panel figure with all spectra plotted
    - A multiple panel figure with each spectrum in its own panel
    - Multiple files containing a single panel, done for each spectrum
"""


import argparse as ap
from typing import Tuple
from matplotlib import pyplot as plt

from pypython import spectumutil
from pypython import spectrumplot

plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['axes.labelsize'] = 15

import warnings
warnings.filterwarnings("ignore", module="matplotlib")


def setup_script(
) -> tuple:
    """
    Parse the different modes this script can be run from the command line.

    Returns
    -------
    setup: tuple
        A list containing all of the different setup of parameters for plotting.
    """

    p = ap.ArgumentParser(description=__doc__)

    # Required arguments
    p.add_argument("root",
                   type=str,
                   help="The root name of the simulation.")

    # Supplementary arguments
    p.add_argument("-wd",
                   "--working_directory",
                   default=".",
                   help="The directory containing the simulation.")

    p.add_argument("-xl",
                   "--xmin",
                   type=float,
                   default=None,
                   help="The lower x-axis boundary to display.")

    p.add_argument("-xu",
                   "--xmax",
                   type=float,
                   default=None,
                   help="The upper x-axis boundary to display.")

    p.add_argument("-s",
                   "--scales",
                   default="logy",
                   choices=["logx", "logy", "loglog", "linlin"],
                   help="The axes scaling to use: logx, logy, loglog, linlin.")

    p.add_argument("-l",
                   "--common_lines",
                   action="store_true",
                   default=False,
                   help="Plot labels for important absorption edges.")

    p.add_argument("-f",
                   "--frequency_space",
                   action="store_true",
                   default=False,
                   help="Create the figure in frequency space.")

    p.add_argument("-sm",
                   "--smooth_amount",
                   type=int,
                   default=5,
                   help="The size of the boxcar smoothing filter.")

    p.add_argument("-e",
                   "--ext",
                   default="png",
                   help="The file extension for the output figure.")

    p.add_argument("--display",
                   action="store_true",
                   default=False,
                   help="Display the plot before exiting the script.")

    args = p.parse_args()

    setup = (
        args.root,
        args.working_directory,
        args.xmin,
        args.xmax,
        args.frequency_space,
        args.common_lines,
        args.scales,
        args.smooth_amount,
        args.ext,
        args.display
    )

    return setup


def plot_all_spectrum_inclinations_in_one_panel(
    root: str, wd: str = "./", xmin: float = None, xmax: float = None, smooth_amount: int = 5,
    frequency_space: bool = False, axes_scales: str = "logy", common_lines: bool = True, file_ext: str = "png"
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot all of the spectra for a model on the same panel, for some comparison
    reasons. This is best done with small wavelength ranges.

    Parameters
    ----------
    root: str
        The root name of the model.
    wd: str [optional]
        The directory where the simulation is stored, by default this assumes
        that it is in the calling directory.
    xmin: float [optional]
        The lower x boundary for the figure.
    xmax: float [optional]
        The upper x boundary for the figure.
    smooth_amount: int [optional]
        The size of the boxcar filter to smooth the spectra.
    frequency_space: bool [optional]
        Create the figure in frequency space instead of wavelength space.
    axes_scales: bool [optional]
        Set the scales for the axes in the plot.
    common_lines: bool [optional]
        Plot labels for common line transitions.
    file_ext: str [optional]
        The extension of the final output file.

    Returns
    -------
    fig: plt.Figure
        The matplotlib Figure object for the created plot.
    ax: plt.Axes
        The matplotlib Axes objects for the plot panels.
    """

    alpha = 0.75
    spectrum_filename = "{}/{}.spec".format(wd, root)
    s = spectumutil.read_spectrum(spectrum_filename)
    ia = spectumutil.get_spectrum_inclinations(s)

    fig, ax = plt.subplots(figsize=(12, 8))

    if frequency_space:
        xlabel = "Freq."
        axes_scales = "loglog"
    else:
        xlabel = "Lambda"
    x = s[xlabel].values

    if frequency_space:
        xlabel = r"Frequency [Hz]"
        ylabel = r"$\nu F_{\nu}$ (erg s$^{-1}$ cm$^{-2}$"
    else:
        xlabel = r"Wavelength [$\AA$]"
        ylabel = r"$F_{\lambda}$ (erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$)"

    # Plot each inclination a in ia on the same ax object

    ymin = +1e99
    ymax = -1e99

    xlims = [x.min(), x.max()]
    if not xmin:
        xmin = xlims[0]
    if not xmax:
        xmax = xlims[1]

    for a in ia:
        y = spectumutil.smooth(s[a].values, smooth_amount)

        tmin, tmax = spectumutil.calculate_axis_y_limits(x, y, xmin, xmax)
        if tmin < ymin:
            ymin = tmin
        if tmax > ymax:
            ymax = tmax

        # Convert into lambda F_lambda which is (I hope) the same as nu F_nu
        if frequency_space:
            y *= s["Lambda"].values

        fig, ax = spectrumplot.plot(
            x, y, xmin, xmax, xlabel, ylabel, axes_scales, fig, ax, label=str(a) + r"$^{\circ}$", alpha=alpha
        )

    ax.set_ylim(ymin, ymax)
    ax.legend(loc="lower left")

    if common_lines:
        if axes_scales == "loglog" or axes_scales == "logx":
            logx = True
        else:
            logx = False
        ax = spectumutil.ax_add_line_id(ax, spectumutil.common_lines_list(), logx)

    fig.tight_layout(rect=[0.015, 0.015, 0.985, 0.985])
    fig.savefig("{}/{}_spectra_single.{}".format(wd, root, file_ext))

    return fig, ax


def plot_spectrum_inclinations_on_one_figure_in_subpanels(
    root: str, wd: str = "./", xmin: float = None, xmax: float = None, smooth_amount: int = 5,
    frequency_space: bool = False, axes_scales: str = "logy", common_lines: bool = True, file_ext: str = "png"
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot each separate spectrum in an individual panel, on one figure.

    Parameters
    ----------
    root: str
        The root name of the model.
    wd: str [optional]
        The directory where the simulation is stored, by default this assumes
        that it is in the calling directory.
    xmin: float [optional]
        The lower x boundary for the figure.
    xmax: float [optional]
        The upper x boundary for the figure.
    smooth_amount: int [optional]
        The size of the boxcar filter to smooth the spectra.
    frequency_space: bool [optional]
        Create the figure in frequency space instead of wavelength space.
    axes_scales: bool [optional]
        Set the scales for the axes in the plot.
    common_lines: bool [optional]
        Plot labels for common line transitions.
    file_ext: str [optional]
        The extension of the final output file.

    Returns
    -------
    fig: plt.Figure
        The matplotlib Figure object for the created plot.
    ax: plt.Axes
        The matplotlib Axes objects for the plot panels.
    """

    fig, ax = spectrumplot.plot_spectrum_inclinations_in_subpanels(
        root, wd, xmin, xmax, smooth_amount, common_lines, frequency_space, axes_scales
    )
    fig.savefig("{}/{}_spectra.{}".format(wd, root, file_ext))

    return fig, ax


def plot_spectrum_inclination_in_individual_figures(
    root: str, wd: str = "./", xmin: float = None, xmax: float = None, smooth_amount: int = 5,
    frequency_space: bool = False, axes_scales: str = "logy", file_ext: str = "png"
) -> None:
    """
    Plot each separate spectrum as its own figure.

    Parameters
    ----------
    root: str
        The root name of the model.
    wd: str [optional]
        The directory where the simulation is stored, by default this assumes
        that it is in the calling directory.
    xmin: float [optional]
        The lower x boundary for the figure.
    xmax: float [optional]
        The upper x boundary for the figure.
    smooth_amount: int [optional]
        The size of the boxcar filter to smooth the spectra.
    frequency_space: bool [optional]
        Create the figure in frequency space instead of wavelength space.
    axes_scales: bool [optional]
        Set the scales for the axes in the plot.
    file_ext: str [optional]
        The extension of the final output file.

    Returns
    -------
    fig: plt.Figure
        The matplotlib Figure object for the created plot.
    ax: plt.Axes
        The matplotlib Axes objects for the plot panels.
    """

    alpha = 0.75
    spectrum_filename = "{}/{}.spec".format(wd, root)
    s = spectumutil.read_spectrum(spectrum_filename)
    ia = spectumutil.get_spectrum_inclinations(s)

    if frequency_space:
        xlabel = "Freq."
        axes_scales = "loglog"
    else:
        xlabel = "Lambda"
    x = s[xlabel].values

    if frequency_space:
        xlabel = r"Frequency [Hz]"
        ylabel = r"$\nu F_{\nu}$ (erg s$^{-1}$ cm$^{-2}$"
    else:
        xlabel = r"Wavelength [$\AA$]"
        ylabel = r"$F_{\lambda}$ (erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$)"

    # Plot each inclination a in ia on different fig and ax objects

    for a in ia:
        y = spectumutil.smooth(s[a].values, smooth_amount)
        # Convert into lambda F_lambda which is (I hope) the same as nu F_nu
        if frequency_space:
            y *= s["Lambda"].values

        fig, ax = spectrumplot.plot(x, y, xmin, xmax, xlabel, ylabel, axes_scales, alpha=alpha)
        if axes_scales == "loglog" or axes_scales == "logx":
            logx = True
        else:
            logx = False
        ax = spectumutil.ax_add_line_id(ax, spectumutil.common_lines_list(), logx)
        ax.set_title("Inclination i = {}".format(str(a)) + r"$^{\circ}$")
        fig.tight_layout(rect=[0.015, 0.015, 0.985, 0.985])
        fig.savefig("{}/{}_i{}_spectrum.{}".format(wd, root, str(a), file_ext))

    return


def main(
    setup: tuple = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    The main function of the script. First, the important wind quantaties are
    plotted. This is then followed by the important ions.

`   Parameters
    ----------
    setup: tuple
        A tuple containing the setup parameters to run the script. If this
        isn't provided, then the script will parse them from the command line.

        setup = (
            root,
            wd,
            xmin,
            xmax,
            frequency_space,
            common_lines,
            axes_scales,
            smooth_amount,
            file_ext,
            display
        )

    Returns
    -------
    fig: plt.Figure
        The matplotlib Figure object for the created plot.
    ax: plt.Axes
        The matplotlib Axes objects for the plot panels.
    """

    if setup:
        root, wd, xmin, xmax, frequency_space, common_lines, axes_scales, smooth_amount, file_ext, display = setup
    else:
        root, wd, xmin, xmax, frequency_space, common_lines, axes_scales, smooth_amount, file_ext, display = \
            setup_script()

    root = root.replace("/", "")

    fig, ax = plot_all_spectrum_inclinations_in_one_panel(
        root, wd, xmin, xmax, smooth_amount, frequency_space, axes_scales, common_lines, file_ext
    )

    fig, ax = plot_spectrum_inclinations_on_one_figure_in_subpanels(
        root, wd, xmin, xmax, smooth_amount, frequency_space, axes_scales, False, file_ext
    )

    plot_spectrum_inclination_in_individual_figures(
        root, wd, xmin, xmax, smooth_amount, frequency_space, axes_scales, file_ext
    )

    if display:
        plt.show()

    return fig, ax


if __name__ == "__main__":
    fig, ax = main()
