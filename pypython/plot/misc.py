#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot miscellaneous things."""

from matplotlib import pyplot as plt

from pypython import smooth_array
from pypython.plot import normalize_figure_style
from pypython.plot.spectrum import add_line_ids, common_lines


def plot_spectrum_physics_process_contributions(contribution_spectra,
                                                inclination,
                                                root,
                                                wd=".",
                                                xmin=None,
                                                xmax=None,
                                                ymin=None,
                                                ymax=None,
                                                scale="logy",
                                                line_labels=True,
                                                sm=5,
                                                lw=2,
                                                alpha=0.75,
                                                file_ext="png",
                                                display=False):
    """Description of the function.
    todo: some of these things really need re-naming..... it seems very confusing
    Parameters
    ----------
    Returns
    -------
    fig: plt.Figure
        The plt.Figure object for the created figure
    ax: plt.Axes
        The plt.Axes object for the created figure"""
    normalize_figure_style()

    fig, ax = plt.subplots(figsize=(12, 8))

    for name, spectrum in contribution_spectra.items():
        ax.plot(spectrum["Lambda"], smooth_array(spectrum[inclination], sm), label=name, linewidth=lw, alpha=alpha)

    if scale == "logx" or scale == "loglog":
        ax.set_xscale("log")
    if scale == "logy" or scale == "loglog":
        ax.set_yscale("log")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.legend(loc="upper center", ncol=len(contribution_spectra))
    ax.set_xlabel(r"Wavelength [$\AA$]")
    ax.set_ylabel(r"Flux F$_{\lambda}$ [erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]")

    if line_labels:
        if scale == "logx" or scale == "loglog":
            logx = True
        else:
            logx = False
        ax = add_line_ids(ax, common_lines(), logx=logx)

    fig.tight_layout(rect=[0.015, 0.015, 0.985, 0.985])
    fig.savefig("{}/{}_spec_processes.{}".format(wd, root, file_ext), dpi=300)
    if file_ext != "png":
        fig.savefig("{}/{}_spec_processes.png".format(wd, root), dpi=300)

    if display:
        plt.show()
    else:
        plt.close()

    return fig, ax
