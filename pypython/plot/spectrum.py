#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt

from pypython import (SPECTRUM_UNITS_FLM, SPECTRUM_UNITS_FNU, SPECTRUM_UNITS_LNU, Spectrum, get_root_name)
from pypython.constants import ANGSTROM, C
from pypython.plot import (finish_figure, get_xy_subset, normalize_figure_style, remove_extra_axes, set_axes_scales,
                           subplot_dims)

MIN_SPEC_COMP_FLUX = 1e-15

# Helper functions -------------------------------------------------------------


def _add_line_labels(ax, labels, scale, linestyle="dashed", offset=0.0):
    """Add labels of lines or absorption edges to ax.

    Parameters
    ----------
    ax: plt.Axes
        The subplot to add labels to.
    labels: tuple or list
        The labels to add.
    scale: str
        The scaling of the axes.
    """
    if scale == "loglog" or scale == "logx":
        logx = True
    else:
        logx = False
    ax = add_line_ids(ax, labels, linestyle=linestyle, logx=logx, offset=offset)

    return ax


def _convert_labels_to_frequency_space(lines, freq=False, spectrum=None):
    """Convert the given list of lines/edges from Angstrom to Hz.

    Parameters
    ----------
    lines: List[str, float]
        The list of labels to convert from wavelength to frequency space.
    freq: bool
        The flag to indicate to convert to frequency space
    spectrum: pypython.Spectrum
        A spectrum object, used to find the units of the spectrum.
    """

    if spectrum:
        if spectrum.units in [SPECTRUM_UNITS_FNU, SPECTRUM_UNITS_LNU]:
            for i in range(len(lines)):
                lines[i][1] = C / (lines[i][1] * ANGSTROM)
    elif freq:
        for i in range(len(lines)):
            lines[i][1] = C / (lines[i][1] * ANGSTROM)

    return lines


def _get_inclinations(spectrum, inclinations):
    """Get the inclination angles which will be used.

    If all is passed, then the inclinations from the spectrum object will
    be used. Otherwise, this will simply create a list of strings of the
    inclinations.

    Parameters
    ----------
    spectrum: pypython.Spectrum
        The spectrum object.
    inclinations: list
        A list of inclination angles wanted to be plotted.
    """
    if type(inclinations) != list:
        inclinations = str(inclinations)

    if type(inclinations) == str:
        inclinations = [inclinations]

    if len(inclinations) > 1:
        if inclinations[0] == "all":  # ignore "all" if other inclinations are passed
            inclinations = inclinations[1:]
    else:
        if inclinations[0] == "all":
            inclinations = spectrum.inclinations

    return [str(inclination) for inclination in inclinations]


def _plot_subplot(ax, spectrum, things_to_plot, xmin, xmax, alpha, scale, use_flux, skip_sparse):
    """Plot some things to a provided matplotlib ax object.

    This function is used to do a lot of the plotting heavy lifting in this
    sub-module. It's still fairly flexible to be used outside of the main
    plotting functions, however. You are just required to pass an axes to
    plot onto.

    Parameters
    ----------
    ax: plt.Axes
        A matplotlib axes object to plot onto.
    spectrum: pypython.Spectrum
        The spectrum object to plot. The current spectrum wishing to be set
        must be correct, otherwise the wrong thing may be plotted.
    things_to_plot: str or list or tuple of str
        A collection of names of things to plot to iterate over.
    xmin: float
        The lower x boundary of the plot.
    xmax: float
        The upper x boundary of the plot.
    alpha: float
        The transparency of the spectra plotted.
    scale: str
        The scaling of the plot axes.
    use_flux: bool
        Plot the spectrum as a flux or nu Lnu instead of flux density or
        luminosity.
    skip_sparse: bool
        Enable skipping of spectra which are very sparse, thus noisy.

    Returns
    -------
    ax: pyplot.Axes
        The modified matplotlib Axes object.
    """
    if type(things_to_plot) is str:
        things_to_plot = things_to_plot,

    n_skipped = 0

    xlims = []
    ylims = []

    for thing in things_to_plot:

        y = spectrum[thing]
        if skip_sparse and len(y[y < MIN_SPEC_COMP_FLUX]) > 0.7 * len(y):
            n_skipped += 1
            continue

        # If plotting in frequency space, of if the units then the flux needs
        # to be converted in nu F nu

        if use_flux:
            if spectrum.units == SPECTRUM_UNITS_FLM:
                y *= spectrum["Lambda"]
            else:
                y *= spectrum["Freq."]

        if spectrum.units == SPECTRUM_UNITS_FLM:
            x = spectrum["Lambda"]
        else:
            x = spectrum["Freq."]

        x, y = get_xy_subset(x, y, xmin, xmax)

        ax.plot(x, y, label=thing, alpha=alpha)

        xlims.append([x.min(), x.max()])
        ylims.append([y.min(), y.max()])

    if n_skipped == len(things_to_plot):
        print("Nothing was plotted due to all the spectra being too sparse")
        return ax

    ax.legend(loc="lower left")
    ax = set_axes_scales(ax, scale)
    ax = set_spectrum_axes_labels(ax, spectrum, use_flux)

    return ax


def add_line_ids(ax,
                 lines,
                 linestyle="dashed",
                 ynorm=0.90,
                 logx=False,
                 offset=25,
                 rotation="vertical",
                 fontsize=15,
                 whitespace_scale=2):
    """Add labels for line transitions or other regions of interest onto a
    matplotlib figure. Labels are placed at the top of the panel and dashed
    lines, with zorder = 0, are drawn from top to bottom.

    Parameters
    ----------
    ax: plt.Axes
        The axes (plt.Axes) to add line labels too
    lines: list
        A list containing the line name and wavelength in Angstroms
        (ordered by wavelength)
    linestyle: str [optional]
        The type of line to draw to show where the transitions are. Allowed
        values [none, dashed, top]
    ynorm: float [optional]
        The normalized y coordinate to place the label.
    logx: bool [optional]
        Use when the x-axis is logarithmic
    offset: float [optional]
        The amount to offset line labels along the x-axis
    rotation: str [optional]
        Vertical or horizontal rotation for text ids
    fontsize: int [optional]
        The fontsize of the labels
    whitespace_scale: float [optional]
        The amount to scale the upper y limit of the plot by, to add whitespace
        for the line labels.

    Returns
    -------
    ax: plt.Axes
        The updated axes object.
    """
    nlines = len(lines)
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()

    for i in range(nlines):
        x = lines[i][1]
        if x < xlims[0]:
            continue
        if x > xlims[1]:
            continue
        label = lines[i][0]
        if linestyle == "dashed":
            ax.axvline(x, linestyle="--", linewidth=0.5, color="k", zorder=1)
        if linestyle == "thick":
            ax.axvline(x, linestyle="-", linewidth=2, color="k", zorder=1)
        elif linestyle == "top":
            pass  # todo: implement
        x = x - offset

        # Calculate the x location of the label in axes coordinates

        if logx:
            xnorm = (np.log10(x) - np.log10(xlims[0])) / (np.log10(xlims[1]) - np.log10(xlims[0]))
        else:
            xnorm = (x - xlims[0]) / (xlims[1] - xlims[0])

        ax.text(xnorm,
                ynorm,
                label,
                ha="center",
                va="center",
                rotation=rotation,
                fontsize=fontsize,
                transform=ax.transAxes)

    ax.set_ylim(ylims[0], ylims[1] * whitespace_scale)

    return ax


def common_lines(freq=False, spectrum=None):
    """Return a list containing the names of line transitions and the
    wavelength of the transition in Angstroms. Instead of returning the
    wavelength, the frequency can be returned instead. It is also possible to
    return in log space.

    Parameters
    ----------
    freq: bool [optional]
        Label the transitions in frequency space
    spectrum: pypython.Spectrum
        The spectrum object. Used to get the units.

    Returns
    -------
    line: List[List[str, float]]
        A list of lists where each element of the list is the name of the
        transition/edge and the rest wavelength of that transition in
        Angstroms.
    """

    lines = [
        [r"N \textsc{iii} / O \textsc{iii}", 305],
        [r"P \textsc{v}", 1118],
        [r"Ly$\alpha$ / N \textsc{v}", 1216],
        ["", 1242],
        [r"O \textsc{v} / Si \textsc{iv}", 1371],
        ["", 1400],
        [r"N \textsc{iv}", 1489],
        [r"C \textsc{iv}", 1548],
        ["", 1550],
        [r"He \textsc{ii}", 1640],
        [r"N \textsc{iii]}", 1750],
        [r"Al \textsc{iii}", 1854],
        [r"C \textsc{iii]}", 1908],
        [r"Mg \textsc{ii}", 2798],
        [r"Ca \textsc{ii}", 3934],
        ["", 3969],
        [r"H$_{\delta}$", 4101],
        [r"H$_{\gamma}$", 4340],
        [r"He \textsc{ii}", 4389],
        [r"He \textsc{ii}", 4686],
        [r"H$_{\beta}$", 4861],
        [r"Na \textsc{i}", 5891],
        ["", 5897],
        [r"H$_{\alpha}$", 6564],
    ]

    lines = _convert_labels_to_frequency_space(lines, freq, spectrum)

    return lines


def photoionization_edges(freq=False, spectrum=False):
    """Return a list containing the names of line transitions and the
    wavelength of the transition in Angstroms. Instead of returning the
    wavelength, the frequency can be returned instead. It is also possible to
    return in log space.

    Parameters
    ----------
    freq: bool [optional]
        Label the transitions in frequency space
    spectrum: pypython.Spectrum
        The spectrum object. Used to get the units.

    Returns
    -------
    edges: List[List[str, float]]
        A list of lists where each element of the list is the name of the
        transition/edge and the rest wavelength of that transition in
        Angstroms.
    """

    edges = [
        [r"He \textsc{ii}", 229],
        [r"He \textsc{i}", 504],
        ["Lyman", 912],
        ["Balmer", 3646],
        ["Paschen", 8204],
    ]

    edges = _convert_labels_to_frequency_space(edges, freq, spectrum)

    return edges


def set_spectrum_axes_labels(ax, spectrum, use_flux):
    """Set the units of a given matplotlib axes.

    Parameters
    ----------
    ax: plt.Axes
        The axes object to update.
    spectrum: pypython.Spectrum
        The spectrum being plotted. Used to determine the axes labels.
    use_flux: bool
        If flux/nu Lnu is being plotted instead of flux density or
        luminosity.

    Returns
    -------
    ax: plt.Axes
        The updated axes object.
    """
    if use_flux:
        if spectrum.units == SPECTRUM_UNITS_LNU:
            ax.set_xlabel(r"Rest-frame Frequency [Hz]")
            ax.set_ylabel(r"$\nu L_{\nu}$ [erg s$^{-1}$]")
        elif spectrum.units == SPECTRUM_UNITS_FLM:
            ax.set_xlabel(r"Rest-frame Wavelength [\AA]")
            ax.set_ylabel(r"$\lambda F_{\lambda}$ at " + f"{spectrum.distance } pc " + r"[erg s$^{-1}$]")
        else:
            ax.set_xlabel(r"Rest-frame Frequency [Hz]")
            ax.set_ylabel(r"$\nu F_{\nu}$ at " + f"{spectrum.distance } pc " + r"[erg s$^{-1}$ cm$^{-2}$]")
    else:
        if spectrum.units == SPECTRUM_UNITS_LNU:
            ax.set_xlabel(r"Rest-frame Frequency [Hz]")
            ax.set_ylabel(r"$L_{\nu}$ [erg s$^{-1}$ Hz$^{-1}$]")
        elif spectrum.units == SPECTRUM_UNITS_FLM:
            ax.set_xlabel(r"Rest-frame Wavelength [\AA]")
            ax.set_ylabel(r"$F_{\lambda}$ at " + f"{spectrum.distance } pc " + r"[erg s$^{-1}$ cm$^{-2}$ \AA$^{-1}$]")
        else:
            ax.set_xlabel(r"Rest-frame Frequency [Hz]")
            ax.set_ylabel(r"$F_{\nu}$ at " + f"{spectrum.distance } pc " + r"[erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$]")

    return ax


# Plotting functions -----------------------------------------------------------


def optical_depth(spectrum,
                  inclinations="all",
                  xmin=None,
                  xmax=None,
                  scale="loglog",
                  label_edges=True,
                  frequency_space=True,
                  display=False):
    """Plot the continuum optical depth spectrum.

    Create a plot of the continuum optical depth against either frequency or
    wavelength. Frequency space is the default and preferred. This function
    returns the Figure and Axes object.

    Parameters
    ----------
    spectrum: pypython.Spectrum
        The spectrum object.
    inclinations: str or list or tuple [optional]
        A list of inclination angles to plot, but all is also an acceptable
        choice if all inclinations are to be plotted.
    xmin: float [optional]
        The lower x boundary for the figure
    xmax: float [optional]
        The upper x boundary for the figure
    scale: str [optional]
        The scale of the axes for the plot.
    label_edges: bool [optional]
        Label common absorption edges of interest onto the figure
    frequency_space: bool [optional]
        Create the figure in frequency space instead of wavelength space
    display: bool [optional]
        Display the final plot if True.

    Returns
    -------
    fig: plt.Figure
        matplotlib Figure object.
    ax: plt.Axes
        matplotlib Axes object.
    """
    if spectrum.current != "spec_tau":
        spectrum.set("spec_tau")

    normalize_figure_style()
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    # Determine if we're plotting in frequency or wavelength space and then
    # determine the inclinations we want to plot

    if frequency_space:
        xlabel = "Freq."
    else:
        xlabel = "Lambda"

    if not xmin:
        xmin = np.min(spectrum[xlabel])
    if not xmax:
        xmax = np.max(spectrum[xlabel])

    inclinations = _get_inclinations(spectrum, inclinations)

    # Now loop over the inclinations and plot each one, skipping ones which are
    # completely empty

    for inclination in inclinations:
        if inclination != "all":
            if inclination not in spectrum.inclinations:  # Skip inclinations which don't exist
                continue
        label = f"{inclination}" + r"$^{\circ}$"

        x, y = get_xy_subset(spectrum[xlabel], spectrum[inclination], xmin, xmax)

        if np.count_nonzero(y) == 0:  # skip arrays which are all zeros
            continue

        ax.plot(x, y, linewidth=2, label=label)

    ax = set_axes_scales(ax, scale)
    ax.set_ylabel(r"Continuum Optical Depth")

    if frequency_space:
        ax.set_xlabel(r"Rest-frame Frequency [Hz]")
    else:
        ax.set_xlabel(r"Rest-frame Wavelength [$\AA$]")

    ax.legend(loc="upper left")

    if label_edges:
        ax = _add_line_labels(ax, photoionization_edges(frequency_space), scale)

    fig = finish_figure(fig)
    fig.savefig(f"{spectrum.fp}/{spectrum.root}_spec_optical_depth.png")

    if display:
        plt.show()

    return fig, ax


def reprocessing(spectrum, xmin=None, xmax=None, scale="loglog", label_edges=True, alpha=0.75, display=False):
    """Create a plot to show the amount of reprocessing in the model.

    Parameters
    ----------

    Returns
    -------
    fig: plt.Figure
        matplotlib Figure object.
    ax: plt.Axes
        matplotlib Axes object.
    """
    if "spec_tau" not in spectrum.avail_spectrum:
        raise ValueError("There is no spec_tau spectrum so cannot create this plot")
    if "spec" not in spectrum.avail_spectrum and "log_spec" not in spectrum.avail_spectrum:
        raise ValueError("There is no observer spectrum so cannot create this plot")

    normalize_figure_style()
    fig, ax = plt.subplots(figsize=(12, 7))
    ax2 = ax.twinx()

    ax = set_axes_scales(ax, scale)
    ax2 = set_axes_scales(ax2, scale)

    # Plot the optical depth

    spectrum.set("spec_tau")

    for n, inclination in enumerate(spectrum.inclinations):
        y = spectrum[inclination]
        if np.count_nonzero == 0:
            continue
        ax.plot(spectrum["Freq."],
                y,
                color=f"C{n + 2}",
                label=r"$\tau_{" + f"{inclination}" + r"^{\circ}}$",
                alpha=alpha)

    ax.legend(loc="upper left")
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel("Rest-frame Frequency [Hz]")
    ax.set_ylabel("Continuum Optical Depth")

    if label_edges:
        ax = _add_line_labels(ax, photoionization_edges(freq=True), scale, linestyle="none")

    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)

    # Plot the emitted and created spectrum

    spectrum.set("spec")

    for thing in ["Created", "Emitted"]:
        x, y = get_xy_subset(spectrum["Freq."], spectrum[thing], xmin, xmax)

        if spectrum.spatial_units == SPECTRUM_UNITS_FLM:
            y *= spectrum["Lambda"]
        else:
            y *= spectrum["Freq."]

        ax2.plot(x, y, label=thing, alpha=alpha)

    ax2.legend(loc="upper right")
    ax2.set_ylabel(f"Flux {spectrum.distance} pc " + r"[erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]")

    fig.tight_layout(rect=[0.015, 0.015, 0.985, 0.985])
    fig.savefig(f"{spectrum.fp}/{spectrum.root}_reprocessing.png")

    if display:
        plt.show()

    return fig, ax


def components(spectrum, xmin=None, xmax=None, scale="loglog", alpha=0.65, use_flux=False, display=False):
    """Plot the different components of the spectrum.

    The components are the columns labelled with words, rather than inclination
    angles. These columns are not supposed to add together, i.e. all of them
    together shouldn't equal the Emitted spectrum, which is the angle
    averaged escaping flux/luminosity.

    Parameters
    ----------
    spectrum: pypython.Spectrum
        The spectrum object.
    xmin: float [optional]
        The minimum x boundary to plot.
    xmax: float [optional]
        The maximum x boundary to plot.
    scale: str [optional]
        The scaling of the axes.
    alpha: float [optional]
        The line transparency on the plot.
    use_flux: bool [optional]
        Plot in flux or nu Lnu instead of flux density or luminosity.
    display: bool [optional]
        Display the object once plotted.

    Returns
    -------
    fig: plt.Figure
        matplotlib Figure object.
    ax: plt.Axes
        matplotlib Axes object.
    """
    normalize_figure_style()

    fig, ax = plt.subplots(2, 1, figsize=(12, 10))

    ax[0] = _plot_subplot(ax[0], spectrum, ["CenSrc", "Disk", "WCreated"], xmin, xmax, alpha, scale, use_flux, True)
    ax[1] = _plot_subplot(ax[1], spectrum, ["Created", "Emitted", "Wind", "HitSurf"], xmin, xmax, alpha, scale,
                          use_flux, True)

    fig.tight_layout(rect=[0.015, 0.015, 0.985, 0.985])
    fig.savefig(f"{spectrum.fp}/{spectrum.root}_{spectrum.current}_components.png")

    if display:
        plt.show()

    return fig, ax


def observer(spectrum,
             inclinations,
             xmin=None,
             xmax=None,
             scale="logy",
             use_flux=False,
             label_lines=True,
             display=False):
    """Plot the request observer spectrum.

    If all is passed to inclinations, then all the observer angles will be
    plotted on a single figure. This function will only be available if there
    is a .spec or .log_spec file available in the passed spectrum, it does not
    work for spec_tot, etc. For these spectra, plot.spectrum.spectrum_components
    should be used instead, or plot.plot for something finer tuned.

    Parameters
    ----------
    spectrum: pypython.Spectrum
        The spectrum being plotted, "spec" or "log_spec" should be the active
        spectrum.
    inclinations: list or str
        The inclination angles to plot.
    xmin: float [optional]
        The lower x boundary of the plot.
    xmax: float [optional]
        The upper x boundary of the plot.
    scale: str [optional]
        The scale of the axes.
    use_flux: bool [optional]
        Plot the flux instead of flux density.
    label_lines: bool [optional]
        Label common spectrum lines.
    display: bool [optional]
        Display the figure once plotted.

    Returns
    -------
    fig: plt.Figure
        matplotlib Figure object.
    ax: plt.Axes
        matplotlib Axes object.
    """

    if spectrum.current not in ["spec", "log_spec"]:
        spectrum.set("spec")

    normalize_figure_style()
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    inclinations = _get_inclinations(spectrum, inclinations)

    wrong_input = []
    for inclination in inclinations:
        if inclination not in spectrum.inclinations:
            wrong_input.append(inclination)

    if len(wrong_input) > 0:
        print(f"The following inclinations provided are not in the spectrum inclinations and will be skipped:"
              f" {', '.join(wrong_input)}")
        for to_remove in wrong_input:
            inclinations.remove(to_remove)

    if len(inclinations) == 0:
        print(f"Returning empty figure without creating plot as there is nothing to plot")
        return fig, ax

    ax = _plot_subplot(ax, spectrum, inclinations, xmin, xmax, 1.0, scale, use_flux, False)

    if label_lines:
        ax = _add_line_labels(ax, common_lines(spectrum=spectrum), scale, linestyle="none")

    fig.tight_layout(rect=[0.015, 0.015, 0.985, 0.985])
    fig.savefig(f"{spectrum.fp}/{spectrum.root}_{spectrum.current}.png")

    if display:
        plt.show()

    return fig, ax


def multiple_models(output_name,
                    spectra,
                    spectrum_type,
                    things_to_plot,
                    xmin=None,
                    xmax=None,
                    use_flux=False,
                    alpha=0.7,
                    scale="loglog",
                    label_lines=True,
                    log_spec=False,
                    smooth=None,
                    distance=None,
                    display=False):
    """Plot multiple spectra, from multiple models, given in the list of
    spectra provided.

    Spectrum file paths are passed and then each spectrum is loaded in as a
    Spectrum object. Each spectrum must have the same units and are also assumed
    to be at the same distance.

    In this function, it is possible to compare Emitted or Created spectra. It
    is agnostic to the type of spectrum file being plotted, unlike
    spectrum_observer.

    todo: label absorption edges if spec_tau is selected

    Parameters
    ----------
    output_name: str
        The name of the output .png file.
    spectra: str
        The file paths of the spectra to plot.
    spectrum_type: str
        The type of spectrum to plot, i.e. spec or spec_tot.
    things_to_plot: str or list of str or tuple of str
        The things which will be plotted, i.e. '45' or ['Created', '45', '60']
    xmin: float [optional]
        The lower x boundary of the plot
    xmax: float [optional]
        The upper x boundary for the plot
    use_flux: bool [optional]
        Plot in flux units, instead of flux density.
    alpha: float [optional]
        The transparency of the plotted spectra.
    scale: str [optional]
        The scaling of the axes.
    label_lines: bool [optional]
        Label common emission and absorption features, will not work with
        spec_tau.
    log_spec: bool [optional]
        Use either the linear or logarithmically spaced spectra.
    smooth: int [optional]
        The amount of smoothing to apply to the spectra.
    distance: float [optional]
        The distance to scale the spectra to in parsecs.
    display: bool [optional]
        Display the figure after plotting, or don't.

    Returns
    -------
    fig: plt.Figure
        matplotlib Figure object.
    ax: plt.Axes
        matplotlib Axes object.
    """
    normalize_figure_style()

    if type(spectra) is str:
        spectra = list(spectra)

    if len(spectra) == 0:
        raise ValueError("An empty argument was passed for spectra")

    spectra_to_plot = []

    for spectrum in spectra:
        if type(spectrum) is not Spectrum:
            root, fp = get_root_name(spectrum)
            spectra_to_plot.append(Spectrum(root, fp, spectrum_type, log_spec, smooth, distance))
        else:
            spectra_to_plot.append(spectrum)

    units = list(dict.fromkeys([spectrum.units for spectrum in spectra_to_plot]))

    if len(units) > 1:
        msg = ""
        for spectrum in spectra_to_plot:
            msg += f"{spectrum.units} : {spectrum.fp}{spectrum.root}.{spectrum.current}\n"
        raise ValueError(f"Some of the spectra have different units, unable to plot:\n{msg}")

    # Now, this is some convoluted code to get the inclination angles
    # get only the unique, sorted, values if inclination == "all"

    if things_to_plot == "all":
        things_to_plot = ()
        for spectrum in spectra_to_plot:  # have to do it like this, as spectrum.inclinations is a tuple
            things_to_plot += spectrum.inclinations
        if len(things_to_plot) == 0:
            raise ValueError("\"all\" does not seem to have worked, try specifying what to plot instead")
        things_to_plot = tuple(sorted(list(dict.fromkeys(things_to_plot))))  # Gets sorted, unique values from tuple
    else:
        things_to_plot = things_to_plot.split(",")
        things_to_plot = tuple(things_to_plot)

    n_to_plot = len(things_to_plot)
    n_rows, n_cols = subplot_dims(n_to_plot)

    if n_rows > 1:
        figsize = (12, 12)
    else:
        figsize = (12, 5)

    fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    fig, ax = remove_extra_axes(fig, ax, n_to_plot, n_rows * n_cols)
    ax = ax.flatten()

    for n, thing in enumerate(things_to_plot):
        for spectrum in spectra_to_plot:
            try:
                y = spectrum[thing]
            except KeyError:
                continue  # We will skip key errors, as models may have different inclinations

            if np.count_nonzero(y) == 0:  # skip arrays which are all zeros
                continue

            if use_flux:
                if spectrum.units == SPECTRUM_UNITS_FLM:
                    y *= spectrum["Lambda"]
                else:
                    y *= spectrum["Freq."]

            if spectrum.units == SPECTRUM_UNITS_FLM:
                x = spectrum["Lambda"]
            else:
                x = spectrum["Freq."]

            x, y = get_xy_subset(x, y, xmin, xmax)

            label = spectrum.fp.replace("_", r"\_") + spectrum.root.replace("_", r"\_")
            ax[n].plot(x, y, label=label, alpha=alpha)

            # Calculate the y-axis limits to keep all spectra within the
            # plot area

            # if not xmin:
            #     xmin = x.min()
            # if not xmax:
            #     xmax = x.max()
            #
            # if scale == "linlin" or scale == "logx":
            #     white_space_factor = 1.1
            # else:
            #     white_space_factor = 10

            # test_ymin, test_ymax = get_y_lims_for_x_lims(x, y, xmin, xmax, white_space_factor)
            #
            # if test_ymin < ymin:
            #     ymin = test_ymin
            # if test_ymax > ymax:
            #     ymax = test_ymax

        ax[n] = set_axes_scales(ax[n], scale)
        ax[n] = set_spectrum_axes_labels(ax[n], spectra_to_plot[0], use_flux)

        if thing.isdigit():
            ax[n].set_title(f"{thing}" + r"$^{\circ}$")
        else:
            ax[n].set_title(f"{thing}")

        if label_lines:
            ax[n] = _add_line_labels(ax[n], common_lines(spectrum=spectra_to_plot[0]), scale, linestyle="none")

    ax[0].legend(fontsize=10).set_zorder(0)
    fig.tight_layout(rect=[0.015, 0.015, 0.985, 0.985])
    fig.savefig(f"{output_name}.png")

    if display:
        plt.show()

    return fig, ax
