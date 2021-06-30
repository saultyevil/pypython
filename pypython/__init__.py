#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pypython - making using Python a wee bit easier.

pypython is a companion python package to handle and analyse the data which
comes out of a Python simulation.
"""

import copy
import re
import textwrap
import time
from os import listdir, path, remove
from pathlib import Path
from platform import system
from shutil import which
from subprocess import run

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import boxcar, convolve

from pypython.constants import (BOLTZMANN, CMS_TO_KMS, PARSEC, PI, PLANCK, VLIGHT)
from pypython.error import RunError
from pypython.math import vector
from pypython.physics.blackhole import gravitational_radius
from pypython.simulation.grid import get_parameter_value

# Constants --------------------------------------------------------------------

MINIMUM_PY_VERSION = "85"
MINIMUM_PY_SUB_VERSION = ""
PY_VERSION = ""

SPECTRUM_UNITS_LNU = "erg/s/Hz"
SPECTRUM_UNITS_FNU = "erg/s/cm^-2/Hz"
SPECTRUM_UNITS_FLM = "erg/s/cm^-2/A"
SPECTRUM_UNITS_UNKNOWN = "unknown"

LEN_WHEN_1D_MODEL = 4

WIND_COORD_TYPE_CYLINDRICAL = WIND_COORD_TYPE_CARTESIAN = "rectilinear"
WIND_COORD_TYPE_POLAR = "polar"
WIND_COORD_TYPE_SPHERICAL = "spherical"
WIND_COORD_TYPE_UNKNOWN = "unknown"

WIND_VELOCITY_UNITS_KMS = "kms"
WIND_VELOCITY_UNITS_CMS = "cms"
WIND_VELOCITY_UNITS_LIGHT = "c"

WIND_DISTANCE_UNITS_CM = "cm"
WIND_DISTANCE_UNITS_M = "m"
WIND_DISTANCE_UNITS_KM = "km"
WIND_DISTANCE_UNITS_RG = "rg"

CELL_MODEL_POWERLAW = 1
CELL_MODEL_EXPONENTIAL = 2

DEBUG_MASK_CELL_SPEC = False

# Functions --------------------------------------------------------------------


def _check_ascending(x):
    """Check if an array is sorted in ascending order.

    Parameters
    ----------
    x: np.ndarray, list
        The array to check.
    """
    return np.all(np.diff(x) >= 0)


def check_sorted_array_ascending(x):
    """Check if an array is sorted in ascending or descending order.

    If the array is not sorted, a ValueError will be raised.

    Parameters
    ----------
    x: np.ndarray, list
        The array to check.

    Returns
    -------
        Returns True if the array is in ascending order, otherwise will return
        False if in descending order.
    """
    if not _check_ascending(x):
        if _check_ascending(x.copy()[::-1]):
            return False
        else:
            raise ValueError("check_sorted_array_ascending: the array provided is not sorted at all")

    return True


def cleanup_data(fp=".", verbose=False):
    """Remove data symbolic links created by Python.

    Search recursively from the specified directory for symbolic links named
    data. This script will only work on Unix systems where the find command is
    available.
    todo: update to a system agnostic method to find symbolic links like pathlib

    Parameters
    ----------
    fp: str
        The starting directory to search recursively from for symbolic links
    verbose: bool [optional]
        Enable verbose output

    Returns
    -------
    n_del: int
        The number of symbolic links deleted
    """
    n_del = 0

    os = system().lower()
    if os != "darwin" and os != "linux":
        raise OSError("your OS does not work with this function, sorry!")

    # - type l will only search for symbolic links

    cmd = run_command(["find", ".", "-type", "l", "-name", "data"], path.expanduser(fp))
    stdout, stderr = cmd.stdout, cmd.stderr

    if stderr:
        print("sent from stderr")
        print(stderr.decode("utf-8"))

    if stdout and verbose:
        print(f"Deleting data symbolic links in the following directories:\n\n{stdout.decode('utf-8')[:-1]}")
    else:
        print("No data symlinks to delete")
        return n_del

    directories = stdout.decode("utf-8").split()

    for directory in directories:
        current = fp + directory[1:]
        sh = run_command(["rm", current], fp)
        stdout, stderr = sh.stdout, sh.stderr
        if stderr:
            print(stderr.decode("utf-8"))
        else:
            n_del += 1

    return n_del


def create_wind_save_tables(root, fp=".", ion_density=False, cell_spec=False, version=None, verbose=False):
    """Run windsave2table in a directory to create the standard data tables.

    The function can also create a root.all.complete.txt file which merges all
    the data tables together into one (a little big) file.

    Parameters
    ----------
    root: str
        The root name of the Python simulation
    fp: str
        The directory where windsave2table will run
    ion_density: bool [optional]
        Use windsave2table in the ion density version instead of ion fractions
    cell_spec: bool [optional]
        Use windsave2table to get the cell spectra.
    version: str [optional]
        The version number of windsave2table to use
    verbose: bool [optional]
        Enable verbose output
    """
    name = "windsave2table"
    if version:
        name += version

    in_path = which(name)
    if not in_path:
        raise OSError(f"{name} not in $PATH and executable")

    files_before = listdir(fp)

    if not Path(f"{fp}/data").exists():
        run_command("Setup_Py_Dir", fp)

    command = [name]
    if ion_density:
        command.append("-d")
    if cell_spec:
        command.append("-xall")
    command.append(root)

    cmd = run_command(command, fp, verbose)
    if cmd.returncode != 0:
        raise RunError(
            f"windsave2table has failed to run, possibly due to an incompatible version\n{cmd.stdout.decode('utf-8')}")

    files_after = listdir(fp)

    # Move the new files in fp/tables

    s = set(files_before)
    new_files = [x for x in files_after if x not in s]
    Path(f"{fp}/tables").mkdir(exist_ok=True)
    for new in new_files:
        try:
            Path(f"{fp}/{new}").rename(f"{fp}/tables/{new}")
        except PermissionError:
            time.sleep(1.5)
            Path(f"{fp}/{new}").rename(f"{fp}/tables/{new}")

    return cmd.returncode


def find(pattern, fp="."):
    """Find files of the given pattern recursively.

    This is used to find a number files given a gloable pattern, i.e. *.spec,
    *.pf. When *.py is used, it'll ignore out.pf and py_wind.pf files. To find
    py_wind.pf files, use py_wind.pf as the pattern.

    Parameters
    ----------
    pattern: str
        Patterns to search recursively for, i.e. *.pf, *.spec, tde_std.pf
    fp: str [optional]
        The directory to search from, if not specified in the pattern.
    """

    files = [str(file_) for file_ in Path(f"{fp}").rglob(pattern)]
    if ".pf" in pattern:
        files = [file_ for file_ in files if "out.pf" not in file_ and "py_wind" not in file_]

    try:
        files.sort(key=lambda var: [int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
    except TypeError as e:
        print(e)
        print(f"warning: {find.__name__}: have been unable to sort output, be careful as results may not be "
              f"reproducible")

    return files


def get_array_index(x, target):
    """Return the index for a given value in an array.

    If an array with duplicate values is passed, the first instance of that
    value will be returned. The array must also be sorted, in either ascending
    or descending order.

    Parameters
    ----------
    x: np.ndarray
        The array of values.
    target: float
        The value, or closest value, to find the index of.

    Returns
    -------
    The index for the target value in the array x.
    """
    if check_sorted_array_ascending(x):
        if target < np.min(x):
            return 0
        if target > np.max(x):
            return -1
    else:
        if target < np.min(x):
            return -1
        if target > np.max(x):
            return 0

    return np.abs(x - target).argmin()


def get_python_version():
    """Check the version of Python available in $PATH.

    There are a number of features in this package which are not
    available in older versions of Python.

    Returns
    -------
    version: str
        The version string of the currently compiled Python.
    """

    command = run_command(["py", "--version"])
    stdout = command.stdout.decode("utf-8").split("\n")
    stderr = command.stderr.decode("utf-8")

    if stderr:
        raise SystemError(f"{stderr}")

    version = None
    for line in stdout:
        if line.startswith("Python Version"):
            version = line[len("Python Version") + 1:]

    # if version is None:
    #     raise SystemError("Unable to determine Python version")
    #
    # sub_version = version.lstrip("0123456789")
    # main_version = version[:version.index(sub_version)]
    #
    # if main_version < MINIMUM_PY_VERSION or sub_version < MINIMUM_PY_VERSION:
    #     raise SystemError(f"Python version {main_version}{sub_version} below minimum version of {MINIMUM_PY_VERSION}"
    #                       f"{MINIMUM_PY_SUB_VERSION}")

    return version


def get_root_name(fp):
    """Get the root name of a Python simulation.

    Extracts both the file path and the root name of the simulation.

    Parameters
    ----------
    fp: str
        The directory path to a Python .pf file

    Returns
    -------
    root: str
        The root name of the Python simulation
    where: str
        The directory path containing the provided Python .pf file
    """
    if type(fp) is not str:
        raise TypeError("expected a string as input for the file path, not whatever you put")

    dot = fp.rfind(".")
    slash = fp.rfind("/")

    root = fp[slash + 1:dot]
    fp = fp[:slash + 1]

    if fp == "":
        fp = "./"

    return root, fp


def smooth_array(array, width):
    """Smooth a 1D array of data using a boxcar filter.

    Parameters
    ----------
    array: np.array[float]
        The array to be smoothed.
    width: int
        The size of the boxcar filter.

    Returns
    -------
    smoothed: np.ndarray
        The smoothed array
    """
    if width is None or width == 0:
        return array

    array = np.reshape(array, (len(array), ))  # todo: why do I have to do this? safety probably

    return convolve(array, boxcar(width) / float(width), mode="same")


def run_py_optical_depth(root, photosphere=None, fp=".", verbose=False):

    command = ["py_optical_depth"]
    if photosphere:
        command.append(f"-p {float(photosphere)}")
    command.append(root)

    cmd = run_command(command, fp)
    stdout, stderr = cmd.stdout, cmd.stderr

    if verbose:
        print(stdout.decode("utf-8"))

    if stderr:
        print(stderr.decode("utf-8"))


def run_py_wind(root, commands, fp="."):
    """Run py_wind with the provided commands.

    Parameters
    ----------
    root: str
        The root name of the model.
    commands: list[str]
        The commands to pass to py_wind.
    fp: [optional] str
        The directory containing the model.

    Returns
    -------
    output: list[str]
        The stdout output from py_wind.
    """
    cmd_file = f"{fp}/.tmpcmds.txt"

    with open(cmd_file, "w") as f:
        for i in range(len(commands)):
            f.write(f"{commands[i]}\n")

    with open(cmd_file, "r") as stdin:
        sh = run(["py_wind", root], stdin=stdin, capture_output=True, cwd=fp)

    stdout, stderr = sh.stdout, sh.stderr
    if stderr:
        print(stderr.decode("utf-8"))

    remove(cmd_file)

    return stdout.decode("utf-8").split("\n")


# Cell spectra -----------------------------------------------------------------


class CellSpectra:
    """A class to store the cell spectra accumulated during the ionization
    cycles.

    Cells which do not have a spectrum will return None.
    """
    def __init__(self, root, fp=".", nx=0, nz=0, force_make_spectra=False, delim=None):
        """Initialize the object.

        Reads in the cell spectra into a 1D list. This function will attempt
        to run windsave2table to create the cell spectra files if they do not
        exist.

        Parameters
        ----------
        root: str
            The root name of the Python simulation.
        fp: str [optional]
            The directory containing the model.
        nx: int [optional]
            The number of cells in the x direction.
        nz: int [optional]
            The number of cells in the z direction.
        force_make_spectra: bool [optional]
            Force windsave2table to be run to re-make the files in the
            tables directory.
        delim: str [optional]
            The delimiter used in the wind table files.
        """
        self.root = root

        self.fp = fp
        if self.fp[-1] != "/":
            self.fp += "/"
        self.pf = self.fp + self.root + ".pf"

        # Set initial conditions to create member variables

        self.nx = int(nx)
        self.nz = int(nz)
        self.header = None
        self.cells = None
        self.spectra = None

        self.original = None

        # Try to read in the spectra, if we can't then we'll try and run
        # windsave2table. It is also possible to force the re-creation of the
        # spectra files

        if force_make_spectra:
            self.create_wind_tables()

        try:
            self.get_cell_spectra(delim)
        except IOError:
            create_wind_save_tables(root, fp, cell_spec=True)
            self.get_cell_spectra(delim)

    # Methods ------------------------------------------------------------------

    def create_wind_tables(self):
        """Force the creation of wind save tables for the model.

        This is best used when a simulation has been re-run, as the
        library is unable to detect when the currently available wind
        tables do not reflect a new simulation. This function will
        create the standard wind tables, as well as the fractional and
        density ion tables and create the xspec cell spectra files.
        """

        create_wind_save_tables(self.root, self.fp, ion_density=True)
        create_wind_save_tables(self.root, self.fp, ion_density=False)
        create_wind_save_tables(self.root, self.fp, cell_spec=True)

    def get_cell_spectra(self, delim=None):
        """Read in the cell spectra.

        This function will read in spectra from across multiple files,
        if there are multiple files at least.
        """
        self.header = []
        cell_spectra = []
        frequency_bins = []

        # Loop over each file. Each time self.header is updated, but we store
        # the rest into an array which gets hstacked to make a single array

        for fp in find("*xspec.*.txt", self.fp):

            with open(fp, "r") as f:
                spectrum_file = f.readlines()

            spectra_lines = []

            for line in spectrum_file:
                if len(line) == 0 or line.startswith("#"):
                    continue
                if delim:
                    line = line.split(delim)
                else:
                    line = line.split()

                spectra_lines.append(line)

            self.header += spectra_lines[0][1:]
            array = np.array(spectra_lines[1:], dtype=np.float64)
            frequency_bins.append(array[:, 0])
            cell_spectra.append(array[:, 1:])

        cell_spectra = [np.hstack(cell_spectra)]
        frequency_bins = np.array(frequency_bins[0], dtype=np.float64)  # assuming they're all the same...
        cell_spectra = cell_spectra[0]

        # Now extract the cell indices from the header, of course done
        # differently depending on if the model is 1D or 2D.

        if len(self.header[0]) > LEN_WHEN_1D_MODEL:
            self.cells = [(int(cell[1:4]), int(cell[5:])) for cell in self.header]
        else:
            self.cells = [(int(cell[1:4]), 0) for cell in self.header]

        # If nx or nz were not given, then determine the number of coordinates
        # from the parameter files or from the cells array

        if self.nx == 0 or self.nz == 0:
            try:
                self.nx = int(get_parameter_value(self.pf, "Wind.dim.in.x_or_r.direction"))
                if len(self.header[1]) > LEN_WHEN_1D_MODEL:
                    self.nz = int(get_parameter_value(self.pf, "Wind.dim.in.z_or_theta.direction"))
            except (ValueError, IOError):
                self.nx = self.cells[-1][0] + 1
                self.nz = self.cells[-1][1] + 1

        # The final step is to create a 2D array of Nones and then populate the
        # cells which have spectra with a dict with keys Freq. and Flux

        self.spectra = np.array([None for _ in range(self.nx * self.nz)], dtype=dict).reshape(self.nx, self.nz)
        for n, (i, j) in enumerate(self.cells):
            self.spectra[i, j] = {"Freq.": np.copy(frequency_bins), "Flux": cell_spectra[:, n]}

    def get_elem_number_from_ij(self, i, j):
        """Get the wind element number for a given i and j index.

        Used when indexing into a 1D array, such as in Python itself.

        Parameters
        ----------
        i: int
            The i-th index of the cell.
        j: int
            The j-th index of the cell.
        """
        return int(self.nz * i + j)

    def get_ij_from_elem_number(self, elem):
        """Get the i and j index for a given wind element number.

        Used when converting a wind element number into two indices for use
        in this package.

        Parameters
        ----------
        elem: int
            The element number.
        """
        return np.unravel_index(elem, (self.nx, self.nz))

    def plot(self, i, j, energetic=True, scale="loglog", fig=None, ax=None, figsize=(12, 6)):
        """Plot the spectrum for cell (i, j).

        Simple plotting function, if you want something more advanced then
        check out pypython.plot.spectrum.

        Parameters
        ----------
        i: int
            The i-th index for the cell.
        j: int
            The j-th index for the cell.
        energetic: bool
            Plot in energetic (nu * J_nu) units.
        scale: str
            The axes scaling for the plot.
        fig: pyplot.Figure
            A matplotlib Figure object to edit. Needs ax to also be supplied.
        ax: pyplot.Axes
            A matplotlib Axes object to edit. Needs fig to also be supplied.
        figsize: tuple(int, int)
            The size of the figure (width, height) in inches (sigh).
        """
        spectrum = self.spectra[i, j]
        if spectrum is None:
            raise ValueError(f"no modelled cell spectra for cell ({i}, {j}) as cell is probably not in the wind")

        if not fig and not ax:
            plot.normalize_figure_style()
            fig, ax = plt.subplots(figsize=figsize)
        elif not fig and ax or fig and not ax:
            raise ValueError("fig and ax need to be supplied together")

        if energetic:
            ax.plot(spectrum["Freq."], spectrum["Freq."] * spectrum["Flux"], label="Spectrum")
            ax.set_ylabel(r"$\nu J_{\nu}$ [ergs s$^{-1}$ cm$^{-2}$]")
        else:
            ax.plot(spectrum["Freq."], spectrum["Flux"], label="Spectrum")
            ax.set_ylabel(r"$J_{\nu}$ [ergs s$^{-1}$ cm$^{-2}$ Hz$^{-1}$]")

        ax.set_xlabel("Rest-frame Frequency")
        ax = set_axes_scales(ax, scale)
        fig.suptitle(f"Spectrum in cell ({i}, {j})")

        return fig, ax

    def smooth(self):
        raise NotImplementedError()

    @staticmethod
    def show(block=True):
        """Show a plot which has been created.

        Wrapper around pyplot.show().

        Parameters
        ----------
        block: bool
            Use blocking or non-blocking figure display.
        """
        plt.show(block=block)

    # Built in stuff -----------------------------------------------------------

    def __getitem__(self, pos):
        i, j = pos
        return self.spectra[i, j]

    def __setitem__(self, pos, value):
        i, j = pos
        self.spectra[i, j] = value

    def __str__(self):
        return print(self.spectra)


# Modelled cell spectra --------------------------------------------------------


class ModelledCellSpectra:
    """A class to store the modelled cell spectra used in the ionization cycles
    to calculate various quantities.

    Cells which do not have a spectrum will return None.
    """
    def __init__(self, root, fp=".", force_make_tables=False, delim=None):
        """Initialize the object.

        Reads in the root.spec.txt file, which contains a bunch of parameters
        used to make a model spectrum.

        Parameters
        ----------
        root: str
            The root name of the Python simulation.
        fp: str [optional]
            The directory containing the model.
        force_make_tables: bool [optional]
            Force windsave2table to be run to re-make the files in the
            tables directory.
        delim: str [optional]
            The delimiter used in the wind table files.
        """
        self.root = root

        self.fp = fp
        if self.fp[-1] != "/":
            self.fp += "/"
        self.pf = self.fp + self.root + ".pf"

        # Set initial conditions to create member variables

        self.nx = 0
        self.nz = 0
        self.n_cells = 0
        self.n_bands = 0
        self.header = None
        self.cells = []
        self.model_parameters = None
        self.spectra = None

        self.n_bins_per_band = 500

        # Read in the model parameters, i.e. the root.spec.txt file, and
        # convert those into a flux. The wind tables can be forcibly created
        # if need be

        if force_make_tables:
            self.create_wind_tables()

        self.get_model_spectra(delim)
        self.construct_models()

    # Methods ------------------------------------------------------------------

    def construct_models(self):
        """Use the model parameters to create the flux models.

        This uses the model parameters, in self.model_parameters, to construct
        models of the cell spectrum. The number of frequency bins for each
        band is controlled by self.n_bins_per_band, so the number of frequency
        bins for the whole model is self.n_bins_per_band * self.nbands (which
        saw photons).
        """
        self.spectra = np.array([None for _ in range(self.n_cells)]).reshape(self.nx, self.nz)

        # We need to loop over each cell and construct the flux for each
        # band but they will be joined together at the end

        for n in range(self.n_cells):
            i, j = self.get_ij_from_elem_number(n)
            cell_model = self.model_parameters[i, j]

            cell_flux = []
            cell_frequency = []

            for band in cell_model:
                band_freq_min = band["fmin"]
                band_freq_max = band["fmax"]

                # In some cases freq_min == freq_max == 0. These are bands which
                # did not see any photons so we cannot construct a spectrum
                # from this

                if band_freq_max > band_freq_min:
                    model_type = band["spec_mod_type"]
                    frequency = np.logspace(np.log10(band_freq_min), np.log10(band_freq_max), self.n_bins_per_band)

                    # There are two model types, a power law or an exponential
                    # model

                    if model_type == CELL_MODEL_POWERLAW:
                        f_nu = 10**(band["pl_log_w"] + np.log10(frequency) * band["pl_alpha"])
                    elif model_type == CELL_MODEL_EXPONENTIAL:
                        f_nu = band["exp_w"] * np.exp((-1 * PLANCK * frequency) / (band["exp_temp"] * BOLTZMANN))
                    else:
                        f_nu = np.zeros_like(frequency)

                    cell_frequency.append(frequency)
                    cell_flux.append(f_nu)

            # If we constructed a model, then set it in the array. Otherwise,
            # the cell will still contain None

            if len(cell_flux) != 0:
                cell_frequency = np.hstack(cell_frequency)
                if not np.all(np.diff(cell_frequency) >= 0):
                    raise ValueError("frequency bins are not increasing across the bands, for some reason")
                cell_flux = np.hstack(cell_flux)
                self.spectra[i, j] = {"Freq.": cell_frequency, "Flux": cell_flux}
                self.cells.append((i, j))

    def create_wind_tables(self):
        """Force the creation of wind save tables for the model.

        This is best used when a simulation has been re-run, as the
        library is unable to detect when the currently available wind
        tables do not reflect a new simulation. This function will
        create the standard wind tables, as well as the fractional and
        density ion tables and create the xspec cell spectra files.
        """

        create_wind_save_tables(self.root, self.fp, ion_density=True)
        create_wind_save_tables(self.root, self.fp, ion_density=False)
        create_wind_save_tables(self.root, self.fp, cell_spec=True)

    def get_elem_number_from_ij(self, i, j):
        """Get the wind element number for a given i and j index.

        Used when indexing into a 1D array, such as in Python itself.

        Parameters
        ----------
        i: int
            The i-th index of the cell.
        j: int
            The j-th index of the cell.
        """
        return int(self.nz * i + j)

    def get_ij_from_elem_number(self, elem):
        """Get the i and j index for a given wind element number.

        Used when converting a wind element number into two indices for use
        in this package.

        Parameters
        ----------
        elem: int
            The element number.
        """
        return np.unravel_index(elem, (self.nx, self.nz))

    def get_model_spectra(self, delim):
        """Read in the model spectrum parameters.

        The model spectrum parameters are read into self.model_parameters and
        are stored such as model_parameters[i, j][nband] where i and j are the
        cell indices and nband is a band number. Each model parameter is
        stored as a dict of the different parameters.

        Parameters
        ----------
        delim: str
            The file delimiter.
        """
        fp = self.fp + self.root + ".spec.txt"
        if not path.exists(fp):
            fp = self.fp + "tables/" + self.root + ".spec.txt"
            if not path.exists(fp):
                raise IOError(f"unable to find a {self.root}.spec file")

        # Read the file in, obviously, store all the lines into a numpy array
        # other than the header

        with open(fp, "r") as f:
            models = f.readlines()

        model_lines = []

        for line in models:
            line = line.strip()
            if delim:
                line = line.split(delim)
            else:
                line = line.split()
            if len(line) == 0 or line[0] == "#":
                continue
            model_lines.append(line)

        models = np.array(model_lines[1:], dtype=np.float64)

        # Now we can determine the parameters read in, the number of bands
        # and the number of cells in the wind

        self.header = model_lines[0]
        self.n_bands = int(np.max(models[:, self.header.index("nband")])) + 1

        self.nx = int(np.max(models[:, self.header.index("i")])) + 1
        try:
            self.nz = int(np.max(models[:, self.header.index("j")])) + 1
        except ValueError:
            self.nz = 1
        self.n_cells = self.nx * self.nz

        # Store the model parameters as model_parameters[i, j][n_band]
        # where i and j are cell indices and n_band is the band number

        self.model_parameters = np.array([None for _ in range(self.n_cells)]).reshape(self.nx, self.nz)

        for nelem in range(self.n_cells):
            bands = []
            for n in range(self.n_bands):
                band = {}
                for m, col in enumerate(self.header):
                    band[col] = models[nelem + (n * self.n_cells), m]
                bands.append(band)

            i, j = self.get_ij_from_elem_number(nelem)
            self.model_parameters[i, j] = bands

    def plot(self, i, j, energetic=True, scale="loglog", fig=None, ax=None, figsize=(12, 6)):
        """Plot the spectrum for cell (i, j).

        Simple plotting function, if you want something more advanced then
        check out pypython.plot.spectrum.

        Parameters
        ----------
        i: int
            The i-th index for the cell.
        j: int
            The j-th index for the cell.
        energetic: bool
            Plot in energetic (nu * J_nu) units.
        scale: str
            The axes scaling for the plot.
        fig: pyplot.Figure
            A matplotlib Figure object to edit. Needs ax to also be supplied.
        ax: pyplot.Axes
            A matplotlib Axes object to edit. Needs fig to also be supplied.
        figsize: tuple(int, int)
            The size of the figure (width, height) in inches (sigh).
        """
        model = self.spectra[i, j]
        if model is None:
            raise ValueError(f"no modelled cell spectra for cell ({i}, {j}) as cell is probably not in the wind")

        if not fig and not ax:
            plot.normalize_figure_style()
            fig, ax = plt.subplots(figsize=figsize)
        elif not fig and ax or fig and not ax:
            raise ValueError("fig and ax need to be supplied together")

        if energetic:
            ax.plot(model["Freq."], model["Freq."] * model["Flux"], label="Model")
            ax.set_ylabel(r"$\nu J_{\nu}$ [ergs s$^{-1}$ cm$^{-2}$]")
        else:
            ax.plot(model["Freq."], model["Flux"], label="Model")
            ax.set_ylabel(r"$J_{\nu}$ [ergs s$^{-1}$ cm$^{-2}$ Hz$^{-1}$]")

        ax.set_xlabel("Rest-frame Frequency")
        ax = set_axes_scales(ax, scale)
        fig.suptitle(f"Spectrum in cell ({i}, {j})")

        return fig, ax

    @staticmethod
    def show(block=True):
        """Show a plot which has been created.

        Wrapper around pyplot.show().

        Parameters
        ----------
        block: bool
            Use blocking or non-blocking figure display.
        """
        plt.show(block=block)

    # Built in stuff -----------------------------------------------------------

    def __getitem__(self, pos):
        i, j = pos
        return self.spectra[i, j]

    def __setitem__(self, pos, value):
        i, j = pos
        self.spectra[i, j] = value

    def __str__(self):
        return print(self.spectra)


# Spectrum class ---------------------------------------------------------------


class Spectrum:
    """A class to store PYTHON .spec and .log_spec files.

    The Python spectra are read in and stored within a dict of dicts,
    where each column name is the spectrum name and the columns in that
    dict are the names of the columns in the spectrum file. The data is
    stored as numpy arrays.
    """
    def __init__(self, root, fp=".", default=None, log_spec=True, smooth=None, distance=None, delim=None):
        """Create the Spectrum object.

        Construct the file path of the spectrum files given the
        root, directory and whether the logarithmic spectrum is used or not.
        The different spectra are then read in, with either the .spec or the
        first spectrum file read in being the default index choice.

        Parameters
        ----------
        root: str
            The root name of the model.
        fp: str [optional]
            The directory containing the model.
        default: str [optional]
            The default spectrum to make the available spectrum for indexing.
        log_spec: bool [optional]
            Read in the logarithmic version of the spectra.
        smooth: int [optional]
            The amount of smoothing to use.
        distance: float [optional]
            The distance of the spectrum flux, in units of parsec.
        delim: str [optional]
            The deliminator in the spectrum file.
        """
        self.root = root

        self.fp = fp
        if self.fp[-1] != "/":
            self.fp += "/"
        self.pf = self.fp + self.root + ".pf"

        self.log_spec = log_spec
        if default and self.log_spec:
            if not default.startswith("log_") and default != "spec_tau":
                default = "log_" + default

        # Initialize the important members
        # These are for the "current/target" spectrum. This is done like this
        # so you can use spectrum.distance to look at the currently set spectrum
        # without having to index like spectrum.distance["spec"].

        self.spectrum = {}
        self.columns = ()
        self.inclinations = ()
        self.n_inclinations = 0
        self.units = SPECTRUM_UNITS_UNKNOWN
        self.distance = 100

        # Anything with avail_ are for recording all of the spectra.
        # todo: I wish I had a better way to do this lol

        self.original_spectra = None

        self.available = []
        self.avail_spectrum = {}
        self.avail_columns = {}
        self.avail_inclinations = {}
        self.avail_n_inclinations = {}
        self.avail_units = {}
        self.avail_distance = {}

        # Now we can read in the spectra and set the default/target spectrum
        # for the object. We can also re-scale to a different distance.

        self.get_spectra(delim)

        if default:
            self.current = default
        else:
            self.current = self.available[0]

        self.set(self.current)

        if distance:
            self.rescale_to_distance(distance)

        # Smooth all the spectra. A copy of the unsmoothed spectra is kept
        # in the member self.original.

        if smooth:
            self.smooth(smooth)

    # Private methods ----------------------------------------------------------

    def _plot_observer_spectrum(self, label_lines=False):
        """Plot the spectrum components and observer spectra on a 1x2 panel
        plot. The left panel has the components, whilst the right panel has the
        observer spectrum.

        Parameters
        ----------
        label_lines: bool
            Plot line IDs.
        """
        if "spec" not in self.available and "log_spec" not in self.available:
            raise IOError("A .spec/.log_spec file was not read in, cannot use this function")

        normalize_figure_style()

        fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharey="row")

        # Plot the components of the observer spectrum, i.e Emitted, Created,
        # Disc, etc.

        for component in self.columns[:-self.n_inclinations]:
            if component in ["Lambda", "Freq."]:
                continue
            ax[0] = self._plot_thing(component, label_lines, ax[0])

        for line in ax[0].get_lines():  # Set the different spectra to have a transparency
            line.set_alpha(0.7)
        ax[0].legend(ncol=2, loc="upper right").set_zorder(0)

        # Now plot the observer spectra

        for inclination in self.inclinations:
            ax[1] = self._plot_thing(inclination, label_lines, ax[1])

        for label, line in zip(self.inclinations, ax[1].get_lines()):
            line.set_alpha(0.7)
            line.set_label(str(label) + r"$^{\circ}$")
        ax[1].set_ylabel("")
        ax[1].legend(ncol=2, loc="upper right").set_zorder(0)

        # Final clean up to make a nice spectrum

        ax[0].set_title("Components")
        ax[1].set_title("Observer spectra")
        fig.tight_layout(rect=[0.015, 0.015, 0.985, 0.985])
        fig.subplots_adjust(wspace=0)

        return fig, ax

    def _plot_thing(self, thing, label_lines=False, ax_update=None):
        """Plot a specific column in a spectrum file.

        Parameters
        ----------
        thing: str
            The name of the thing to be plotted.
        label_lines: bool
            Plot line IDs.
        ax_update: plt.Axes
            An plt.Axes object to update, i.e. to plot on.
        """
        normalize_figure_style()

        if ax_update:
            ax = ax_update
        else:
            fig, ax = plt.subplots(figsize=(9, 5))

        ax.set_yscale("log")
        ax.set_xscale("log")

        # How things are plotted depends on the units of the spectrum

        if self.units == SPECTRUM_UNITS_FLM:
            ax.plot(self.spectrum["Lambda"], self.spectrum[thing], label=thing)
            ax.set_xlabel(r"Wavelength [\AA]")
            ax.set_ylabel(r"Flux Density " + f"{self.distance:.2e}" + r"pc [erg s$^{-1}$ cm$^{-2}$ \AA$^{-1}$]")
            if label_lines:
                ax = add_line_ids(ax, common_lines(freq=False), logx=True)
        else:
            ax.plot(self.spectrum["Freq."], self.spectrum[thing], label=thing)
            ax.set_xlabel("Frequency [Hz]")
            if self.units == SPECTRUM_UNITS_LNU:
                ax.set_ylabel(r"Luminosity [erg s$^{-1}$ Hz$^{-1}$]")
            else:
                ax.set_ylabel(r"Flux Density " + f"{self.distance:.2e}" + r"pc [erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$]")
            if label_lines:
                ax = add_line_ids(ax, common_lines(freq=True), logx=True)

        if ax_update:
            return ax
        else:
            fig.tight_layout(rect=[0.015, 0.015, 0.985, 0.985])
            return fig, ax

    # Methods ------------------------------------------------------------------

    def get_spectra(self, delim=None):
        """Read in a spectrum file given in self.filepath. The spectrum is
        stored as a dictionary in self.spectrum where each key is the name of
        the columns.

        Parameters
        ----------
        delim: str [optional]
            A custom delimiter, useful for reading in files which have sometimes
            between delimited with commas instead of spaces.
        """
        n_read = 0
        files_to_read = ["spec", "spec_tot", "spec_tot_wind", "spec_wind", "spec_tau"]

        # Read in each spec file type, and store each spectrum as a key in
        # self.avail_spec, etc.

        for spec_type in files_to_read:
            fp = self.fp + self.root + "."
            if self.log_spec and spec_type != "spec_tau":
                spec_type = "log_" + spec_type
            fp += spec_type
            if not path.exists(fp):
                continue

            n_read += 1
            self.avail_spectrum[spec_type] = {}
            self.avail_units[spec_type] = SPECTRUM_UNITS_UNKNOWN
            self.avail_distance[spec_type] = 0.0

            with open(fp, "r") as f:
                spectrum_file = f.readlines()

            # Read in the spectrum file. Ignore empty lines and lines which have
            # been commented out by #

            spectrum = []

            for line in spectrum_file:
                line = line.strip()
                if delim:
                    line = line.split(delim)
                else:
                    line = line.split()
                if "Units:" in line:
                    self.avail_units[spec_type] = line[4][1:-1]
                    if self.avail_units[spec_type] in [SPECTRUM_UNITS_FLM, SPECTRUM_UNITS_FNU]:
                        self.avail_distance[spec_type] = float(line[6])
                    else:
                        self.avail_distance[spec_type] = 0
                if len(line) == 0 or line[0] == "#":
                    continue
                spectrum.append(line)

            # Extract the header columns of the spectrum. This assumes the first
            # read line in the spectrum is the header.

            header = []  # wish this was short enough to do in a list comprehension

            for i, column_name in enumerate(spectrum[0]):
                if column_name[0] == "A":
                    j = column_name.find("P")
                    column_name = column_name[1:j].lstrip("0")  # remove leading 0's for, i.e., 01 degrees
                header.append(column_name)

            spectrum = np.array(spectrum[1:], dtype=np.float64)

            # Add the spectrum to self.avail_spectrum[spec_type]. The keys of
            # the dictionary are the column names in the spectrum, i.e. what
            # is in the header

            for i, column_name in enumerate(header):
                self.avail_spectrum[spec_type][column_name] = spectrum[:, i]

            inclinations = []  # this could almost be a list comprehension...

            for col in header:
                if col.isdigit() and col not in inclinations:
                    inclinations.append(col)

            self.avail_columns[spec_type] = tuple(header)
            self.avail_inclinations[spec_type] = tuple(inclinations)
            self.avail_n_inclinations[spec_type] = len(inclinations)

        if n_read == 0:
            raise IOError(f"Unable to open any spectrum files for {self.root} in {self.fp}")

        self.available = tuple(self.avail_spectrum.keys())

    def plot(self, name=None, spec_type=None, label_lines=False):
        """Plot the spectra or a single component in a single figure. By
        default this creates a 1 x 2 of the components on the left and the
        observer spectra on the right. Useful for when in an interactive
        session.

        Parameters
        ----------
        name: str
            The name of the thing to plot.
        spec_type: str
            The spectrum the thing to plot belongs in.
        label_lines: bool
            Plot line IDs.
        """
        current = self.current  # keep track of this

        # If name is given, then plot that column of the spectrum. Otherwise
        # assume we just want to plot all columns in the spec file

        if name:
            if spec_type:
                self.set(spec_type)
            if name not in self.columns:
                raise ValueError(f"{name} is not available in the {self.current} spectrum")
            fig, ax = self._plot_thing(name, label_lines)
            if name.isdigit():
                name += r"$^{\circ}$"
            ax.set_title(name.replace("_", r"\_"))
        else:
            self.current = "spec"
            fig, ax = self._plot_observer_spectrum(label_lines)

        # self.current can be changed by the above functions, so we must now
        # change it back now. This is bad programming, and shouldn't be
        # required really.

        self.current = current

        return fig, ax

    def set(self, name):
        """Set a spectrum as the default.

        Sets a different spectrum to be the currently available spectrum for
        indexing.

        Parameters
        ----------
        name: str
            The name of the spectrum, i.e. log_spec or spec_tot, etc. The
            available spectrum types are stored in self.available.
        """
        if self.log_spec and not name.startswith("log_") and name != "spec_tau":
            name = "log_" + name

        if name not in self.available:
            raise IndexError(f"spectrum {name} is not available: available are {self.available}")

        self.spectrum = self.avail_spectrum[name]
        self.columns = self.avail_columns[name]
        self.inclinations = self.avail_inclinations[name]
        self.n_inclinations = self.avail_n_inclinations[name]
        self.units = self.avail_units[name]
        self.distance = self.avail_distance[name]

    @staticmethod
    def show(block=True):
        """Show a plot which has been created.

        Wrapper around pyplot.show().

        Parameters
        ----------
        block: bool
            Use blocking or non-blocking figure display.
        """
        plt.show(block=block)

    def smooth(self, width=5, to_smooth=None):
        """Smooth the spectrum flux/luminosity bins.

        If this is used after the spectrum has already been smoothed, then the
        "original" is copied back into the spectrum before smoothing again. This
        way the function does not smooth an already smoothed spectrum.

        Parameters
        ----------
        width: int [optional]
            The width of the boxcar filter (in bins).
        to_smooth: list or tuple of strings [optional]
            A list or tuple
        """
        if self.original_spectra is None:
            self.original_spectra = copy.deepcopy(self.spectrum)
        else:
            self.spectrum = copy.deepcopy(self.original_spectra)

        if to_smooth is None:
            to_smooth = ("Created", "WCreated", "Emitted", "CenSrc", "Disk", "Wind", "HitSurf", "Scattered") + tuple(
                self.inclinations)
        elif type(to_smooth) is str:
            to_smooth = to_smooth,
        elif type(to_smooth) is tuple or type(to_smooth) is list:
            pass
        else:
            raise ValueError("unknown format for argument to_smooth, must be a tuple/list of str or str")

        # Loop over each available spectrum and smooth it

        for key in self.available:
            if key == "spec_tau":  # todo: cleaner way to skip spec_tau
                continue
            for thing_to_smooth in to_smooth:
                try:
                    self.avail_spectrum[key][thing_to_smooth] = \
                        smooth_array(self.avail_spectrum[key][thing_to_smooth], width)
                except KeyError:
                    pass  # some spectra do not have the inclination angles...

    def rescale_to_distance(self, distance):
        """Rescale the flux to the given distance.

        Parameters
        ----------
        distance: float or int
            The distance to scale the flux to.
        """
        if type(distance) is not float and type(distance) is not int:
            raise ValueError("distance is not a float or integer")

        for spectrum in self.available:
            if spectrum == "spec_tau":
                continue
            if self.avail_units[spectrum] == SPECTRUM_UNITS_LNU:
                continue
            for key in self.avail_spectrum[spectrum].keys():
                if key in ["Lambda", "Freq."]:
                    continue
                self.avail_spectrum[spectrum][key] *= \
                    (self.avail_distance[spectrum] * PARSEC)**2 / (distance * PARSEC)**2
            self.avail_distance[spectrum] = distance

        self.distance = distance

    def restore_original_spectra(self):
        """Restore the spectrum to its original unsmoothed form."""

        self.spectrum = copy.deepcopy(self.original_spectra)

    # Built in stuff -----------------------------------------------------------

    def __getitem__(self, key):
        if key not in self.available:
            return self.spectrum[key]
        else:
            return self.avail_spectrum[key]

    def __setitem__(self, key, value):
        if key not in self.available:
            self.spectrum[key] = value
        else:
            self.avail_spectrum[key] = value

    def __str__(self):
        msg = f"Spectrum for the model {self.root} in {self.fp}\n"
        msg += f"Available spectra: {self.available}\n"
        msg += f"Current spectrum {self.current}\n"
        if "spec" in self.available or "log_spec" in self.available:
            if self.log_spec:
                key = "log_spec"
            else:
                key = "spec"
            msg += f"Spectrum inclinations: {self.avail_inclinations[key]}\n"
        if "tau_spec" in self.available:
            msg += f"Optical depth inclinations {self.avail_inclinations['tau_spec']}\n"

        return textwrap.dedent(msg)


# Wind class -------------------------------------------------------------------


class Wind:
    """A class to store 1D and 2D Python wind tables.

    Contains methods to extract variables, as well as convert various
    indices into other indices.
    todo: I should include kwargs in a bunch of these functions
    """
    def __init__(self,
                 root,
                 fp=".",
                 get_cell_spec=True,
                 spatial_units="cm",
                 co_mass=None,
                 velocity_units="kms",
                 masked=True,
                 force_make_tables=False,
                 version=None,
                 delim=None):
        """Initialize the Wind object.

        Each of the available wind save tables or ion tables are read in, and
        stored in the same dictionary. To access each paramter, using the get()
        method is perferred. However, it is also possible to index using the
        regular [ ] operator. To get an ion, the index is as follows
        w["H"]["density"]["i01"]. If using get(), it is insted as
        get("H_i01d") or get("H_i01f") for the ion fractions.

        Parameters
        ----------
        root: str
            The root name of the Python simulation.
        fp: str
            The directory containing the model.
        get_cell_spec: bool
            Load in the the cell spectra as well.
        spatial_units: str
            The distance units of the wind.
        co_mass: float
            The mass of the central object, optional to use in conjunction with
            distance_units == "rg"
        velocity_units: str [optional]
            The velocity units of the wind.
        masked: bool [optional]
            Store the wind parameters as masked arrays.
        force_make_tables: bool [optional]
            Force windsave2table to be run to re-make the files in the
            tables directory.
        delim: str [optional]
            The delimiter used in the wind table files.
        """
        self.root = root

        self.fp = fp
        if self.fp[-1] != "/":
            self.fp += "/"
        self.pf = self.fp + self.root + ".pf"

        self.version = version

        # Set initial conditions for all of the important variables

        self.nx = 1
        self.nz = 1
        self.n_elem = 1
        self.x_coords = None
        self.y_coords = None
        self.x_cen_coords = None
        self.y_cen_coords = None
        self.axes = None
        self.parameters = ()
        self.elements = ()
        self.spectra = ()
        self.wind = {}
        self.coord_system = WIND_COORD_TYPE_UNKNOWN
        self.gravitational_radius = -1

        # Get the conversion factors for distance and velocity units

        spatial_units = spatial_units.lower()
        if spatial_units not in [
                WIND_DISTANCE_UNITS_CM, WIND_DISTANCE_UNITS_M, WIND_DISTANCE_UNITS_KM, WIND_DISTANCE_UNITS_RG
        ]:
            raise ValueError(f"Unknown distance units {spatial_units}: allowed are {WIND_DISTANCE_UNITS_CM}, "
                             f"{WIND_DISTANCE_UNITS_M}, {WIND_DISTANCE_UNITS_KM} or {WIND_DISTANCE_UNITS_RG}")

        self._set_velocity_conversion_factor(velocity_units)

        # Now we can read in the different elements of the wind save tables and
        # initialize most of the variables above. If no files are found, then
        # windsave2table is automatically run. If that doesn't work, then the
        # will script raise an exception. It is also possible to force the
        # re-creation of the tables.

        if force_make_tables:
            self.create_wind_tables()

        try:
            self.get_all_tables(get_cell_spec, delim)
        except IOError:
            self.create_wind_tables()
            self.get_all_tables(get_cell_spec, delim)

        self.columns = self.wind.keys()

        if masked:
            self.create_masked_arrays()

        # Now we can convert or set the units, as the wind has been read in

        self.spatial_units = "unknown"

        if spatial_units == WIND_DISTANCE_UNITS_RG:
            self.convert_cm_to_rg(co_mass)  # self.spatial_units is set in here whenever we convert between units
        else:
            self.spatial_units = spatial_units

    # Private methods ----------------------------------------------------------

    def _get_element_variable(self, element_name, ion_name):
        """Helper function to get the fraction or density of an ion depending
        on the final character of the requested variable.

        Parameters
        ----------
        element_name: str
            The element symbol, i.e. H, He.
        ion_name: str
            The name of the element, in the format i_01, i_01f, i_01d, etc.
        """

        ion_frac_or_den = ion_name[-1]
        if not ion_frac_or_den.isdigit():
            ion_name = ion_name[:-1]
            if ion_frac_or_den == "d":
                variable = self.wind[element_name]["density"][ion_name]
            elif ion_frac_or_den == "f":
                variable = self.wind[element_name]["fraction"][ion_name]
            else:
                raise ValueError(f"{ion_frac_or_den} is an unknown ion type, try f or d")
        else:
            variable = self.wind[element_name]["density"][ion_name]

        return variable

    def _get_wind_coordinates(self):
        """Get coordinates of the wind.

        This returns the 2d array of coordinate points for the grid or
        1d depending on the coordinate type. This is different from
        using self.n_coords which returns only the axes points.
        """
        if self.coord_system == WIND_COORD_TYPE_SPHERICAL:
            n_points = self.wind["r"]
            m_points = np.zeros_like(n_points)
        elif self.coord_system == WIND_COORD_TYPE_CYLINDRICAL:
            n_points = self.wind["x"]
            m_points = self.wind["z"]
        else:
            m_points = np.log10(self.wind["r"])
            n_points = np.deg2rad(self.wind["theta"])

        return n_points, m_points

    def _get_wind_indices(self):
        """Get cell indices of the grid cells of the wind.

        This returns the 2d array of grid cell indices for the grid or
        1d depending on the coordinate type.
        """
        if self.coord_system == WIND_COORD_TYPE_SPHERICAL:
            n_points = self.wind["i"]
            m_points = np.zeros_like(n_points)
        elif self.coord_system == WIND_COORD_TYPE_CYLINDRICAL:
            n_points = self.wind["i"]
            m_points = self.wind["j"]
        else:
            raise ValueError("cannot plot with the cell indices for polar winds due to how matplotlib works")

        return n_points, m_points

    def _mask_ions_for_element(self, element_to_mask):
        """Create masked arrays for a single element.

        This acts as a wrapper function to reduce the number of indented for
        loops, improving readability of created_masked_arrays().

        Parameters
        ----------
        element_to_mask: str
            The name of the element to mask.
        """
        element = self.wind[element_to_mask]

        for ion_type in element.keys():
            for ion in element[ion_type]:
                element[ion_type][ion] = np.ma.masked_where(self.wind["inwind"] < 0, element[ion_type][ion])

        self.wind[element_to_mask] = element

    @staticmethod
    def _rename_j_to_j_bar(table, header):
        """Rename j, the mean intensity, to j_bar.

        In old versions of windsave2table, the mean intensity of the field
        was named j which created a conflict for 2D models which have a j
        cell index.

        Parameters
        ----------
        table: str
            The name of the table
        header: list[str]
            Rename the header for j to j_bar.
        """
        count = header.count("j")
        if count < 1:
            return header
        if count > 2:
            raise ValueError(f"unexpected format: too many j's ({count}) in header for {table} table")

        if "z" in header:
            if count == 1:  # This avoids new windsave2table where J -> j_bar
                return header
            idx = header.index("j") + 1
            idx += header[idx:].index("j")
            header[idx] = "j_bar"
        else:
            idx = header.index("j")
            header[idx] = "j_bar"

        return header

    def _set_velocity_conversion_factor(self, units):
        """Set the velocity conversion factor.

        Parameters
        ----------
        units: str
            The velocity units.
        """

        units = units.lower()
        if units not in [WIND_VELOCITY_UNITS_CMS, WIND_VELOCITY_UNITS_KMS, WIND_VELOCITY_UNITS_LIGHT]:
            raise ValueError(f"unknown velocity units {units}. Allowed units [{WIND_VELOCITY_UNITS_KMS}, "
                             f"{WIND_VELOCITY_UNITS_CMS}, {WIND_VELOCITY_UNITS_LIGHT}]")

        self.velocity_units = units

        if units == WIND_VELOCITY_UNITS_KMS:
            self.velocity_conversion_factor = CMS_TO_KMS
        elif units == WIND_VELOCITY_UNITS_CMS:
            self.velocity_conversion_factor = 1
        else:
            self.velocity_conversion_factor = 1 / VLIGHT

    def _setup_coords(self):
        """Set up the various coordinate variables."""

        # Setup the x/r coordinates

        if "x" in self.parameters:
            self.x_coords = tuple(np.unique(self.wind["x"]))
            self.x_cen_coords = tuple(np.unique(self.wind["xcen"]))
        else:
            self.x_coords = tuple(np.unique(self.wind["r"]))
            self.x_cen_coords = tuple(np.unique(self.wind["rcen"]))

        # Setup the z/theta coordinates

        if "z" in self.parameters:
            self.y_coords = tuple(np.unique(self.wind["z"]))
            self.y_cen_coords = tuple(np.unique(self.wind["zcen"]))
        elif "theta" in self.parameters:
            self.y_coords = tuple(np.unique(self.wind["theta"]))
            self.y_cen_coords = tuple(np.unique(self.wind["theta_cen"]))

        # Set up the coordinate system in used and the axes names available

        if self.nz == 1:
            self.coord_system = WIND_COORD_TYPE_SPHERICAL
            self.axes = ["r", "r_cen"]
        elif "r" in self.parameters and "theta" in self.parameters:
            self.coord_system = WIND_COORD_TYPE_POLAR
            self.axes = ["r", "theta", "r_cen", "theta_cen"]
        else:
            self.coord_system = WIND_COORD_TYPE_CYLINDRICAL
            self.axes = ["x", "z", "x_cen", "z_cen"]

    # Methods ------------------------------------------------------------------

    def convert_cm_to_rg(self, co_mass_in_msol=None):
        """Convert the spatial units from cm to gravitational radius.

        If the mass of the central source isn't supplied, then the parameter
        file will be searched for the mass.

        Parameters
        ----------
        co_mass_in_msol: float
            The mass of the central object in solar masses.

        Returns
        -------
        rg: float
            The gravitational radius for the model.
        """

        if self.spatial_units == WIND_DISTANCE_UNITS_RG:
            return

        if not co_mass_in_msol:
            try:
                co_mass_in_msol = float(get_parameter_value(self.pf, "Central_object.mass(msol)"))
            except Exception as e:
                print(e)
                raise ValueError("unable to find CO mass from parameter file, please supply the mass instead")

        rg = gravitational_radius(co_mass_in_msol)

        if self.coord_system in [WIND_COORD_TYPE_SPHERICAL, WIND_COORD_TYPE_POLAR]:
            self.wind["r"] /= rg
        else:
            self.wind["x"] /= rg
            self.wind["z"] /= rg

        self.spatial_units = WIND_DISTANCE_UNITS_RG
        self.gravitational_radius = rg
        self._setup_coords()

        return rg

    def convert_rg_to_cm(self, co_mass_in_msol=None):
        """Convert the spatial units from gravitational radius to cm.

        If the mass of the central source isn't supplied, then the parameter
        file will be searched for the mass.

        Parameters
        ----------
        co_mass_in_msol: float
            The mass of the central object in solar masses.

        Returns
        -------
        rg: float
            The gravitational radius for the model.
        """

        if self.spatial_units == WIND_DISTANCE_UNITS_CM:
            return

        if not co_mass_in_msol:
            try:
                co_mass_in_msol = float(get_parameter_value(self.pf, "Central_object.mass(msol)"))
            except Exception as e:
                print(e)
                raise ValueError("unable to find CO mass from parameter file, please supply the mass instead")

        rg = gravitational_radius(co_mass_in_msol)

        if self.coord_system == WIND_COORD_TYPE_SPHERICAL or self.coord_system == WIND_COORD_TYPE_POLAR:
            self.wind["r"] *= rg
        else:
            self.wind["x"] *= rg
            self.wind["z"] *= rg

        self.spatial_units = WIND_DISTANCE_UNITS_CM
        self.gravitational_radius = rg
        self._setup_coords()

        return rg

    def convert_velocity_units(self, units):
        """Convert the velocity units of the outflow.

        The velocity is first converted back into the base units of the model.
        Then the velocity conversion factor is set again and the units are
        converted using this.

        Parameters
        ----------
        units: str
            The new velocity units for the outflow.
        """
        print("warning: this function is currently untested")

        if units == self.velocity_units:
            return

        # Convert the units back into the base units, i.e. cm/s

        self.wind["v_x"] /= self.velocity_conversion_factor
        self.wind["v_y"] /= self.velocity_conversion_factor
        self.wind["v_z"] /= self.velocity_conversion_factor

        if self.coord_system == WIND_COORD_TYPE_CYLINDRICAL:
            self.wind["v_l"] /= self.velocity_conversion_factor
            self.wind["v_rot"] /= self.velocity_conversion_factor
            self.wind["v_r"] /= self.velocity_conversion_factor

        # Set the new unit conversion scale

        self._set_velocity_conversion_factor(units)

        # Now convert into the new units requested

        self.wind["v_x"] *= self.velocity_conversion_factor
        self.wind["v_y"] *= self.velocity_conversion_factor
        self.wind["v_z"] *= self.velocity_conversion_factor

        if self.coord_system == WIND_COORD_TYPE_CYLINDRICAL:
            self.wind["v_l"] *= self.velocity_conversion_factor
            self.wind["v_rot"] *= self.velocity_conversion_factor
            self.wind["v_r"] *= self.velocity_conversion_factor

    def create_masked_arrays(self):
        """Mask cells which are not in the wind.

        Convert each array into a masked array using the in-wind. This
        is helpful when using pcolormesh, as matplotlib will ignore
        masked out cells so there will be no background colour to a
        color plot.
        """
        to_mask_wind = list(self.parameters)

        # Create masked array for wind parameters
        # Remove some of the columns from the standard wind parameters, as these
        # shouldn't be masked otherwise weird things will happen

        for item_to_remove in [
                "x", "z", "r", "theta", "xcen", "zcen", "rcen", "theta_cen", "i", "j", "inwind", "cell_spec"
        ]:
            try:
                to_mask_wind.remove(item_to_remove)
            except ValueError:  # sometimes a key wont exist and this catches it
                continue

        for col in to_mask_wind:
            self.wind[col] = np.ma.masked_where(self.wind["inwind"] < 0, self.wind[col])

        # Create masked arrays for the wind ions, loop over each element read in

        for element in self.elements:
            self._mask_ions_for_element(element)

        # Create masked array for cell and model spectra, but only in a special
        # mode set by a global variable lol

        if DEBUG_MASK_CELL_SPEC:
            for cell_spec_type in self.spectra:
                self.wind[cell_spec_type].spectra = np.ma.masked_where(self.wind["inwind"] < 0,
                                                                       self.wind[cell_spec_type].spectra)

    def create_wind_tables(self):
        """Force the creation of wind save tables for the model.

        This is best used when a simulation has been re-run, as the
        library is unable to detect when the currently available wind
        tables do not reflect a new simulation. This function will
        create the standard wind tables, as well as the fractional and
        density ion tables and create the xspec cell spectra files.
        """

        create_wind_save_tables(self.root, self.fp, ion_density=True, version=self.version)
        create_wind_save_tables(self.root, self.fp, ion_density=False, version=self.version)
        create_wind_save_tables(self.root, self.fp, cell_spec=True, version=self.version)

    def get(self, parameter):
        """Get a parameter array. This is just another way to access the
        dictionary self.variables and is a nice wrapper around getting ion
        fractions of densities.

        To get an ion fraction or density use, e.g., C_i04f or C_i04d
        respectively.

        Parameters
        ----------
        parameter: str
            The name of the parameter to get.
        """
        element_name = parameter[:2].replace("_", "")
        ion_name = parameter[2:].replace("_", "")

        if element_name in self.elements:
            variable = self._get_element_variable(element_name, ion_name)
        else:
            variable = self.wind[parameter]

        return variable

    def get_all_tables(self, get_cell_spec, delim):
        """Read all the wind save tables into the object.

        Parameters
        ----------
        get_cell_spec: bool
            Also read in the cell spectra. In old version of Python, these were
            not available.
        delim: str
            The file delimiter.
        """
        self.get_wind_parameters(delim)
        self.get_wind_elements(delim=delim)
        if get_cell_spec:
            try:  # In case someone asks for cell spec when we can't get them
                self.wind["cell_spec"] = CellSpectra(self.root, self.fp, self.nx, self.nz)
                self.wind["cell_model"] = ModelledCellSpectra(self.root, self.fp)
                self.spectra += ("cell_spec", "cell_model")
            except ValueError:
                print(f"unable to load cell or model spectra for model {self.fp}{self.root} as windsave2table is too "
                      f"old")

    def get_elem_number_from_ij(self, i, j):
        """Get the wind element number for a given i and j index.

        Used when indexing into a 1D array, such as in Python itself.

        Parameters
        ----------
        i: int
            The i-th index of the cell.
        j: int
            The j-th index of the cell.
        """
        return self.nz * i + j

    def get_ij_from_elem_number(self, elem):
        """Get the i and j index for a given wind element number.

        Used when converting a wind element number into two indices for use
        in this package.

        Parameters
        ----------
        elem: int
            The element number.
        """
        return np.unravel_index(elem, (self.nx, self.nz))

    def get_sight_line_coordinates(self, theta):
        """Get the vertical z coordinates for a given set of x coordinates and
        inclination angle.

        Parameters
        ----------
        theta: float
            The angle of the sight line to extract from. Given in degrees.
        """
        return np.array(self.x_coords, dtype=np.float64) * np.tan(PI / 2 - np.deg2rad(theta))

    def get_variable_along_sight_line(self, theta, parameter):
        """Extract a variable along a given sight line.

        Parameters
        ----------
        theta: float
            The angle to extract along.
        parameter: str
            The parameter to extract.
        """
        if self.coord_system == WIND_COORD_TYPE_POLAR:
            raise NotImplementedError("This hasn't been implemented for polar winds, lol")

        if type(theta) is not float:
            theta = float(theta)

        z_array = np.array(self.y_coords, dtype=np.float64)
        z_coords = self.get_sight_line_coordinates(theta)

        values = np.zeros_like(z_coords, dtype=np.float64)
        w_array = self.get(parameter)

        for x_index, z in enumerate(z_coords):  # todo: I wonder if this can be vectorized
            z_index = get_array_index(z_array, z)
            values[x_index] = w_array[x_index, z_index]

        return np.array(self.x_coords), z_array, values

    def get_wind_elements(self, elements_to_get=None, delim=None):
        """Read in the ion parameters.

        Reads each element in and its ions into a dictionary. This function will
        try to read in both ion fractions and densities.

        Each element will have a dict of two keys, either fraction or density.
        Inside each dict with be more dicts of keys of the available ions for
        that element.

        Parameters
        ----------
        elements_to_get: List[str] or Tuple[str]
            The elements to read ions in for.
        delim: str [optional]
            The file delimiter.
        """
        if elements_to_get is None:
            elements_to_get = ("H", "He", "C", "N", "O", "Si", "Fe")
        else:
            if type(elements_to_get) not in [str, list, tuple]:
                raise TypeError("ions_to_get should be a tuple/list of strings or a string")

        # Loop over each element wind table

        n_elements_read = 0

        for element in elements_to_get:
            element = element.capitalize()
            self.wind[element] = {}

            # Loop over the different ion "types", i.e. frac or den.
            # ion_type_index_name is used to give the name for the keys in the
            # dictionary as frac and den is used in python, but I prefer to use
            # fraction and density

            for ion_type, ion_type_index_name in zip(["frac", "den"], ["fraction", "density"]):

                fp = self.fp + self.root + "." + element + "." + ion_type + ".txt"

                if not path.exists(fp):
                    fp = self.fp + "tables/" + self.root + "." + element + "." + ion_type + ".txt"
                    if not path.exists(fp):
                        continue

                n_elements_read += 1

                if element not in self.elements:
                    self.elements += element,

                with open(fp, "r") as f:
                    ion_file = f.readlines()

                wind_lines = []

                for line in ion_file:
                    if delim:
                        line = line.split(delim)
                    else:
                        line = line.split()
                    if len(line) == 0 or line[0] == "#":
                        continue
                    wind_lines.append(line)

                # Now construct the dict of ions, in the layout as described
                # in the doc string. First, we have to find out where the output
                # we actually wants start

                if wind_lines[0][0].isdigit() is False:
                    columns = tuple(wind_lines[0])
                    index = columns.index("i01")
                else:
                    columns = tuple(np.arange(len(wind_lines[0]), dtype=np.dtype.str))
                    index = 0

                columns = columns[index:]
                wind_lines = np.array(wind_lines[1:], dtype=np.float64)[:, index:]

                # Now we can populate the dictionary with the different columns
                # of the file

                self.wind[element][ion_type_index_name] = {}
                for index, col in enumerate(columns):
                    self.wind[element][ion_type_index_name][col] = wind_lines[:, index].reshape(self.nx, self.nz)

        if n_elements_read == 0 and len(self.columns) == 0:
            raise IOError(
                "Unable to open any parameter or ion tables: try running windsave2table with the correct version")

    def get_wind_parameters(self, delim=None):
        """Read in the wind parameters.

        This reads in the master, heat, gradient, converge and spec file into
        a dictionary. Each header of the tables is a key in the dictionary.

        Parameters
        ----------
        delim: str [optional]
            The deliminator in the wind table files.
        """
        wind_all = []
        wind_columns = []

        # Read in each file, one by one, if they exist. This makes the
        # assumption that all the tables are the same size.

        n_read = 0
        files_to_read = ["master", "heat", "gradient", "converge"]

        for table in files_to_read:
            fp = self.fp + self.root + "." + table + ".txt"
            if not path.exists(fp):
                fp = self.fp + "tables/" + self.root + "." + table + ".txt"
                if not path.exists(fp):
                    continue
            n_read += 1

            with open(fp, "r") as f:
                wind_file = f.readlines()

            # todo: need some method to detect incorrect syntax

            wind_list = []

            for line in wind_file:
                line = line.strip()
                if delim:
                    line = line.split(delim)
                else:
                    line = line.split()
                if len(line) == 0 or line[0] == "#":
                    continue
                wind_list.append(line)

            # Keep track of each file header and add the wind lines for the
            # current file into wind_all, the list of lists, the master list and
            # Add the wind parameters to the wind_all list, but not the
            # header

            if wind_list[0][0].isdigit() is False:
                header = wind_list[0]
                if table == "heat":
                    header = self._rename_j_to_j_bar(table, header)
                wind_columns += header
            else:
                wind_columns += list(np.arange(len(wind_list[0]), dtype=np.dtype.str))

            wind_all.append(np.array(wind_list[1:], dtype=np.float64))

        if n_read == 0:
            raise IOError(f"Unable to open any wind tables for root {self.root} directory {self.fp}")

        # Determine the number of nx and nz elements. There is a basic check to
        # only check for nz if a 'j' column exists, i.e. if it is a 2d model.

        i_col = wind_columns.index("i")
        self.nx = int(np.max(wind_all[0][:, i_col]) + 1)

        if "z" in wind_columns or "theta" in wind_columns:
            j_col = wind_columns.index("j")
            self.nz = int(np.max(wind_all[0][:, j_col]) + 1)
        self.n_elem = int(self.nx * self.nz)

        # Now we join the different wind tables together and create a dictionary
        # of all of the parameters. We also can now set up the list of available
        # parameters and setup the coordinate parameters too

        wind_all = np.hstack(wind_all)

        for index, col in enumerate(wind_columns):
            if col in self.wind.keys():
                continue
            self.wind[col] = wind_all[:, index].reshape(self.nx, self.nz)

        self.parameters = tuple(self.wind.keys())
        self._setup_coords()

        # Convert velocity into desired units and also calculate the cylindrical
        # velocities.

        if self.coord_system == WIND_COORD_TYPE_CYLINDRICAL:
            self.project_cartesian_velocity_to_cylindrical()

        self.wind["v_x"] *= self.velocity_conversion_factor
        self.wind["v_y"] *= self.velocity_conversion_factor
        self.wind["v_z"] *= self.velocity_conversion_factor

    def plot(self, variable_name, use_cell_coordinates=True, scale="loglog", log_variable=True):
        """Create a plot of the wind for the given variable.

        Only one thing can be plotted at once, in their own figure window. More
        advanced plotting things can be found in pypython.plot.wind, or write
        something yourself.

        Parameters
        ----------
        variable_name: str
            The name of the variable to plot. Ions are accessed as, i.e.,
            H_i01, He_i02, etc.
        use_cell_coordinates: bool [optional]
            Plot using the cell coordinates instead of cell index numbers
        scale: str [optional]
            The type of scaling for the axes
        log_variable: bool [optional]
            Plot the variable in logarithmic units
        """
        variable = self.get(variable_name)

        if use_cell_coordinates:
            n_points, m_points = self._get_wind_coordinates()
        else:
            n_points, m_points = self._get_wind_indices()

        if self.coord_system == "spherical":
            fig, ax = plot_1d_wind(n_points, variable, self.spatial_units, scale=scale)
        else:
            if log_variable:
                variable = np.log10(variable)
            fig, ax = plot_2d_wind(n_points, m_points, variable, self.spatial_units, self.coord_system, scale=scale)

        if len(ax) == 1:
            ax = ax[0, 0]
            title = f"{variable_name}".replace("_", " ")
            if self.coord_system == "spherical":
                ax.set_ylabel(title)
            else:
                ax.set_title(title)

        return fig, ax

    def plot_cell_spec(self, i, j, energetic=False):
        """Plot the spectrum for a cell.

        This will plot the cell spectrum and the model of that spectrum used
        internally by Python to calculate the heating/cooling rates.

        Parameters
        ----------
        i: int
            The i-th index for the cell.
        j: int
            The j-th index for the cell.
        energetic: bool
            Plot in energetic (nu * J_nu) units.
        """
        if not self.spectra:
            raise ValueError("no cell spectra have been read in")

        fig, ax = self.wind["cell_spec"].plot(i, j, energetic)
        fig, ax = self.wind["cell_model"].plot(i, j, energetic, fig=fig, ax=ax)
        ax.legend()

        return fig, ax

    def project_cartesian_velocity_to_cylindrical(self):
        """Project cartesian velocities into cylindrical velocities.

        This makes the variables v_r, v_rot and v_l available in
        variables dictionary. Only works for cylindrical coordinates
        systems, which outputs the velocities in cartesian coordinates.
        """
        v_l = np.zeros_like(self.wind["v_x"])
        v_rot = np.zeros_like(v_l)
        v_r = np.zeros_like(v_l)
        n1, n2 = v_l.shape

        for i in range(n1):
            for j in range(n2):
                cart_point = np.array([self.wind["x"][i, j], 0, self.wind["z"][i, j]])
                # todo: don't think I need to do this check anymore
                if self.wind["inwind"][i, j] < 0:
                    v_l[i, j] = 0
                    v_rot[i, j] = 0
                    v_r[i, j] = 0
                else:
                    cart_velocity_vector = np.array(
                        [self.wind["v_x"][i, j], self.wind["v_y"][i, j], self.wind["v_z"][i, j]])
                    cyl_velocity_vector = vector.project_cartesian_to_cylindrical_coordinates(
                        cart_point, cart_velocity_vector)
                    if type(cyl_velocity_vector) is int:
                        continue
                    v_l[i, j] = np.sqrt(cyl_velocity_vector[0]**2 + cyl_velocity_vector[2]**2)
                    v_rot[i, j] = cyl_velocity_vector[1]
                    v_r[i, j] = cyl_velocity_vector[0]

        self.wind["v_l"] = v_l * self.velocity_conversion_factor
        self.wind["v_rot"] = v_rot * self.velocity_conversion_factor
        self.wind["v_r"] = v_r * self.velocity_conversion_factor

    @staticmethod
    def show(block=True):
        """Show a plot which has been created.

        Wrapper around pyplot.show().

        Parameters
        ----------
        block: bool
            Use blocking or non-blocking figure display.
        """
        plt.show(block=block)

    # Built in stuff -----------------------------------------------------------

    def __getitem__(self, key):
        return self.wind[key]

    def __setitem__(self, key, value):
        self.wind[key] = value

    def __str__(self):
        txt = f"root: {self.root}\nfilepath: {self.fp}\ncoordinate system: {self.coord_system}\n" \
              f"parameters: {self.parameters}\nelements: {self.elements}\nspectra: {self.spectra}"

        return textwrap.dedent(txt)


# These are put here to solve a circular dependency ----------------------------

import pypython.math
import pypython.spectrum
import pypython.physics
import pypython.plot
import pypython.util

from pypython.plot import normalize_figure_style, set_axes_scales
from pypython.plot.spectrum import add_line_ids, common_lines
from pypython.plot.wind import plot_1d_wind, plot_2d_wind
from pypython.util import run_command


# Create the __all__ list of modules -------------------------------------------

# __all__ = [
#     "math", "physics", "plot", "simulation", "spectrum", "util", "constants", "error", "check_sorted_array_ascending",
#     "cleanup_data", "create_wind_save_tables", "find", "get_array_index", "get_python_version", "get_root_name",
#     "smooth_array", "run_py_optical_depth", "run_py_wind", "CellSpectra", "ModelledCellSpectra", "Spectrum", "Wind"
# ]
