#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import textwrap
import time
from os import path, listdir, remove
from pathlib import Path
from platform import system
from shutil import which
from subprocess import PIPE, Popen
from typing import List, Tuple, Union
import copy
from matplotlib import pyplot as plt
from enum import Enum

import numpy as np
from scipy.signal import boxcar, convolve

from pypython.math import vector
from pypython.plot import normalize_figure_style, ax_add_line_ids, common_lines
from pypython.physics.constants import CMS_TO_KMS, VLIGHT, PI
from pypython.plot.wind import plot_1d_wind, plot_2d_wind

name = "pypython"


# Spectrum class --------------


UNITS_LNU = "erg/s/Hz"
UNITS_FNU = "erg/s/cm^-2/Hz"
UNITS_FLAMBDA = "erg/s/cm^-2/A"


class Spectrum:
    """A class to store PYTHON .spec and .log_spec files.
    The PYTHON spectrum is read in and stored within a dict, where each column
    name is a key and the data is stored as a numpy array."""
    def __init__(self,
                 root: str,
                 cd: str = ".",
                 default: str = None,
                 log: bool = False,
                 smooth: int = None,
                 delim: str = None):
        """Initialise a Spectrum object. This method will construct the file path
        of the spectrum file given the root, containing directory and whether
        the logarithmic spectrum is used or not. The spectrum is then read in.

        Parameters
        ----------
        root: str
            The root name of the model.
        cd: str [optional]
            The directory containing the model.
        default: str [optional]
            The default spectrum to make the available spectrum for indexing.
        log: bool [optional]
            Read in the logarithmic spectrum.
        smooth: int [optional]
            The amount of smoothing to use.
        delim: str [optional]
            The deliminator in the spectrum file.
        """

        self.root = root

        self.fp = cd
        self.logspec = log
        if log and not default.startswith("log_"):
            default = "log_" + default
        if self.fp[-1] != "/":
            self.fp += "/"

        self.all_spectrum = {}
        self.all_columns = {}
        self.all_inclinations = {}
        self.all_n_inclinations = {}
        self.all_units = {}

        # self.unsmoothed is a variable which keeps a copy of the spectrum for
        # safe keeping if it is smoothed

        self.unsmoothed = None

        # The next method call reads in the spectrum and initializes the above
        # member variables. We also keep track of what spectra have been loaded
        # in and set the "target" spectrum for indexing

        self.read_in_spectra(delim)
        self.available = tuple(self.all_spectrum.keys())

        # Now set the units, etc., to the target spectrum. If default is
        # provided, then this is used as the default spectrum.

        if default:
            if default in self.available:
                self.current = default
            else:
                raise ValueError(
                    f"{self.root}.{default} is not available as it has not been read in"
                )
        else:
            self.current = self.available[0]

        self.spectrum = self.all_spectrum[self.current]
        self.columns = self.all_columns[self.current]
        self.inclinations = self.all_inclinations[self.current]
        self.n_inclinations = self.all_n_inclinations[self.current]
        self.units = self.all_units[self.current]

        if smooth:
            self.smooth(smooth)

    def read_in_spectra(self, delim: str = None):
        """Read in a spectrum file given in self.filepath. The spectrum is stored
        as a dictionary in self.spectrum where each key is the name of the
        columns.

        Parameters
        ----------
        delim: str [optional]
            A custom delimiter, useful for reading in files which have sometimes
            between delimited with commas instead of spaces."""

        n_read = 0
        files_to_read = [
            "spec", "spec_tot", "spec_tot_wind", "spec_wind", "spec_tau"
        ]

        for spec_type in files_to_read:
            fpath = self.fp + self.root + "."
            if self.logspec and spec_type != "spec_tau":
                fpath += "log_"
            fpath += spec_type

            if not path.exists(fpath):
                continue

            n_read += 1
            self.all_spectrum[spec_type] = {}
            self.all_units[spec_type] = "unknown"

            with open(fpath, "r") as f:
                spectrum_file = f.readlines()

            # Read in the spectrum file, ignoring empty lines and lines which have
            # been commented out by # at the beginning
            # todo: need some method to detect incorrect syntax

            spectrum = []

            for line in spectrum_file:
                line = line.strip()
                if delim:
                    line = line.split(delim)
                else:
                    line = line.split()
                if "Units:" in line:
                    self.all_units[spec_type] = line[4][1:-1]
                if len(line) == 0 or line[0] == "#":
                    continue
                spectrum.append(line)

            # Extract the header columns of the spectrum. This assumes the first
            # read line in the spectrum is the header. If no header is found, then
            # the columns are numbered instead

            header = []

            if spectrum[0][0] == "Freq." or spectrum[0][0] == "Lambda":
                for i, column_name in enumerate(spectrum[0]):
                    if column_name[0] == "A":
                        j = column_name.find("P")
                        column_name = column_name[1:j]
                    header.append(column_name)
                spectrum = np.array(spectrum[1:], dtype=np.float)
            else:
                header = np.arange(len(spectrum[0]))

            # Add the actual spectrum to the spectrum dictionary, the keys of the
            # dictionary are the column names as given above. Set the header and
            # also the inclination angles here as well

            for i, column_name in enumerate(header):
                self.all_spectrum[spec_type][column_name] = spectrum[:, i]

            inclinations = []

            for col in header:
                if col.isdigit() and col not in inclinations:
                    inclinations.append(col)

            self.all_columns[spec_type] = tuple(header)
            self.all_inclinations[spec_type] = tuple(inclinations)
            self.all_n_inclinations[spec_type] = len(inclinations)

        if n_read == 0:
            raise IOError(f"Unable to open any spectrum files in {self.fp}")

    def smooth(self,
               width: int = 5,
               to_smooth: Union[List[str], Tuple[str], str] = None):
        """Smooth the spectrum flux/luminosity bins.

        Parameters
        ----------
        width: int [optional]
            The width of the boxcar filter (in bins).
        to_smooth: list or tuple of strings [optional]
            A list or tuple"""

        # Create a backup of the unsmoothed array before it is smoothed it

        if self.unsmoothed is None:
            self.unsmoothed = copy.deepcopy(self.spectrum)

        # Get the input parameters for smoothing and make sure it's good input

        if type(width) is not int:
            try:
                width = int(width)
            except ValueError:
                print(f"Unable to cast {width} into an int")
                return

        if to_smooth is None:
            to_smooth = ("Created", "WCreated", "Emitted", "CenSrc", "Disk",
                         "Wind", "HitSurf", "Scattered") + tuple(
                             self.inclinations)
        elif type(to_smooth) is str:
            to_smooth = to_smooth,
        else:
            raise ValueError(
                "unknown format for to_smooth, must be a tuple of strings or string"
            )

        # Loop over each available spectrum and smooth it

        for key in self.available:
            for thing_to_smooth in to_smooth:
                try:
                    self.spectrum[key][thing_to_smooth] = convolve(
                        self.spectrum[key][thing_to_smooth],
                        boxcar(width) / float(width),
                        mode="same")
                except KeyError:
                    continue

    def unsmooth(self):
        """Restore the spectrum to its unsmoothed form."""

        self.spectrum = copy.deepcopy(self.unsmoothed)

    def _plot_specific(self,
                       name: str,
                       label_lines: bool = False,
                       ax_update: plt.Axes = None):
        """Plot a specific column in a spectrum file.

        Parameters
        ----------
        label_lines: bool
            Plot line IDs.
        ax_update: plt.Axes
            An plt.Axes object to update, i.e. to plot on."""

        normalize_figure_style()

        if not ax_update:
            fig, ax = plt.subplots(figsize=(9, 5))
        else:
            ax = ax_update

        ax.set_yscale("log")
        ax.set_xscale("log")

        if self.units == UNITS_FLAMBDA:
            ax.plot(self.spectrum["Lambda"], self.spectrum[name], label=name)
            ax.set_xlabel(r"Wavelength [\AA]")
            ax.set_ylabel(
                r"Flux Density 100 pc [erg s$^{-1}$ cm$^{-2}$ \AA$^{-1}$]")
            if label_lines:
                ax = ax_add_line_ids(ax, common_lines(False), logx=True)
        else:
            ax.plot(self.spectrum["Freq."], self.spectrum[name], label=name)
            ax.set_xlabel("Frequency [Hz]")
            if self.units == UNITS_LNU:
                ax.set_ylabel(r"Luminosity 100 pc [erg s$^{-1}$ Hz$^{-1}$]")
            else:
                ax.set_ylabel(
                    r"Flux Density 100 pc [erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$]")
            if label_lines:
                ax = ax_add_line_ids(ax, common_lines(True), logx=True)

        if not ax_update:
            fig.tight_layout(rect=[0.015, 0.015, 0.985, 0.985])
            return fig, ax
        else:
            return ax

    def _spec_plot_all(self, label_lines: bool = False):
        """Plot the spectrum components and observer spectra on a 1x2 panel
        plot. The left panel has the components, whilst the right panel has
        the observer spectrum.

        Parameters
        ----------
        label_lines: bool
            Plot line IDs."""

        normalize_figure_style()

        fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharey="row")

        for component in self.columns[:-self.n_inclinations]:
            if component in ["Lambda", "Freq."]:
                continue
            ax[0] = self._plot_specific(component, label_lines, ax[0])

        for line in ax[0].get_lines():
            line.set_alpha(0.7)
        ax[0].legend(ncol=2, loc="upper right").set_zorder(0)

        for inclination in self.inclinations:
            ax[1] = self._plot_specific(inclination, label_lines, ax[1])

        for label, line in zip(self.inclinations, ax[1].get_lines()):
            line.set_alpha(0.7)
            line.set_label(str(label) + r"$^{\circ}$")
        ax[1].set_ylabel("")
        ax[1].legend(ncol=2, loc="upper right").set_zorder(0)

        ax[0].set_title("Components")
        ax[1].set_title("Observer spectra")

        fig.tight_layout(rect=[0.015, 0.015, 0.985, 0.985])
        fig.subplots_adjust(wspace=0)

        return fig, ax

    def plot(self, name: str = None, label_lines: bool = False):
        """Plot the spectra or a single component in a single figure. By default
        this creates a 1 x 2 of the components on the left and the observer
        spectra on the right. Useful for when in an interactive session.

        Parameters
        ----------
        name: str
            The name of the thing to plot.
        label_lines: bool
            Plot line IDs."""

        # todo:
        # This is some badness inspired by Python. This is done, for now, as
        # I haven't implemented a way to plot other spectra quickly this way

        ot = self.current
        self.current = "spec"

        if name:
            if name not in self.columns:
                print(f"{name} is not in the spectrum columns")
                return
            fig, ax = self._plot_specific(name, label_lines)
            if name.isdigit():
                name += r"$^{\circ}$"
            ax.set_title(name.replace("_", r"\_"))
        else:
            # todo: update with more functions to plot spec_tot w/o name etc
            if "spec" not in self.available and "log_spec" not in self.available:
                raise IOError(
                    f"Unable to plot without parameter 'name' as there is no {self.root}.spec file"
                )
            fig, ax = self._spec_plot_all(label_lines)

        self.current = ot

        return fig, ax

    def show(self, block=True):
        """Show any plots which have been generated."""

        plt.show(block=block)

    def set(self, name):
        """Set a different spectrum to be the target."""

        if self.logspec and not name.startswith("log_"):
            name = "log_" + name

        if name not in self.available:
            raise ValueError(
                f"Spectrum {name} is not available: available {self.available}"
            )

        self.current = name
        self.spectrum = self.all_spectrum[self.current]
        self.columns = self.all_columns[self.current]
        self.inclinations = self.all_inclinations[self.current]
        self.n_inclinations = self.all_n_inclinations[self.current]
        self.units = self.all_units[self.current]

    def __getitem__(self, key):
        """Return an array in the spectrum dictionary when indexing."""

        if key not in self.available:
            return self.spectrum[key]
        else:
            return self.all_spectrum[key]

    def __setitem__(self, key, value):
        """Allows to modify the arrays in the spectrum dictionary."""

        if key not in self.available:
            self.spectrum[key] = value
        else:
            self.all_spectrum[key] = value

    def __str__(self):
        """Print the basic details about the spectrum."""

        msg = f"Spectrum for the model {self.root} in {self.fp}\n"
        msg += f"Available spectra: {self.available}\n"
        msg += f"Current spectrum {self.current}\n"
        if "spec" in self.available or "log_spec" in self.available:
            msg += f"Spectrum inclinations: {self.inclinations['spec']}\n"
        if "tau_spec" in self.available:
            msg += f"Optical depth inclinations {self.inclinations['tau_spec']}\n"

        return textwrap.dedent(msg)


# Wind class ------------------

class Wind:
    """A class to store 1D and 2D Python wind tables. Contains methods to
    extract variables, as well as convert various indices into other indices.
    todo: add dot notation for accessing dictionaries.
    """
    def __init__(self,
                 root: str,
                 cd: str = ".",
                 velocity_units: str = "kms",
                 mask: bool = True,
                 delim: str = None):
        """Initialize the Wind object.

        Parameters
        ----------
        root: str
            The root name of the Python simulation.
        cd: str
            The directory containing the model.
        mask: bool [optional]
            Store the wind parameters as masked arrays.
        delim: str [optional]
            The delimiter used in the wind table files.
        """

        # class CoordSystem(Enum):
        #     SPHERICAL = 1
        #     POLAR = 2
        #     RECTILINEAR = 3

        self.root = root
        self.fp = cd
        if self.fp[-1] != "/":
            self.fp += "/"
        self.nx = 1
        self.nz = 1
        self.n_elem = 1
        self.m_coords = ()
        self.n_coords = ()
        self.m_cen_coords = ()
        self.n_cen_coords = ()
        self.axes = ()
        self.parameters = ()
        self.elements = ()
        self.variables = {}
        self.coord_system = "unknown"

        # Set up the velocity units and conversion factors

        if velocity_units not in ["cms", "kms", "c"]:
            print(
                f"unknown velocity units {velocity_units}. Allowed units [kms, cms, c]"
            )
            exit(1)

        self.velocity_units = velocity_units
        if velocity_units == "kms":
            self.velocity_conversion_factor = CMS_TO_KMS
        elif velocity_units == "cms":
            self.velocity_conversion_factor = 1
        else:
            self.velocity_conversion_factor = 1 / VLIGHT

        # The next method reads in the wind and initializes the above members.
        # If no wind tables can be found in read_in_wind_parameters, an IOError
        # is raised. If raised, try to create the wind table and read the
        # wind parameters again

        try:
            self.read_in_wind_parameters(delim)
        except IOError:
            print("trying to run windsave2table to generate wind tables")
            create_wind_save_tables(self.root, self.fp, ion_density=True)
            create_wind_save_tables(self.root, self.fp, ion_density=False)
            self.read_in_wind_parameters(delim)
        self.read_in_wind_ions(delim)
        self.columns = self.parameters + self.elements

        # Convert velocity into desired units and also calculate the cylindrical
        # velocities. This doesn't work for polar or spherical coordinates as
        # they will not have these velocities

        if self.coord_system == "rectilinear":
            self.project_cartesian_velocity_to_cylindrical()

        self.variables["v_x"] *= self.velocity_conversion_factor
        self.variables["v_y"] *= self.velocity_conversion_factor
        self.variables["v_z"] *= self.velocity_conversion_factor

        # Create masked cells, if that's the users deepest desire for their
        # data

        if mask:
            self.create_masked_arrays()

    def read_in_wind_parameters(self, delim: str = None):
        """Read in the wind parameters.

        Parameters
        ----------
        delim: str [optional]
            The deliminator in the wind table files.
        """
        wind_all = []
        wind_columns = []

        # Read in each file, one by one, if they exist. Note that this makes
        # the assumption that all the tables are the same size.

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

            # Read in the wind_save table, ignoring empty lines and comments.
            # Each file is stored as a list of lines within a list, so a list
            # of lists.
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
            # current file into wind_all, the list of lists, the master list

            if wind_list[0][0].isdigit() is False:
                wind_columns += wind_list[0]
            else:
                wind_columns += list(
                    np.arrange(len(wind_list[0]), dtype=np.str))

            wind_all.append(np.array(wind_list[1:], dtype=np.float64))

        if n_read == 0:
            raise IOError(
                f"Unable to open any wind tables for root {self.root} directory {self.fp}"
            )

        # Determine the number of nx and nz elements. There is a basic check to
        # only check for nz if a j column exists, i.e. if it is a 2d model.

        i_col = wind_columns.index("i")
        self.nx = int(np.max(wind_all[0][:, i_col]) + 1)

        if "z" in wind_columns or "theta" in wind_columns:
            j_col = wind_columns.index("j")
            self.nz = int(np.max(wind_all[0][:, j_col]) + 1)
        self.n_elem = int(self.nx * self.nz)  # the int() is for safety

        wind_all = np.hstack(wind_all)

        # Assign each column header to a key in the dictionary, ignoring any
        # column which is already in the dict and extract the x and z
        # coordinates

        for index, col in enumerate(wind_columns):
            if col in self.variables.keys():
                continue
            self.variables[col] = wind_all[:, index].reshape(self.nx, self.nz)
            self.parameters += col,

        # Get the x/r coordinates

        if "x" in self.parameters:
            self.m_coords = tuple(np.unique(self.variables["x"]))
            self.m_cen_coords = tuple(np.unique(self.variables["xcen"]))
        else:
            self.m_coords = tuple(np.unique(self.variables["r"]))
            self.m_cen_coords = tuple(np.unique(self.variables["rcen"]))

        # Get the z/theta coordinates

        if "z" in self.parameters:
            self.n_coords = tuple(np.unique(self.variables["z"]))
            self.n_cen_coords = tuple(np.unique(self.variables["zcen"]))
        elif "theta" in self.parameters:
            self.n_coords = tuple(np.unique(self.variables["theta"]))
            self.n_cen_coords = tuple(np.unique(self.variables["theta_cen"]))

        # Record the coordinate system and the axes labels

        if self.nz == 1:
            self.coord_system = "spherical"
            self.axes = ["r", "r_cen"]
        elif "r" in self.parameters and "theta" in self.parameters:
            self.coord_system = "polar"
            self.axes = ["r", "theta", "r_cen", "theta_cen"]
        else:
            self.coord_system = "rectilinear"
            self.axes = ["x", "z", "x_cen", "z_cen"]

    def read_in_wind_ions(self,
                          delim: str = None,
                          elements_to_get: Union[List[str], Tuple[str],
                                                 str] = None):
        """Read in the ion parameters.

        Parameters
        ----------
        delim: str [optional]
            The file delimiter.
        elements_to_get: List[str] or Tuple[str]
            The elements to read ions in for.
        """

        if elements_to_get is None:
            elements_to_get = ("H", "He", "C", "N", "O", "Si", "Fe")
        else:
            if type(elements_to_get) not in [str, list, tuple]:
                print(
                    "ions_to_get should be a tuple/list of strings or a string"
                )
                exit(1)

        # Read in each ion file, one by one. The ions will be stored in the
        # self.variables dict as,
        # key = ion name
        # values = dict of ion keys, i.e. i_01, i_02, etc, and the values
        # in this dict will be the values of that ion

        ion_types_to_get = ["frac", "den"]
        ion_types_index_names = ["fraction", "density"]

        n_elements_read = 0

        for element in elements_to_get:
            element = element.capitalize()  # for safety...
            self.elements += element,

            # Each element will have a dict of two keys, either frac or den.
            # Inside each dict with be more dicts of keys where the values are
            # arrays of the

            self.variables[element] = {}

            for ion_type, ion_type_index_name in zip(ion_types_to_get,
                                                     ion_types_index_names):
                fp = self.fp + self.root + "." + element + "." + ion_type + ".txt"
                if not path.exists(fp):
                    fp = self.fp + "tables/" + self.root + "." + element + "." + ion_type + ".txt"
                    if not path.exists(fp):
                        continue
                n_elements_read += 1
                with open(fp, "r") as f:
                    ion_file = f.readlines()

                # Read in ion the ion file. this can be done in a list
                # comprehension, I think, but I want to skip commented out lines
                # and I think it's better(?) to do it this way

                wind = []

                for line in ion_file:
                    if delim:
                        line = line.split(delim)
                    else:
                        line = line.split()
                    if len(line) == 0 or line[0] == "#":
                        continue
                    wind.append(line)

                # Now construct the tables, how this is done is described in
                # some of the comments above

                if wind[0][0].isdigit() is False:
                    columns = tuple(wind[0])
                    index = columns.index("i01")
                else:
                    columns = tuple(np.arrange(len(wind[0]), dtype=np.str))
                    index = 0
                columns = columns[index:]
                wind = np.array(wind[1:], dtype=np.float64)[:, index:]

                self.variables[element][ion_type_index_name] = {}
                for index, col in enumerate(columns):
                    self.variables[element][ion_type_index_name][
                        col] = wind[:, index].reshape(self.nx, self.nz)

        if n_elements_read == 0 and len(self.columns) == 0:
            raise IOError(
                "Unable to open any parameter or ion tables: Have you run windsave2table?"
            )

    def project_cartesian_velocity_to_cylindrical(self):
        """Project the cartesian velocities of the wind into cylindrical
         coordinates.
         """
        v_l = np.zeros_like(self.variables["v_x"])
        v_rot = np.zeros_like(v_l)
        v_r = np.zeros_like(v_l)
        n1, n2 = v_l.shape

        for i in range(n1):
            for j in range(n2):
                cart_point = [
                    self.variables["x"][i, j], 0, self.variables["z"][i, j]
                ]
                # todo: don't think I need to do this check anymore
                if self.variables["inwind"][i, j] < 0:
                    v_l[i, j] = 0
                    v_rot[i, j] = 0
                    v_r[i, j] = 0
                else:
                    cart_velocity_vector = [
                        self.variables["v_x"][i, j],
                        self.variables["v_y"][i, j], self.variables["v_z"][i,
                                                                           j]
                    ]
                    cyl_velocity_vector = vector.project_cartesian_to_cylindrical_coordinates(
                        cart_point, cart_velocity_vector)
                    if type(cyl_velocity_vector) is int:
                        # todo: some error has happened, print a warning...
                        continue
                    v_l[i, j] = np.sqrt(cyl_velocity_vector[0]**2 +
                                        cyl_velocity_vector[2]**2)
                    v_rot[i, j] = cyl_velocity_vector[1]
                    v_r[i, j] = cyl_velocity_vector[0]

        self.variables["v_l"] = v_l * self.velocity_conversion_factor
        self.variables["v_rot"] = v_rot * self.velocity_conversion_factor
        self.variables["v_r"] = v_r * self.velocity_conversion_factor

    def create_masked_arrays(self):
        """Convert each array into a masked array, where the mask is defined by
        the inwind variable.
        """
        to_mask_wind = list(self.parameters)

        # Remove some of the columns, as these shouldn't be masked because
        # weird things will happen when creating a plot. This doesn't need to
        # be done for the wind ions as they don't have the below items in their
        # data structures

        for item_to_remove in [
                "x", "z", "r", "theta", "xcen", "zcen", "rcen", "theta_cen",
                "i", "j", "inwind"
        ]:
            try:
                to_mask_wind.remove(item_to_remove)
            except ValueError:
                continue

        # First, create masked arrays for the wind parameters which is simple
        # enough.

        for col in to_mask_wind:
            self.variables[col] = np.ma.masked_where(
                self.variables["inwind"] < 0, self.variables[col])

        # Now, create masked arrays for the wind ions. Have to do it for each
        # element and each ion type and each ion. This is probably slow :)

        for element in self.elements:
            for ion_type in self.variables[element].keys():
                for ion in self.variables[element][ion_type].keys():
                    self.variables[element][ion_type][
                        ion] = np.ma.masked_where(
                            self.variables["inwind"] < 0,
                            self.variables[element][ion_type][ion])

    def _get_element_variable(self, element_name: str, ion_name: str):

        ion_frac_or_den = ion_name[-1]
        if not ion_frac_or_den.isdigit():
            ion_name = ion_name[:-1]
            if ion_frac_or_den == "d":
                variable = self.variables[element_name]["density"][ion_name]
            elif ion_frac_or_den == "f":
                variable = self.variables[element_name]["fraction"][ion_name]
            else:
                raise ValueError(f"{ion_frac_or_den} is an unknown ion type, try f or d")
        else:
            variable = self.variables[element_name]["density"][ion_name]

        return variable

    def get(self, parameter: str,) -> Union[np.ndarray, np.ma.core.MaskedArray]:
        """Get a parameter array. This is just another way to access the
        dictionary self.variables.

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
            variable = self.variables[parameter]

        return variable

    def get_sight_line_coordinates(self, theta: float):
        """Get the vertical z coordinates for a given set of x coordinates and
        inclination angle.

        Parameters
        ----------
        theta: float
            The angle of the sight line to extract from. Given in degrees.
        """
        return np.array(self.m_coords,
                        dtype=np.float64) * np.tan(PI / 2 - np.deg2rad(theta))

    def get_variable_along_sight_line(self,
                                      theta: float,
                                      parameter: str,
                                      fraction: bool = False):
        """Extract a variable along a given sight line.

        Parameters
        ----------
        """
        if self.coord_system == "polar":
            raise NotImplementedError()

        if type(theta) is not float:
            theta = float(theta)

        z_array = np.array(self.n_coords, dtype=np.float64)
        z_coords = self.get_sight_line_coordinates(theta)
        values = np.zeros_like(z_coords, dtype=np.float64)
        w_array = self.get(parameter)

        # This is the actual work which extracts along a sight line

        for x_index, z in enumerate(z_coords):
            z_index = get_array_index(z_array, z)
            values[x_index] = w_array[x_index, z_index]

        return np.array(self.m_coords), z_array, values

    def _get_wind_coordinates(self):

        if self.coord_system == "spherical":
            n_points = self.variables["r"]
            m_points = np.zeros_like(n_points)
        elif self.coord_system == "rectilinear":
            n_points = self.variables["x"]
            m_points = self.variables["z"]
        else:
            m_points = np.log10(self.variables["r"])
            n_points = np.deg2rad(self.variables["theta"])

        return n_points, m_points

    def _get_wind_indices(self):

        if self.coord_system == "spherical":
            n_points = self.variables["i"]
            m_points = np.zeros_like(n_points)
        elif self.coord_system == "rectilinear":
            n_points = self.variables["i"]
            m_points = self.variables["i"]
        else:
            raise ValueError("Cannot plot with the cell indices for polar winds")

        return n_points, m_points

    def plot(self,
             variable_name: str,
             use_cell_coordinates: bool = True,
             fraction: bool = False,
             scale: str = "loglog"):
        """Create a plot of the wind for the given variable.
        Parameters
        ----------
        variable_name: str
            The name of the variable to plot. Ions are accessed as, i.e.,
            H_i01, He_i02, etc.
        use_cell_coordinates: bool [optional]
            Plot using the cell coordinates instead of cell index numbers
        fraction: bool [optional]
            Plot ion fractions instead of density
        scale: str [optional]
            The type of scaling for the axes
        """
        variable = self.get(variable_name)

        # Next, we have to make sure we get the correct coordinates

        if use_cell_coordinates:
            n_points, m_points = self._get_wind_coordinates()
        else:
            n_points, m_points = self._get_wind_indices()

        if self.coord_system == "spherical":
            fig, ax = plot_1d_wind(n_points, variable, scale=scale)
        else:
            fig, ax = plot_2d_wind(n_points, m_points, variable,
                                   self.coord_system, scale=scale)

        # Finally, label the axes with what we actually plotted

        if len(ax) == 1:
            ax = ax[0, 0]
            title = f"{variable_name}".replace("_", " ")
            if self.coord_system == "spherical":
                ax.set_ylabel(title)
            else:
                ax.set_title(title)

        return fig, ax

    def get_elem_number_from_ij(self, i: int, j: int):
        """Get the wind element number for a given i and j index.
        """
        return self.nz * i + j

    def get_ij_from_elem_number(self, elem: int):
        """Get the i and j index for a given wind element number.
        """
        return np.unravel_index(elem, (self.nx, self.nz))

    def show(self, block=True):
        """Show a plot which has been created.
        """
        plt.show(block=block)

    def __getitem__(self, key):
        """Return an array in the variables dictionary when indexing.
        """
        return self.variables[key]

    def __setitem__(self, key, value):
        """Set an array in the variables dictionary.
        """
        self.variables[key] = value

    def __str__(self):
        """Print basic details about the wind.
        """
        txt = "root: {}\nfilepath: {}\ncoordinate system:{}\nparameters: {}\nelements: {}\n".format(
            self.root, self.fp, self.coord_system, self.parameters,
            self.elements)

        return txt


# Functions ----------------


def cleanup_data(fp: str = ".", verbose: bool = False):
    """Search recursively from the specified directory for symbolic links named
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
    cmd = "cd {}; find . -type l -name 'data'".format(fp)
    stdout, stderr = Popen(cmd, stdout=PIPE, stderr=PIPE,
                           shell=True).communicate()
    stdout = stdout.decode("utf-8")
    stderr = stderr.decode("utf-8")

    if stderr:
        print("sent from stderr")
        print(stderr)

    if stdout and verbose:
        print(
            "deleting data symbolic links in the following directories:\n\n{}".
            format(stdout[:-1]))
    else:
        print("no data symlinks to delete")
        return n_del

    directories = stdout.split()

    for directory in directories:
        current = fp + directory[1:]
        cmd = "rm {}".format(current)
        stdout, stderr = Popen(cmd, stdout=PIPE, stderr=PIPE,
                               shell=True).communicate()
        if stderr:
            print(stderr.decode("utf-8"))
        else:
            n_del += 1

    return n_del


def get_file(pattern: str, fp: str = "."):
    """Find files of the given pattern recursively.

    Parameters
    ----------
    pattern: str
        Patterns to search recursively for, i.e. *.pf, *.spec, tde_std.pf
    fp: str [optional]
        The directory to search from, if not specified in the pattern.
    """

    files = [str(file_) for file_ in Path(f"{fp}").rglob(pattern)]
    if ".pf" in pattern:
        files = [
            file_ for file_ in files
            if "out.pf" not in file_ and "py_wind" not in file_
        ]
    files.sort(key=lambda var: [
        int(x) if x.isdigit() else x
        for x in re.findall(r'[^0-9]|[0-9]+', var)
    ])

    return files


def get_array_index(x: np.ndarray, target: float) -> int:
    """Return the index for a given value in an array. This function will not
    be happy if you pass an array with duplicate values. It will always return
    the first instance of the duplicate array.

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
    if target < np.min(x):
        return 0
    if target > np.max(x):
        return -1

    index = np.abs(x - target).argmin()

    return index


def get_root(fp: str) -> Tuple[str, str]:
    """Get the root name of a Python simulation, extracting it from a file path.

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
        raise TypeError(
            "expected a string as input for the file path, not whatever you put"
        )

    dot = fp.rfind(".")
    slash = fp.rfind("/")

    root = fp[slash + 1:dot]
    fp = fp[:slash + 1]
    if fp == "":
        fp = "./"

    return root, fp


def smooth_array(array: Union[np.ndarray, List[Union[float, int]]],
                 width: Union[int, float]) -> np.ndarray:
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

    if type(width) is not int:
        try:
            width = int(width)
        except ValueError:
            print("Unable to cast {} into an int".format(width))
            return array

    if type(array) is not np.ndarray:
        array = np.array(array)

    array = np.reshape(
        array,
        (len(array), ))  # todo: why do I have to do this? safety probably

    return convolve(array, boxcar(width) / float(width), mode="same")


def create_wind_save_tables(root: str,
                            fp: str = ".",
                            ion_density: bool = False,
                            verbose: bool = False) -> None:
    """Run windsave2table in a directory to create the standard data tables. The
    function can also create a root.all.complete.txt file which merges all the
    data tables together into one (a little big) file.

    Parameters
    ----------
    root: str
        The root name of the Python simulation
    fp: str
        The directory where windsave2table will run
    ion_density: bool [optional]
        Use windsave2table in the ion density version instead of ion fractions
    verbose: bool [optional]
        Enable verbose output
    """
    in_path = which("windsave2table")
    if not in_path:
        raise OSError("windsave2table not in $PATH and executable")

    files_before = listdir(fp)

    command = f"cd {fp};"
    if not Path(f"{fp}/data").exists():
        command += "Setup_Py_Dir;"
    command += "windsave2table"
    if ion_density:
        command += " -d"
    command += " {}".format(root)

    cmd = Popen(command, stdout=PIPE, stderr=PIPE, shell=True)
    stdout, stderr = cmd.communicate()

    files_after = listdir(fp)

    if verbose:
        print(stdout.decode("utf-8"))
    if stderr:
        print("There may have been a problem running windsave2table")

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

    return


def run_py_wind_commands(root: str,
                         commands: List[str],
                         fp: str = ".") -> List[str]:
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
    cmd_file = "{}/.tmpcmds.txt".format(fp)

    with open(cmd_file, "w") as f:
        for i in range(len(commands)):
            f.write("{}\n".format(commands[i]))

    sh = Popen("cd {}; py_wind {} < .tmpcmds.txt".format(fp, root),
               stdout=PIPE,
               stderr=PIPE,
               shell=True)
    stdout, stderr = sh.communicate()
    if stderr:
        print(stderr.decode("utf-8"))

    remove(cmd_file)

    return stdout.decode("utf-8").split("\n")
