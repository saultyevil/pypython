#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create a spectrum from a delay_dump file.

It is possible to create spectra which certain physical processes
removed. If the delay dump file is large, then use the SQL version of
the functions, otherwise the regular version is much faster.
"""

import csv
from copy import deepcopy

import numpy as np
# import sqlalchemy
from numba import jit
# from sqlalchemy import Column, Float, Integer
# from sqlalchemy.exc import SQLAlchemyError
# from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy.orm import sessionmaker

from pypython import Spectrum, Wind
from pypython.constants import PARSEC, C
from pypython.error import EXIT_FAIL
from pypython.physics import angstrom_to_hz, hz_to_angstrom

BOUND_FREE_NRES = 20000
UNFILTERED_SPECTRUM = -999

# Base = declarative_base()


def write_delay_dump_spectrum_to_file(root,
                                      wd,
                                      spectrum,
                                      extract_nres,
                                      n_spec,
                                      n_bins,
                                      d_norm_pc,
                                      return_inclinations=False):
    """Write the generated delay dump spectrum to file.

    Parameters
    ----------
    root: str
        The root name of the model.
    wd: str
        The directory containing the model.
    spectrum: np.ndarray
        The delay dump spectrum.
    extract_nres: tuple
        The internal line number for a line to extract.
    n_spec: int
        The number of spectral inclination angles
    n_bins: int
        The number of frequency bins.
    d_norm_pc: float
        The distance normalization of the spectrum.
    return_inclinations: [optional] bool
        Return the inclination angles (if any) of the spectrum

    Returns
    -------
    spectrum: np.ndarray
        The delay dump spectrum.
    inclinations: [optional] np.ndarray
        An array of the inclination angles of the spectrum.
    """

    if extract_nres[0] != UNFILTERED_SPECTRUM:
        fname = "{}/{}_line".format(wd, root)
        for line in extract_nres:
            fname += "_{}".format(line)
        fname += ".delay_dump.spec"
    else:
        fname = "{}/{}.delay_dump.spec".format(wd, root)

    f = open(fname, "w")

    f.write("# Flux Flambda [erg / s / cm^2 / A at {} pc\n".format(d_norm_pc))

    try:
        full_spec = Spectrum(root, wd)
        inclinations = list(full_spec.inclinations)
    except IOError:
        inclinations = np.arange(0, n_spec)

    header = deepcopy(inclinations)

    # Write out the header of the output file

    f.write("{:12s} {:12s}".format("Freq.", "Lambda"))
    for h in header:
        f.write(" {:12s}".format(h))
    f.write("\n")

    # Now write out the spectrum

    for i in range(n_bins):
        freq = spectrum[i, 0]
        wl_angstrom = hz_to_angstrom(freq)
        f.write("{:12e} {:12e}".format(freq, wl_angstrom))
        for j in range(spectrum.shape[1] - 1):
            f.write(" {:12e}".format(spectrum[i, j + 1]))
        f.write("\n")

    f.close()

    # If some lines are being extracted, then we can calculate their luminosities
    # and write these out to file too
    # TODO: update to allow multiple lines to be written out at once

    if extract_nres[0] != UNFILTERED_SPECTRUM and len(extract_nres) == 1:
        output_fname = "{}/{}_line".format(wd, root)
        for line in extract_nres:
            output_fname += "_{}".format(line)
        output_fname += ".line_luminosity.diag"
        f = open(output_fname, "w")
        f.write("Line luminosities -- units [erg / s]\n")
        for i in range(spectrum.shape[1] - 1):
            flux = np.sum(spectrum[:, i + 1])
            lum = 4 * np.pi * (d_norm_pc * PARSEC)**2 * flux
            f.write("Spectrum {} : L = {} erg / s\n".format(header[i], lum))
        f.close()

    if return_inclinations:
        return spectrum, inclinations
    else:
        return spectrum


def read_delay_dump(root, cd=".", mode_dev=False):
    """Process the photons which have been dumped to the delay_dump file.

    Parameters
    ----------
    root: str
        The root name of the simulation.
    cd: str
        The directory containing the simulation.
    mode_dev: bool [optional]
        Use when using the standard format currently in the main repository.

    Returns
    -------
    dumped_photons: pd.DataFrame
        An array containing the dumped photons with the quantities specified
        by the extract dict.
    """

    filename = "{}/{}.delay_dump".format(cd, root)

    # There are cases where LineRes. is not defined within the delay dump file,
    # i.e. in the regular dev version

    if mode_dev:
        names = {
            "Freq.": np.float64,
            "Lambda": np.float64,
            "Weight": np.float64,
            "LastX": np.float64,
            "LastY": np.float64,
            "LastZ": np.float64,
            "Scat.": np.int32,
            "RScat.": np.int32,
            "Delay": np.float64,
            "Spec.": np.int32,
            "Orig.": np.int32,
            "Res.": np.int32
        }
    else:
        names = {
            "Np": np.int32,
            "Freq.": np.float64,
            "Lambda": np.float64,
            "Weight": np.float64,
            "LastX": np.float64,
            "LastY": np.float64,
            "LastZ": np.float64,
            "Scat.": np.int32,
            "RScat.": np.int32,
            "Delay": np.float64,
            "Spec.": np.int32,
            "Orig.": np.int32,
            "Res.": np.int32,
            "LineRes.": np.int32
        }

    # output = pd.read_csv(filename, names=list(names.keys()), dtype=names, delim_whitespace=True, comment="#")

    dd = open(filename, "r")
    for _ in range(5):
        next(dd)

    output = csv.DictReader(dd, fieldnames=names)

    return output


def convert_weight_to_flux(spectrum, spec_cycle_norm, d_norm_pc):
    """Re-normalize the photon weight bins into a Flux per unit wavelength.

    spec_cycle_norm fixes the case where less than the specified number of
    spectral cycles were run, which would mean not all of the flux has been
    generated just yet: spec_norm >= 1.

    Parameters
    ----------
    spectrum: np.ndarray
        The spectrum array containing the frequency and weight bins.
    spec_cycle_norm: float
        The spectrum normalization amount - usually the number of spectrum
    d_norm_pc: [optional] float
        The distance normalization for the flux calculation in parsecs. By
        default this is 100 parsecs.

    Returns
    -------
    spectrum: np.ndarray
        The renormalized spectrum.
    """

    n_bins = spectrum.shape[0]
    n_spec = spectrum.shape[1] - 1
    d_norm_cm = 4 * np.pi * (d_norm_pc * PARSEC)**2

    for i in range(n_bins - 1):
        for j in range(n_spec):
            freq = spectrum[i, 0]
            d_freq = spectrum[i + 1, 0] - freq
            spectrum[i, j + 1] *= (freq**2 * 1e-8) / (d_freq * d_norm_cm * C)

        spectrum[i, 1:] *= spec_cycle_norm

    return spectrum


@jit(nopython=True)
def bin_photon_weights(spectrum, freq_min, freq_max, photon_freqs, photon_weights, photon_spc_i, photon_nres,
                       photon_line_nres, extract_nres, logbins):
    """Bin the photons into frequency bins using jit to attempt to speed
    everything up.

    BOUND_FREE_NRES = NLINES = 20000 has been hardcoded. Any values of nres
    larger than BOUND_FREE_NRES is a bound-free continuum event. If this value
    is changed in Python, then this value needs updating.

    Parameters
    ----------
    spectrum: np.ndarray
        The spectrum array containing the frequency bins.
    freq_min: float
        The minimum frequency to bin.
    freq_max: float
        The maximum frequency to bin.
    photon_freqs: np.ndarray
        The photon frequencies.
    photon_weights: np.ndarray
        The photon weights.
    photon_spc_i: np.ndarray
        The index for the spectrum the photons belong to.
    photon_nres: np.ndarry:
        The Res. values for the photons.
    photon_line_nres: np.ndarray
        The LineRes values for the photons.
    extract_nres: int or np.ndarray or list
        The line number for the line to extract
    logbins: bool
        Use frequency bins spaced in log space.

    Returns
    -------
    spectrum: np.ndarray
        The spectrum where photon weights have been binned.
    """

    n_extract = len(extract_nres)
    n_photons = photon_freqs.shape[0]
    n_bins = spectrum.shape[0]

    if logbins:
        d_freq = (np.log10(freq_max) - np.log10(freq_min)) / n_bins
    else:
        d_freq = (freq_max - freq_min) / n_bins

    for p in range(n_photons):

        if photon_freqs[p] < freq_min or photon_freqs[p] > freq_max:
            continue

        if logbins:
            k = int((np.log10(photon_freqs[p]) - np.log10(freq_min)) / d_freq)
        else:
            k = int((photon_freqs[p] - freq_min) / d_freq)

        if k < 0:
            k = 0
        elif k > n_bins - 1:
            k = n_bins - 1

        # If a single transition is to be extracted, then we do that here. Note
        # that if nres < 0 or nres > NLINES, then it was a continuum scattering
        # event

        if extract_nres[0] != UNFILTERED_SPECTRUM:
            # Loop over each nres which we want to extract
            for i in range(n_extract):
                # If it's last interaction is the nres we want, then extract
                if photon_nres[p] == extract_nres[i]:
                    spectrum[k, photon_spc_i[p]] += photon_weights[p]
                    break
                # Or if it's "belongs" to the nres we want and it's last interaction
                # was a continuum scatter, then extract
                elif photon_line_nres[p] == extract_nres[i]:
                    spectrum[k, photon_spc_i[p]] += photon_weights[p]
                    break
        else:
            spectrum[k, photon_spc_i[p]] += photon_weights[p]

    return spectrum


def create_spectrum_process_breakdown(root,
                                      wl_min,
                                      wl_max,
                                      n_cores_norm=1,
                                      spec_cycle_norm=1,
                                      wd=".",
                                      nres=None,
                                      mode_line_res=True):
    """Get the spectra for the different physical processes which contribute to
    a spectrum. If nres is provided, then only a specific interaction will be
    extracted, otherwise all resonance interactions will.

    Parameters
    ----------
    root: str
        The root name of the simulation.
    wl_min: float
        The lower wavelength bound in Angstroms.
    wl_max: float
        The upper wavelength bound in Angstroms.
    n_cores_norm: int [optional]
        The number of cores normalization constant, i.e. the number of cores used
        to generate the delay_dump file.
    spec_cycle_norm: float [optional]
        The spectral cycle normalization, this is equal to 1 if all spectral
        cycles were run.
    wd: str [optional]
        The directory containing the simulation.
    nres: int [optional]
        A specific interaction to extract, is the nres number from Python.
    mode_line_res: bool [optional]
        Set as True if the delay_dump has the LineRes. value.

    Returns
    -------
    spectra: dict
        A dictionary where the keys are the name of the spectra and the values
        are pd.DataFrames of that corresponding spectrum.
    """

    df = read_delay_dump(root, cd=wd)
    s = Spectrum(root, wd)

    # create dataframes for each physical process, what you can actually get
    # depends on mode_line_res, i.e. if LineRes. is included or not. Store these
    # data frame in a list

    contributions = []
    contribution_names = ["Extracted"]

    # Extract either a specific interaction, or all the interactions. If LineRes.
    # is enabled, then extract the LineRes version of it too

    if nres:
        if type(nres) != int:
            nres = int(nres)
        contributions.append(df[df["Res."] == nres])
        contribution_names.append("Res." + str(nres))
        if mode_line_res:
            contributions.append(df[df["LineRes."] == nres])
            contribution_names.append("LineRes." + str(nres))
    else:
        tmp = df[df["Res."] <= 20000]
        contributions.append(tmp[tmp["Res."] >= 0])
        contribution_names.append("Res.")
        if mode_line_res:
            tmp = df[df["LineRes."] <= 20000]
            contributions.append(tmp[tmp["LineRes."] >= 0])
            contribution_names.append("LineRes.")

    # Extract the scattered spectrum, which is every continuum scatter

    contributions.append(df[df["Res."] == -1])
    contribution_names.append("Scattered")

    # Extract pure BF, FF and ES events, unless we're in Res. mode, which extracts
    # last scatters

    if mode_line_res:
        contributions.append(df[df["LineRes."] == -1])
        contributions.append(df[df["LineRes."] == -2])
        contributions.append(df[df["LineRes."] > 20000])
        contribution_names.append("ES only")
        contribution_names.append("FF only")
        contribution_names.append("BF only")
    else:
        contributions.append(df[df["Res."] == -2])
        contributions.append(df[df["Res."] > 20000])
        contribution_names.append("FF only")
        contribution_names.append("BF only")

    # Create each individual spectrum

    created_spectra = [s]
    for contribution in contributions:
        created_spectra.append(
            create_spectrum(root,
                            wd,
                            dumped_photons=contribution,
                            freq_min=angstrom_to_hz(wl_max),
                            freq_max=angstrom_to_hz(wl_min),
                            n_cores_norm=n_cores_norm,
                            spec_cycle_norm=spec_cycle_norm))
    n_spec = len(created_spectra)

    # dict comprehension to use contribution_names as the keys and the spectra
    # as the values

    return {contribution_names[i]: created_spectra[i] for i in range(n_spec)}


@jit(nopython=True)
def wind_bin_photon_weights(n_photons, nres, photon_x, photon_y, photon_z, photon_nres, photon_weight, x_points,
                            z_points, nx, nz):
    """Bin photon weights by extract location."""

    hist2d_weight = np.zeros((nx, nz))
    hist2d_count = np.zeros((nx, nz))

    for i in range(n_photons):
        if photon_nres[i] != nres:
            continue
        rho = np.sqrt(photon_x[i]**2 + photon_y[i]**2)
        # get array index for rho point
        if rho < np.min(x_points):
            ix = 0
        elif rho > np.max(x_points):
            ix = -1
        else:
            ix = np.abs(x_points - rho).argmin()
        # get array index for z point
        z = np.abs(photon_z[i])
        if z < np.min(z_points):
            iz = 0
        elif z > np.max(z_points):
            iz = -1
        else:
            iz = np.abs(z_points - z).argmin()
        hist2d_weight[ix, iz] += photon_weight[i]
        hist2d_count[ix, iz] += 1

    return hist2d_weight, hist2d_count


def wind_bin_interaction_weight(root, nres, cd=".", n_cores=1):
    """Bin photon weights by extract location.

    Parameters
    ----------
    root: str
        The root name of the model.
    nres: int
        The resonance number of the photon to bin.
    cd: str [optional]
        The directory containing the simulation.
    n_cores: int [optional]
        The number of cores to normalize the binning by.

    Returns
    -------
    hist2d_weight: np.ndarray
        The photon weights. Each element of the array corresponds to a cell on
        the grid.
    """

    w = Wind(root, cd, masked=False)
    x_points = np.array(w.x_coords)
    z_points = np.array(w.y_coords)

    photons = read_delay_dump(root, cd, False)
    if photons.empty:
        print("photon dataframe is empty")
        exit(1)

    hist2d_weight, hist2d_count = wind_bin_photon_weights(len(photons), nres, photons["LastX"].values,
                                                          photons["LastY"].values, photons["LastZ"].values,
                                                          photons["Res."].values, photons["Weight"].values, x_points,
                                                          z_points, w.nx, w.nz)

    hist2d_weight /= n_cores

    name = "{}/{}_wind_Res{}_".format(cd, root, nres)
    np.savetxt(name + "weight.txt", hist2d_weight)
    np.savetxt(name + "count.txt", hist2d_count)

    return hist2d_weight, hist2d_count


def create_spectrum(root,
                    wd=".",
                    extract_nres=(UNFILTERED_SPECTRUM, ),
                    dumped_photons=None,
                    freq_bins=None,
                    freq_min=None,
                    freq_max=None,
                    n_bins=10000,
                    d_norm_pc=100,
                    spec_cycle_norm=1,
                    n_cores_norm=1,
                    logbins=True,
                    mode_dev=False,
                    output_numpy=False):
    """Create a spectrum for each inclination angle using the photons which
    have been dumped to the root.delay_dump file.

    Spectrum frequency bins are rounded to 7 significant figures as this makes
    them the same values as what is output from the Python spectrum.

    Parameters
    ----------
    root: str
        The root name of the simulation.
    wd: [optional] str
        The directory containing the simulation.
    extract_nres: [optional] int
        The internal line number for a line to extract.
    dumped_photons: [optional] pd.DataFrame
        The delay dump photons in a Pandas DataFrame. If this is not provided,
        then it will be read in.
    freq_bins: [optional] np.ndarray
        Frequency bins to use to bin photons.
    freq_min: [optional] float
        The smallest frequency bin.
    freq_max: [optional] float
        The largest frequency bin
    n_bins: [optional] int
        The number of frequency bins.
    d_norm_pc: [optional] float
        The distance normalization of the spectrum.
    spec_cycle_norm: float
        A normalization constant for the spectrum, is > 1.
    n_cores_norm: [optional] int
        The number of cores which were used to generate the delay_dump file
    logbins: [optional] bool
        If True, the frequency bins are spaced equally in log space. Otherwise
        the bins are in linear space.
    mode_dev: bool [optional]
        If True, then LineRes. and Np will NOT be included when being read in

    Returns
    -------
    filtered_spectrum: np.ndarray or pd.DataFrame
        A 2D array containing the frequency in the first column and the
        fluxes for each inclination angle in the other columns.
    """

    if type(extract_nres) != tuple:
        print("extract_nres is not a tuple but is of type {}".format(type(extract_nres)))
        exit(EXIT_FAIL)

    # If the frequency bins have been provided, we need to do some checks to make
    # sure they're sane

    if freq_bins is not None:
        if type(freq_bins) != np.ndarray:
            freq_bins = np.array(freq_bins, dtype=np.float64)
        is_increasing = np.all(np.diff(freq_bins) > 0)
        if not is_increasing:
            raise ValueError("the values for the frequency bins provided are not increasing")
        n_bins = len(freq_bins)

    if dumped_photons is None:
        dumped_photons = read_delay_dump(root, wd, mode_dev)

    if mode_dev:
        line_res = deepcopy(dumped_photons["Res."].values.astype(int))
    else:
        line_res = dumped_photons["LineRes."].values.astype(int)

    n_spec = int(np.max(dumped_photons["Spec."].values)) + 1
    spectrum = np.zeros((n_bins, 1 + n_spec))

    if freq_bins is not None:
        spectrum[:, 0] = freq_bins
    else:
        if not freq_min:
            freq_min = np.min(dumped_photons["Freq."])
        if not freq_max:
            freq_max = np.max(dumped_photons["Freq."])
        if logbins:
            spectrum[:, 0] = np.logspace(np.log10(freq_min), np.log10(freq_max), n_bins, endpoint=True)
        else:
            spectrum[:, 0] = np.linspace(freq_min, freq_max, n_bins, endpoint=True)

    # This function now constructs a spectrum given the photon frequencies and
    # weights as well as any other normalization constants

    freq_min = np.min(spectrum[:, 0])
    freq_max = np.max(spectrum[:, 0])

    spectrum = bin_photon_weights(spectrum, freq_min, freq_max, dumped_photons["Freq."].values,
                                  dumped_photons["Weight"].values, dumped_photons["Spec."].values.astype(int) + 1,
                                  dumped_photons["Res."].values.astype(int), line_res, extract_nres, logbins)

    spectrum[:, 1:] /= n_cores_norm

    spectrum = convert_weight_to_flux(spectrum, spec_cycle_norm, d_norm_pc)

    # Remove the first and last bin, consistent with Python

    n_bins -= 2
    spectrum = spectrum[1:-1, :]

    spectrum, inclinations = write_delay_dump_spectrum_to_file(root,
                                                               wd,
                                                               spectrum,
                                                               extract_nres,
                                                               n_spec,
                                                               n_bins,
                                                               d_norm_pc,
                                                               return_inclinations=True)

    if output_numpy:
        return spectrum
    else:
        lamda = np.reshape(C / spectrum[:, 0] * 1e8, (n_bins, 1))
        spectrum = np.append(lamda, spectrum, axis=1)
        df = pd.DataFrame(spectrum, columns=["Lambda", "Freq."] + inclinations)
        return df


# class Photon(Base):
#     """Photon object for SQL database."""
#     __tablename__ = "Photons"
#     id = Column(Integer, primary_key=True, autoincrement=True)
#     np = Column(Integer)
#     freq = Column(Float)
#     wavelength = Column(Float)
#     weight = Column(Float)
#     x = Column(Float)
#     y = Column(Float)
#     z = Column(Float)
#     scat = Column(Integer)
#     rscat = Column(Integer)
#     delay = Column(Integer)
#     spec = Column(Integer)
#     orig = Column(Integer)
#     res = Column(Integer)
#     lineres = Column(Integer)
#
#     def __repr__(self):
#         return str(self.id)
#
#
# def get_photon_db(root, cd=".", dd_dev=False, commitfreq=1000000):
#     """Create or open a database to store the delay_dump file in an easier to
#     query data structure.
#
#     Parameters
#     ----------
#     root: str
#         The root name of the simulation.
#     cd: str [optional]
#         The directory containing the simulation.
#     dd_dev: bool [optional]
#         Expect the delay_dump file to be in the format used in the main Python
#         repository.
#     commitfreq: int
#         The frequency to which commit the database and avoid out-of-memory
#         errors. If this number is too low, database creation will take a long
#         time.
#
#     Returns
#     -------
#     engine:
#         The SQLalchemy engine.
#     session:
#         The SQLalchemy session.
#     """
#
#     engine = sqlalchemy.create_engine("sqlite:///{}.db".format(root))
#     engine_session = sessionmaker(bind=engine)
#     session = engine_session()
#
#     if dd_dev:
#         column_names = [
#             "Freq", "Lambda", "Weight", "LastX", "LastY", "LastZ", "Scat", "RScat", "Delay", "Spec", "Orig", "Res"
#         ]
#     else:
#         column_names = [
#             "Np", "Freq", "Lambda", "Weight", "LastX", "LastY", "LastZ", "Scat", "RScat", "Delay", "Spec", "Orig",
#             "Res", "LineRes"
#         ]
#     n_columns = len(column_names)
#
#     try:
#         session.query(Photon.weight).first()
#     except SQLAlchemyError:
#         print("{}.db does not exist, so creating now".format(root))
#         with open(cd + "/" + root + ".delay_dump", "r") as infile:
#             nadd = 0
#             Base.metadata.create_all(engine)
#             for n, line in enumerate(infile):
#                 if line.startswith("#"):
#                     continue
#                 try:
#                     values = [float(i) for i in line.split()]
#                 except ValueError:
#                     print("Line {} has values which cannot be converted into a number".format(n))
#                     continue
#                 if len(values) != n_columns:
#                     print("Line {} has unknown format with {} columns:\n{}".format(n, len(values), line))
#                     continue
#                 if dd_dev:
#                     session.add(
#                         Photon(np=int(n),
#                                freq=values[0],
#                                wavelength=values[1],
#                                weight=values[2],
#                                x=values[3],
#                                y=values[4],
#                                z=values[5],
#                                scat=int(values[6]),
#                                rscat=int(values[7]),
#                                delay=int(values[8]),
#                                spec=int(values[9]),
#                                orig=int(values[10]),
#                                res=int(values[11]),
#                                lineres=int(values[11])))
#                 else:
#                     session.add(
#                         Photon(np=int(values[0]),
#                                freq=values[1],
#                                wavelength=values[2],
#                                weight=values[3],
#                                x=values[4],
#                                y=values[5],
#                                z=values[6],
#                                scat=int(values[7]),
#                                rscat=int(values[8]),
#                                delay=int(values[9]),
#                                spec=int(values[10]),
#                                orig=int(values[11]),
#                                res=int(values[12]),
#                                lineres=int(values[13])))
#
#                 nadd += 1
#                 if nadd > commitfreq:
#                     session.commit()
#                     nadd = 0
#
#         session.commit()
#
#     return engine, session
