#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyse or modify simulation conditions.

This module contains functions for analysing the quality of a
simulation, or for modifying the atomic data or parameter files of a
simulation.
"""

import re
from copy import copy
from glob import glob

from pypython.simulation import atomicdata, grid


def check_model_convergence(root, fp=".", return_per_cycle=False, return_converging=False):
    """Check the convergence of a Python simulation by parsing the.

    !!Check_convergence line in the Python diag file.

    Parameters
    ----------
    root: str
        The root name of the Python simulation
    fp: str [optional]
        The working directory of the Python simulation
    return_per_cycle: bool [optional]
        Return the convergence fraction for each cycle
    return_converging: bool [optional]
        Return the number of cells which are still converging instead of the
        number of cells which have converged

    Returns
    -------
    convergence: float or int or list
        The convergence fraction in the final cycle of the simulation. If this
        is -1, then a convergence fraction was not found.
    """
    brief_summary_len = 9
    convergence = [-1]
    converging = [-1]

    with open(f"{fp}/diag_{root}/{root}_0.diag", "r") as f:
        diag = f.readlines()

    prev = ""
    convergence_per_cycle = []
    converging_per_cycle = []

    for i in range(len(diag)):
        line = diag[i]
        if line.find("converged") != -1 and line.find("converging") != -1:
            # Skip if the convergence statistic is from the brief run summary
            if prev.find("Convergence statistics for the wind after the ionization calculation:") != -1:
                i += brief_summary_len
                continue

            line = line.split()

            try:
                convergence = float(line[2].replace("(", "").replace(")", ""))
                convergence_per_cycle.append(convergence)
            except ValueError:
                convergence_per_cycle.append(-1)
                continue

            try:
                converging = float(line[6].replace("(", "").replace(")", ""))
                converging_per_cycle.append(converging)
            except ValueError:
                converging_per_cycle.append(-1)
                continue

        prev = copy(line)

    if return_converging:
        if return_per_cycle:
            return converging_per_cycle
        return converging

    if return_per_cycle:
        return convergence_per_cycle
    return convergence


def model_convergence_components(root, wd="."):
    """Returns a break down in terms of the number of cells which have passed
    the convergence checks on radiation temperature, electron temperature and
    heating and cooling balance.

    Parameters
    ----------
    root: str
        The root name of the Python simulation
    wd: str [optional]
        The working directory of the Python simulation

    Returns
    -------
    n_tr: List[float]
        The fraction of cells which have passed the radiation temperature
        convergence test.
    n_te: List[float]
        The fraction of cells which have passed the electron temperature
        convergence test.
    n_te_max: List[float]
        The fraction of cells which have reached the maximum electron
        temperature.
    n_hc: List[float]
        The fraction of cells which have passed the heating/cooling
        convergence test.
    """
    brief_summary_len = 7
    n_tr = []
    n_te = []
    n_hc = []
    n_te_max = []
    file_found = False

    diag_path = "{}/diag_{}/{}_0.diag".format(wd, root, root)
    try:
        with open(diag_path, "r") as f:
            diag = f.readlines()
        file_found = True
    except IOError:
        pass

    if not file_found:
        print("unable to find {}_0.diag file".format(root))
        return n_tr, n_te, n_hc, n_te_max

    for i in range(len(diag)):
        line = diag[i]
        if line.find("t_r") != -1 and line.find("t_e(real)") != -1 and line.find("hc(real)") != -1:
            if diag[i - 2].find("Convergence statistics for the wind after the ionization calculation:") != -1:
                i += brief_summary_len
                continue

            line = line.split()

            try:
                n_cells = int(diag[i - 1].split()[9])
                n_tr.append(int(line[2]) / n_cells)
                n_te.append(int(line[4]) / n_cells)
                n_te_max.append(int(line[6]) / n_cells)
                n_hc.append(int(line[8]) / n_cells)
            except ValueError:
                n_tr.append(-1)
                n_te.append(-1)
                n_te_max.append(-1)
                n_hc.append(-1)

    return n_tr, n_te, n_te_max, n_hc


def model_error_summary(root, wd=".", n_cores=-1, print_errors=False):
    """Return a dictionary containing each error found in the error summary for
    each processor for a Python simulation.
    todo: create a mode where a dict is returned for each MPI process

    Parameters
    ----------
    root: str
        The root name of the Python simulation
    wd: str [optional]
        The working directory of the Python simulation
    n_cores: int [optional]
        If this is provided, then only the first n_cores processes will be
        checked for errors
    print_errors: bool [optional]
        Print the error summary to screen

    Returns
    -------
    total_errors: dict
        A dictionary containing the total errors over all processors. The keys
        are the error messages and the values are the number of times that
        error occurred.
    """
    total_errors = {}
    diag_files = glob(f"{wd}/diag_{root}/{root}_*.diag")
    diag_files.sort(key=lambda var: [int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])

    if n_cores > 0:
        n_diag_files = n_cores
    else:
        n_diag_files = len(diag_files)

    if n_diag_files == 0:
        print(f"No .diag files were found in {wd}/diag_{root} so cannot find any errors")
        return total_errors

    broken_diag_files = []

    exit_msg = "ended for unknown reasons"

    for diag in diag_files:
        try:
            with open(diag, "r") as f:
                lines = f.readlines()
        except IOError:
            broken_diag_files.append(diag)
            continue

        # Find the final error summary: look over the lines list in reverse

        error_start = -1

        for k, start_line in enumerate(lines):

            # Find start of error summary

            if start_line.find("Error summary:") != -1:
                error_start = k
                if start_line.find("End of program") != -1:
                    exit_msg = "exited successfully"
                else:
                    exit_msg = "was aborted"
            else:
                continue

            # Find the final error summary

            error_end = -1

            for kk, end_line in enumerate(lines[error_start + 3:]):
                end_line = end_line.split()
                if len(end_line):
                    if end_line[0].isdigit() is False:
                        error_end = error_start + kk + 3
                        break

            if error_end == -1:
                broken_diag_files.append(diag)
                break

            # Extract the errors from the diag file

            errors = lines[error_start:error_end]

            for error_line in errors:
                error_words = error_line.split()
                if len(error_words) == 0:
                    continue
                try:
                    error_count = error_words[0]
                except IndexError:
                    print("index error when trying to process line '{}' for {}".format(" ".join(error_words), diag))
                    broken_diag_files.append(diag)
                    break

                if error_count.isdigit():
                    try:
                        error_count = int(error_words[0])
                    except ValueError:
                        continue
                    error_message = " ".join(error_words[2:])
                    try:
                        total_errors[error_message] += error_count
                    except KeyError:
                        total_errors[error_message] = error_count

        if error_start == -1:
            broken_diag_files.append(diag)

    if len(broken_diag_files) == len(diag_files):
        print(f"Unable to find any error summaries for {root + wd}")
        return total_errors

    if print_errors:
        n_reported = len(diag_files) - len(broken_diag_files)
        print(
            f"Total errors reported from {n_reported} of {len(diag_files)} processes for {wd + root}, which {exit_msg}:"
        )
        for key in total_errors.keys():
            n_error = int(total_errors[key])
            if n_error < 1:
                n_error = 1
            print("  {:6d} -- {}".format(n_error, key))

    return total_errors
