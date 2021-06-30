#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Equations for accretion discs.

Includes temperature profiles, calculations for the Eddington luminosity
and accretion rate and also a function to create a crude accretion disc
spectrum.
"""

import numpy as np

from pypython.constants import (MPROT, MSOL, MSOL_PER_YEAR, PI, STEFAN_BOLTZMANN, THOMPSON, C, G)
from pypython.physics.blackbody import planck_lambda, planck_nu
from pypython.physics.blackhole import (gravitational_radius, innermost_stable_circular_orbit)


def alpha_disc_effective_temperature(ri, r_co, m_co, mdot):
    """Standard alpha-disc effective temperature profile.

    Parameters
    ----------
    ri: np.ndarray or float
        An array of radii or a single radius to calculate the temperature at.
    r_co: float
        The radius of the central object.
    m_co: float
        The mass of the central object in units of solar masses.
    mdot: float
        The accretion rate onto the central object in units of solar masses per
        year.

    Returns
    -------
    teff: np.ndarray or float
        The effective temperature at the provided radius or radii.
    """

    m_co *= MSOL
    mdot *= MSOL_PER_YEAR

    with np.errstate(all="ignore"):
        teff4 = (3 * G * m_co * mdot) / (8 * np.pi * ri**3 * STEFAN_BOLTZMANN)
        teff4 *= 1 - (r_co / ri)**0.5

    return teff4**0.25


def create_disc_spectrum(m_co, mdot, r_in, r_out, freq_min, freq_max, freq_units=True, n_freq=5000, n_rings=1000):
    """Create a crude accretion disc spectrum. This works by approximating an
    accretion disc as being a collection of annuli radiating at different
    temperatures and treats them as a blackbody. The emerging spectrum is then
    an ensemble of these blackbodies.

    Parameters
    ----------
    m_co: float
        The mass of the central object in solar masses.
    mdot: float
        The accretion rate onto the central object in solar masses per year.
    r_in: float
        The inner radius of the accretion disc in cm.
    r_out: float
        The outer radius of the accretion disc in cm.
    freq_min: float
        The low frequency boundary of the spectrum to create.
    freq_max: float
        The high frequency boundary of the spectrum to create.
    freq_units: float
        Calculate the spectrum in frequency units, or wavelength units if False.
    n_freq: int
        The number of frequency bins in the spectrum.
    n_rings: int
        The number of rings used to model the accretion disc.

    Returns
    -------
    s: pd.DataFrame
        The accretion disc spectrum. If in frequency units, the columns are
        "Freq." (Hz) and "Lum" (ergs/s/cm^2/Hz). If in wavelength units, the columns are
        "Lambda" (A) and "Lum" (ergs/s/cm^2/A).
    """

    if freq_units:
        xlabel = "Freq."
    else:
        xlabel = "Lambda"

    radii_range = np.logspace(np.log10(r_in), np.log10(r_out), n_rings)
    frequency_range = np.linspace(freq_min, freq_max, n_freq)

    spectrum = {xlabel: frequency_range}
    lum = np.zeros_like(frequency_range)

    # TODO: this can probably be vectorised to be faster

    for i in range(n_rings - 1):
        # Use midpoint of annulus as point on r grid
        r = (radii_range[i + 1] + radii_range[i]) * 0.5
        area_annulus = PI * (radii_range[i + 1]**2 - radii_range[i]**2)

        t_eff = alpha_disc_effective_temperature(r, r_in, m_co, mdot)

        if freq_units:
            f = planck_nu(t_eff, frequency_range)
        else:
            f = planck_lambda(t_eff, frequency_range)

        lum += f * area_annulus * PI

    spectrum["Lum."] = lum

    return spectrum


def eddington_accretion_limit(mbh, efficiency):
    """Calculate the Eddington accretion limit for a black hole. Note that the
    accretion rate can be larger than the Eddington accretion rate. See, for
    example, Foundations of High-Energy Astrophysics by Mario Vietri.

    Parameters
    ----------
    mbh: float
        The mass of the black hole in units of msol.
    efficiency: float
        The efficiency of the accretion process. Less than 1.

    Returns
    -------
    The Eddington accretion rate in units of grams / second.
    """

    mbh *= MSOL

    return (4 * PI * G * mbh * MPROT) / (efficiency * C * THOMPSON)


def eddington_luminosity_limit(mbh):
    """Calculate the Eddington luminosity for accretion onto a black hole.

    Parameters
    ----------
    mbh: float
        The mass of the black hole in units of msol.

    Returns
    -------
    The Eddington luminosity for the black hole in units of ergs / second.
    """

    mbh *= MSOL

    return (4 * PI * G * mbh * C * MPROT) / THOMPSON


def modified_eddington_alpha_disc_effective_temperature(ri, m_co, mdot):
    """The effective temperature profile from Strubbe and Quataert 2009.

    Parameters
    ----------
    ri: np.ndarray or float
        An array of radii or a single radius to calculate the temperature at.
    m_co: float
        The mass of the central object in units of solar masses.
    mdot: float
        The accretion rate onto the central object in units of solar masses per
        year.

    Returns
    -------
    teff: np.ndarray or float
        The effective temperature at the provided radius or radii.
    """

    risco = innermost_stable_circular_orbit(m_co)
    rg = gravitational_radius(m_co)
    ledd = eddington_luminosity_limit(m_co)

    m_co *= MSOL
    mdot *= MSOL_PER_YEAR

    with np.errstate(all="ignore"):
        fnt = 1 - np.sqrt(risco / ri)
        teff4 = (3 * G * m_co * mdot * fnt) / (8 * PI * ri**3 * STEFAN_BOLTZMANN)
        teff4 *= (0.5 + (0.25 + 6 * fnt * (mdot * C**2 / ledd)**2 * (ri / rg)**-2)**0.5)**-1

    return teff4**0.25
