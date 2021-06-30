#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standard models.

Models used in the PYTHON collaboration, including:
    - Schlossman and Vitello 1993 wind model
"""

import numpy as np
# from scipy.optimize import brentq

from pypython.constants import MSOL, MSOL_PER_YEAR, G


class SV93Wind:
    """Create a Schlossman & Vitello (1993) wind.

    This will (or try to) generate a grid of velocities and densities,
    given the parameters.
    """
    def __init__(self, m_co, mdot, r_min, r_max, theta_min, theta_max, accel_length, accel_exp, v_inf, gamma, v0=6e5):
        """Create an SV wind.

        Parameters
        ----------
        m_co:
            The mass of the central object, in solar masses.
        mdot:
            The mass loss rate of the wind, in solar masses per year.
        r_min:
            The inner radius of the wind, in cm.
        r_max:
            The outer radius of the wind, in cm.
        theta_min:
            The opening angle.
        theta_max:
            The opening angle.
        accel_length:
            The distance it takes to reach half the terminal velocity, in cm.
        accel_exp:
            The exponent of the velocity law, controls how fast the streamline
            accelerates to the acceleration length scale.
        v_inf:
            The terminal velocity, in units of escape velocity.
        """

        self.v0 = v0
        self.gamma = gamma
        self.m_co = m_co * MSOL
        self.mdot = mdot * MSOL_PER_YEAR
        self.r_min = r_min
        self.r_max = r_max
        self.theta_min = np.deg2rad(theta_min)
        self.theta_max = np.deg2rad(theta_max)
        self.accel_length = accel_length
        self.accel_exp = accel_exp
        self.v_inf = v_inf

    def find_theta(self, r0):
        """Determine the angle at which the wind emerges from at a special
        radius r from the disk surface."""

        x = ((r0 - self.r_min) / (self.r_max - self.r_min))**self.gamma

        if r0 <= self.r_min:
            theta = np.arctan(np.tan(self.theta_max * r0 / self.r_min))
        elif r0 >= self.r_max:
            theta = self.theta_max
        else:
            theta = self.theta_min + (self.theta_max - self.theta_min) * x

        return theta

    def r0_guess_func(self, r, x):
        """Note that r is a position along the disk."""

        theta = self.find_theta(r)
        rho = np.sqrt(x[0]**2 + x[1]**2)
        rho_guess = r + np.tan(theta) * x[2]

        return rho_guess - rho  # We want to make this zero

    # def find_r0(self, x):
    #     """Determine r0 for a point in the x, y plane."""
    #
    #     # If the vector is in the x-y plane, then this is simple
    #     if x[2] == 0:
    #         return np.sqrt(x[0]**2 + x[1]**2)
    #
    #     # For when the vector is not solely in the x-y plane
    #     rho_min = self.r_min + x[2] * np.tan(self.theta_min)
    #     rho_max = self.r_max + x[2] * np.tan(self.theta_max)
    #     rho = np.sqrt(x[0]**2 + x[1]**2)
    #
    #     if rho <= rho_min:
    #         return self.r_min * rho / rho_min
    #     elif rho >= rho_max:
    #         return self.r_max * rho - rho_max
    #     else:
    #         return brentq(
    #             self.r0_guess_func,
    #             self.r_min,
    #             self.r_max,
    #             args=x,
    #         )

    def escape_velocity(self, r0):
        """Calculate the escape velocity at a point r0."""

        return np.sqrt(2 * G * self.m_co / r0)

    def polodial_velocity(self, dist, r0):
        """Calculate the polodial velocity for a polodial distance l along a
        wind stream line with fixed."""
        tmp = (dist / self.accel_length)**self.accel_exp
        v_term = self.v_inf * self.escape_velocity(r0)
        vl = self.v0 + (v_term - self.v0) * (tmp / (tmp + 1))

        return vl

    # def velocity_vector(self, x):
    #     """Determine the 3d velocity vector in cartesian coordinates."""
    #     r0 = self.find_r0(x)
    #     theta = self.find_theta(r0)
    #
    #     r = np.sqrt(x[0]**2 + x[1]**2)
    #     pol_dist = np.sqrt((r - r0)**2 + x[2]**2)
    #     vl = self.polodial_velocity(pol_dist, r0)
    #
    #     v = np.zeros(3)
    #     v[0] = vl * np.sin(theta)
    #     if r > 0:
    #         v[1] = np.sqrt(G * self.m_co * r0) / r
    #     else:
    #         v[1] = 0
    #     v[2] = np.abs(vl * np.cos(theta))
    #
    #     return v
