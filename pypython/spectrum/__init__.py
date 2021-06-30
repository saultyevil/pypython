#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utility functions for creating and analyzing spectra.

Python has the ability to create a "delay_dump" file, which is a dump of
all extracted photons. Using this file, a spectrum can be created with
various physical processes remove, i.e. resonance scattering. This
module also includes functions for measuring line width and equivalent
widths.
"""

from pypython.spectrum import create, lines
