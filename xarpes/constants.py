# Copyright (C) 2025 xARPES Developers
# This program is free software under the terms of the GNU GPLv3 license.

"""Constants used elsewhere in xARPES."""

from scipy.constants import Boltzmann, elementary_charge, hbar, m_e
import numpy as np

# Physical constants
KILO = 1e3 # 1000 [-]
FWHM2STD = np.sqrt(8 * np.log(2)) # Convert FWHM to std [-]
K_B = Boltzmann / elementary_charge # Boltzmann constant [eV / K]
PREF = (hbar**2 / (2 * m_e)) / elementary_charge * 1e20 # hbar^2 / (2 m_e) [eV Angstrom^2]