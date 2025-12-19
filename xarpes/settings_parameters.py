# Copyright (C) 2025 xARPES Developers
# This program is free software under the terms of the GNU GPLv3 license.

"""User-configurable numerical parameters for xARPES."""

# Extend data range by this many Gaussian sigmas
sigma_extend = 5.0

# Gaussian confidence level expressed in "sigma"
sigma_confidence = 2.0

def parameter_settings(new_sigma_extend=None, new_sigma=None):
    """
    Configure global numerical parameters for xARPES.

    Parameters
    ----------
    new_sigma_extend : float or None
        Number of Gaussian sigmas used to extend arrays before convolution.
    new_sigma : float or None
        Gaussian confidence level expressed in units of sigma
        (e.g. 1, 2, 3). Default is 2.
    """
    global sigma_extend, sigma_confidence

    if new_sigma_extend is not None:
        sigma_extend = float(new_sigma_extend)

    if new_sigma is not None:
        sigma_confidence = float(new_sigma)
