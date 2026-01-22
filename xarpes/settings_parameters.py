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

# ---------------- Defaults for MEM / chi2kink alpha-selection ----------------

mem_defaults = {
    "method": "chi2kink",
    "parts": "both",
    "iter_max": 1e4,
    "alpha_min": 1.0,
    "alpha_max": 9.0,
    "alpha_num": 10,
    "ecut_left": 0.0,
    "ecut_right": None,
    "f_chi_squared": None,
    "W": None,
    "power": 4,
    "mu": 1.0,
    "omega_S": 1.0,
    "sigma_svd": 1e-4,
    "t_criterion": 1e-8,
    "a_guess": 1.0,
    "b_guess": 2.5,
    "c_guess": 3.0,
    "d_guess": 1.5,
    "h_n": None, 
    "impurity_magnitude": 0.0,
    "lambda_el": 0.0,
}

# ---------------- Defaults for bayesian_loop optimisation --------------------

loop_defaults = {
    "converge_iters": 50,
    "tole": 1e-2,
    "scale_vF": 1.0,
    "scale_mb": 1.0,
    "scale_imp": 1.0,
    "scale_kF": 1.0,
    "scale_lambda_el": 1.0,
    "scale_hn": 1.0,
    "opt_iter_max": 1e4,
    "rollback_steps": 10,
    "max_retries": 100,
    "relative_best": 10.0,
    "min_steps_for_regression": 25,
}