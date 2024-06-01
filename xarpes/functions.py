# Copyright (C) 2024 xARPES Developers
# This program is free software under the terms of the GNU GPLv2 license.

"""Separate functions used in conjunction with various classes."""

import numpy as np
from scipy.optimize import leastsq

def error_function(p, xdata, ydata, function, extra_args):
    r"""The error function used inside the fit_leastsq function.

    Parameters
    ----------
    p : ndarray
        Array of parameters during the optimization
    xdata : ndarray
        Array of abscissa values the function is evaluated on
    ydata : ndarray
        Outcomes on ordinate the evaluated function is compared to
    function : function
        Function or class with call method to be evaluated
    extra_args : 
        Arguments provided to function that should not be optimized

    Returns
    -------

    residual : 
    
    """   
    residual = function(xdata, *p, extra_args) - ydata
    return residual


def fit_leastsq(p0, xdata, ydata, function, extra_args):
    r"""Wrapper arround scipy.optimize.leastsq.

    Parameters
    ----------
    p0 : ndarray
        Initial guess for parameters to be optimized
    xdata : ndarray
        Array of abscissa values the function is evaluated on
    ydata : ndarray
        Outcomes on ordinate the evaluated function is compared to
    function : function
        Function or class with call method to be evaluated
    extra_args : 
        Arguments provided to function that should not be optimized

    Returns
    -------
    pfit_leastsq : ndarray
        Array containing the optimized parameters
    perr_leastsq : ndarray
        Covariance matrix of the optimized parameters
    """
    pfit, pcov, infodict, errmsg, success = leastsq(
        error_function, p0, args=(xdata, ydata, function, extra_args),
        full_output=1)

    if (len(ydata) > len(p0)) and pcov is not None:
        s_sq = (error_function(pfit, xdata, ydata, function,
                               extra_args) ** 2).sum() / (len(ydata) - len(p0))
        pcov = pcov * s_sq
    else:
        pcov = np.inf

    error = []
    for i in range(len(pfit)):
        try:
          error.append(np.absolute(pcov[i][i]) ** 0.5)
        except:
          error.append(0.00)
    pfit_leastsq = pfit
    perr_leastsq = np.array(error)

    return pfit_leastsq, perr_leastsq