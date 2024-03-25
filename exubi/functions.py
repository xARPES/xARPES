# File containing basic functionalities used by other routines.

import numpy as np
from scipy.optimize import leastsq

def error_function(p, x, y, function, extra_args):
    """
    The error function used by SciPy.
    """
    return function(x, *p, extra_args) - y


def fit_leastsq(p0, xdata, ydata, function, extra_args): 
    """
    Wrapper arround scipy.optimize.leastsq.
    """
    pfit, pcov, infodict, errmsg, success = \
        leastsq(error_function, p0, args=(xdata, ydata, function, extra_args), full_output=1)
    
    if (len(ydata) > len(p0)) and pcov is not None:
        s_sq = (error_function(pfit, xdata, ydata, function, extra_args)**2).sum()/(len(ydata)-len(p0))
        pcov = pcov * s_sq
    else:
        pcov = np.inf

    error = [] 
    for i in range(len(pfit)):
        try:
          error.append(np.absolute(pcov[i][i])**0.5)
        except:
          error.append(0.00)
    pfit_leastsq = pfit
    perr_leastsq = np.array(error) 
    
    return pfit_leastsq, perr_leastsq 