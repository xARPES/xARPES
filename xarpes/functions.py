# Copyright (C) 2025 xARPES Developers
# This program is free software under the terms of the GNU GPLv3 license.

"""Separate functions mostly used in conjunction with various classes."""

import numpy as np

# Physical constants, default parameters
fwhm_to_std = np.sqrt(8 * np.log(2)) # Convert FWHM to std [-]
sigma_extend = 5 # Extend data range by "5 sigma"


def build_distributions(distributions, parameters):
    r"""TBD
    """
    for dist in distributions:
        if dist.class_name == 'Constant':
            dist.offset = parameters['offset_' + dist.label].value
        elif dist.class_name == 'Linear':
            dist.offset = parameters['offset_' + dist.label].value
            dist.slope = parameters['slope_' + dist.label].value
        elif dist.class_name == 'SpectralLinear':
            dist.amplitude = parameters['amplitude_' + dist.label].value
            dist.peak = parameters['peak_' + dist.label].value
            dist.broadening = parameters['broadening_' + dist.label].value
        elif dist.class_name == 'SpectralQuadratic':
            dist.amplitude = parameters['amplitude_' + dist.label].value
            dist.peak = parameters['peak_' + dist.label].value
            dist.broadening = parameters['broadening_' + dist.label].value
    return distributions


def construct_parameters(distribution_list, matrix_args=None):
    r"""TBD
    """
    from lmfit import Parameters

    parameters = Parameters()

    for dist in distribution_list:
        if dist.class_name == 'Constant':
            parameters.add(name='offset_' + dist.label, value=dist.offset)
        elif dist.class_name == 'Linear':
            parameters.add(name='offset_' + dist.label, value=dist.offset)
            parameters.add(name='slope_' + dist.label, value=dist.slope)
        elif dist.class_name == 'SpectralLinear':
            parameters.add(name='amplitude_' + dist.label,
                           value=dist.amplitude, min=0)
            parameters.add(name='peak_' + dist.label, value=dist.peak)
            parameters.add(name='broadening_' + dist.label,
                           value=dist.broadening, min=0)
        elif dist.class_name == 'SpectralQuadratic':
            parameters.add(name='amplitude_' + dist.label,
                           value=dist.amplitude, min=0)
            parameters.add(name='peak_' + dist.label, value=dist.peak)
            parameters.add(name='broadening_' + dist.label,
                           value=dist.broadening, min=0)

    if matrix_args is not None:
        element_names = list()
        for key, value in matrix_args.items():
            parameters.add(name=key, value=value)
            element_names.append(key)
        return parameters, element_names
    else:
        return parameters


def residual(parameters, xdata, ydata, angle_resolution, new_distributions,
             kinetic_energy, hnuminphi, matrix_element=None,
             element_names=None):
    r"""
    """
    from scipy.ndimage import gaussian_filter

    if matrix_element is not None:
        matrix_parameters = {}
        for name in element_names:
            if name in parameters:
                matrix_parameters[name] = parameters[name].value

    new_distributions = build_distributions(new_distributions, parameters)

    extend, step, numb = extend_function(xdata, angle_resolution)

    model = np.zeros_like(extend)

    for dist in new_distributions:
        if hasattr(dist, 'index') and matrix_element is not None:
            if dist.class_name == 'SpectralQuadratic':
                model += dist.evaluate(extend, kinetic_energy, hnuminphi) \
                * matrix_element(extend, **matrix_parameters)
            else:
                model += dist.evaluate(extend) * matrix_element(extend,
                                                 **matrix_parameters)
        else:
            model += dist.evaluate(extend)

    model = gaussian_filter(model, sigma=step)[numb:-numb if numb else None]
    return model - ydata


def extend_function(abscissa_range, abscissa_resolution):
    r"""TBD
    """
    step_size = np.abs(abscissa_range[1] - abscissa_range[0])
    step = abscissa_resolution / (step_size * fwhm_to_std)
    numb = int(sigma_extend * step)
    extend = np.linspace(abscissa_range[0] - numb * step_size,
                         abscissa_range[-1] + numb * step_size,
                         len(abscissa_range) + 2 * numb)
    return extend, step, numb


def error_function(p, xdata, ydata, function, resolution, yerr, extra_args):
    r"""The error function used inside the fit_leastsq function.

    Parameters
    ----------
    p : ndarray
        Array of parameters during the optimization.
    xdata : ndarray
        Abscissa values the function is evaluated on.
    ydata : ndarray
        Measured values to compare to.
    function : callable
        Function or class with __call__ method to evaluate.
    resolution : float or None
        Convolution resolution (sigma), if applicable.
    yerr : ndarray
        Standard deviations of ydata.
    extra_args : tuple
        Additional arguments passed to function.

    Returns
    -------
    residual : ndarray
        Normalized residuals between model and ydata.
    """
    if resolution:
        from scipy.ndimage import gaussian_filter
        extend, step, numb = extend_function(xdata, resolution)
        model = gaussian_filter(function(extend, *p, *extra_args),
                                sigma=step)
        model = model[numb:-numb if numb else None]
    else:
        model = function(xdata, *p, *extra_args)

    residual = (model - ydata) / yerr
    return residual


def fit_leastsq(p0, xdata, ydata, function, resolution=None,
                yerr=None, *extra_args):
    r"""Wrapper around scipy.optimize.leastsq.

    Parameters
    ----------
    p0 : ndarray
        Initial guess for parameters to be optimized.
    xdata : ndarray
        Abscissa values the function is evaluated on.
    ydata : ndarray
        Measured values to compare to.
    function : callable
        Function or class with __call__ method to evaluate.
    resolution : float or None, optional
        Convolution resolution (sigma), if applicable.
    yerr : ndarray or None, optional
        Standard deviations of ydata. Defaults to ones if None.
    extra_args : tuple
        Additional arguments passed to the function.

    Returns
    -------
    pfit_leastsq : ndarray
        Optimized parameters.
    pcov : ndarray or float
        Scaled covariance matrix of the optimized parameters.
        If the covariance could not be estimated, returns np.inf.
    """
    from scipy.optimize import leastsq

    if yerr is None:
        yerr = np.ones_like(ydata)

    pfit, pcov, infodict, errmsg, success = leastsq(
        error_function,
        p0,
        args=(xdata, ydata, function, resolution, yerr, extra_args),
        full_output=1
    )

    if (len(ydata) > len(p0)) and pcov is not None:
        s_sq = (
            error_function(pfit, xdata, ydata, function, resolution,
                           yerr, extra_args) ** 2
        ).sum() / (len(ydata) - len(p0))
        pcov *= s_sq
    else:
        pcov = np.inf

    return pfit, pcov


def download_examples():
    """Downloads the examples folder from the xARPES code only if it does not
    already exist. Prints executed steps and a final cleanup/failure message.

    Returns
    -------
    0, 1 : int
        Returns 0 if the execution succeeds, 1 if it fails.
    """
    import requests
    import zipfile
    import os
    import shutil
    import io
    import jupytext

    repo_url = 'https://github.com/xARPES/xARPES_examples'
    output_dir = '.' # Directory from which the function is called

    # Check if 'examples' directory already exists
    final_examples_path = os.path.join(output_dir, 'examples')
    if os.path.exists(final_examples_path):
        print("Warning: 'examples' folder already exists. " +
        'No download will be performed.')
        return 1 # Exit the function if 'examples' directory exists

    # Proceed with download if 'examples' directory does not exist
    repo_parts = repo_url.replace('https://github.com/', '').rstrip('/')
    zip_url = f'https://github.com/{repo_parts}/archive/refs/heads/main.zip'

    # Make the HTTP request to download the zip file
    print(f'Downloading {zip_url}')
    response = requests.get(zip_url)
    if response.status_code == 200:
        zip_file_bytes = io.BytesIO(response.content)

        with zipfile.ZipFile(zip_file_bytes, 'r') as zip_ref:
            zip_ref.extractall(output_dir)

        # Path to the extracted main folder
        main_folder_path = os.path.join(output_dir,
                repo_parts.split('/')[-1] + '-main')
        examples_path = os.path.join(main_folder_path, 'examples')

        # Move the 'examples' directory to the target location
        if os.path.exists(examples_path):
            shutil.move(examples_path, final_examples_path)
            print(f"'examples' subdirectory moved to {final_examples_path}")
            # Convert all .Rmd files in the examples directory to .ipynb
            # and delete the .Rmd files
            for dirpath, dirnames, filenames in os.walk(final_examples_path):
                for filename in filenames:
                    if filename.endswith('.Rmd'):
                        full_path = os.path.join(dirpath, filename)
                        jupytext.write(jupytext.read(full_path),
                                       full_path.replace('.Rmd', '.ipynb'))
                        os.remove(full_path) # Deletes .Rmd file afterwards
                        print(f'Converted and deleted {full_path}')

        # Remove the rest of the extracted content
        shutil.rmtree(main_folder_path)
        print(f'Cleaned up temporary files in {main_folder_path}')
        return 0
    else:
        print('Failed to download the repository. Status code: '
             f'{response.status_code}')
        return 1

def set_script_dir():
    r"""This function sets the directory such that the xARPES code can be
    executed either inside IPython environments or as .py scripts from
    arbitrary locations.
    """
    import os
    import inspect
    try:
        # This block checks if the script is running in an IPython environment
        cfg = get_ipython().config
        script_dir = os.getcwd()
    except NameError:
        # If not in IPython, get the caller's file location
        frame = inspect.stack()[1]
        module = inspect.getmodule(frame[0])
        script_dir = os.path.dirname(os.path.abspath(module.__file__))
    except:
        # If __file__ isn't defined, fall back to current working directory
        script_dir = os.getcwd()

    return script_dir