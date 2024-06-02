# Copyright (C) 2024 xARPES Developers
# This program is free software under the terms of the GNU GPLv3 license.

"""Separate functions mostly used in conjunction with various classes."""

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
    
    repo_url = 'https://github.com/xARPES/xARPES_examples'
    output_dir = '.'  # Directory from which the function is called
    
    # Check if 'examples' directory already exists
    final_examples_path = os.path.join(output_dir, 'examples')
    if os.path.exists(final_examples_path):
        print("Warning: 'examples' folder already exists. No download will be performed.")
        return 1 # Exit the function if 'examples' directory exists

    # Proceed with download if 'examples' directory does not exist
    repo_parts = repo_url.replace("https://github.com/", "").rstrip('/')
    zip_url = f"https://github.com/{repo_parts}/archive/refs/heads/main.zip"

    # Make the HTTP request to download the zip file
    print(f"Downloading {zip_url}")
    response = requests.get(zip_url)
    if response.status_code == 200:
        zip_file_bytes = io.BytesIO(response.content)

        with zipfile.ZipFile(zip_file_bytes, 'r') as zip_ref:
            zip_ref.extractall(output_dir)

        # Path to the extracted main folder
        main_folder_path = os.path.join(output_dir, repo_parts.split('/')[-1] + '-main')
        examples_path = os.path.join(main_folder_path, 'examples')

        # Move the 'examples' directory to the target location if it was extracted
        if os.path.exists(examples_path):
            shutil.move(examples_path, final_examples_path)
            print(f"'examples' subdirectory moved to {final_examples_path}")
        else:
            print("'examples' subdirectory not found in the repository.")

        # Remove the rest of the extracted content
        shutil.rmtree(main_folder_path)
        print(f"Cleaned up temporary files in {main_folder_path}")
        return 0
    else:
        print(f"Failed to download the repository. Status code: {response.status_code}")
        return 1


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
        Residual between evaluated function and ydata
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
    import numpy as np
    from scipy.optimize import leastsq
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