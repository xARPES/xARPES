# Copyright (C) 2025 xARPES Developers
# This program is free software under the terms of the GNU GPLv3 license.

"""Separate functions mostly used in conjunction with various classes."""

import numpy as np

def resolve_param_name(params, label, pname):
    """
    Try to find the lmfit param key corresponding to this component `label`
    and bare parameter name `pname` (e.g., 'amplitude', 'peak', 'broadening').
    Works with common token separators.
    """
    import re
    names = list(params.keys())
    # Fast exact candidates
    candidates = (
        f"{pname}_{label}", f"{label}_{pname}",
        f"{pname}:{label}", f"{label}:{pname}",
        f"{label}.{pname}", f"{label}|{pname}",
        f"{label}-{pname}", f"{pname}-{label}",
    )
    for c in candidates:
        if c in params:
            return c

    # Regex fallback: label and pname as tokens in any order
    esc_l = re.escape(str(label))
    esc_p = re.escape(str(pname))
    tok = r"[.:/_\-]"  # common separators
    pat = re.compile(rf"(^|{tok}){esc_l}({tok}|$).*({tok}){esc_p}({tok}|$)")
    for n in names:
        if pat.search(n):
            return n

    # Last resort: unique tail match on pname that also contains the label somewhere
    tails = [n for n in names if n.endswith(pname) and str(label) in n]
    if len(tails) == 1:
        return tails[0]

    # Give up
    return None


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
             kinetic_energy, hnuminPhi, matrix_element=None,
             element_names=None):
    r"""
    """
    from scipy.ndimage import gaussian_filter
    from xarpes.distributions import Dispersion

    if matrix_element is not None:
        matrix_parameters = {}
        for name in element_names:
            if name in parameters:
                matrix_parameters[name] = parameters[name].value

    new_distributions = build_distributions(new_distributions, parameters)

    extend, step, numb = extend_function(xdata, angle_resolution)

    model = np.zeros_like(extend)

    for dist in new_distributions:
        if getattr(dist, 'class_name', type(dist).__name__) == \
            'SpectralQuadratic':
            part = dist.evaluate(extend, kinetic_energy, hnuminPhi)
        else:
            part = dist.evaluate(extend)

        if (matrix_element is not None) and isinstance(dist, Dispersion):
            part *= matrix_element(extend, **matrix_parameters)

        model += part

    model = gaussian_filter(model, sigma=step)[numb:-numb if numb else None]
    return model - ydata


def extend_function(abscissa_range, abscissa_resolution):
    r"""TBD
    """
    from .constants import FWHM2STD
    from . import settings_parameters as xprs
    step_size = np.abs(abscissa_range[1] - abscissa_range[0])
    step = abscissa_resolution / (step_size * FWHM2STD)
    numb = int(xprs.sigma_extend * step)
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
    from scipy.ndimage import gaussian_filter

    if resolution:
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


def MEM_core():
    r"""
    Extracts the unscaled Eliashberg function for a given value of the Lagrange 
    multiplier alpha. It also returns the reconstruction F.
    In essence, this function applies the Newton method to solve 
    """
    return 0



def download_examples():
    """Downloads the examples folder from the main xARPES repository only if it
    does not already exist in the current directory. Prints executed steps and a
    final cleanup/failure message.

    Returns
    -------
    0 or 1 : int
        Returns 0 if the execution succeeds, 1 if it fails.
    """
    import requests
    import zipfile
    import os
    import shutil
    import io
    import jupytext
    import tempfile
    import re

    # Main xARPES repo (examples live under /examples there)
    repo_url = "https://github.com/xARPES/xARPES"
    output_dir = "."  # Directory from which the function is called

    # Target 'examples' directory in the user's current location
    final_examples_path = os.path.join(output_dir, "examples")
    if os.path.exists(final_examples_path):
        print("Warning: 'examples' folder already exists. "
              "No download will be performed.")
        return 1  # Exit the function if 'examples' directory exists

    # --- Determine version from xarpes.__init__.__version__ -----------------
    try:
        # Import inside the function, avoiding circular imports at import time
        import xarpes as _xarpes
        raw_version = getattr(_xarpes, "__version__", None)
    except Exception as exc:
        print(f"Warning: could not import xarpes to determine version: {exc}")
        raw_version = None

    tag_version = None
    if raw_version is not None:
        raw_version = str(raw_version)
        # Strip dev/local suffixes so that '0.3.3.dev1' or '0.3.3+0.gHASH'
        # maps to the tag 'v0.3.3'. If you use plain '0.3.3' already, this is 
        # a no-op.
        m = re.match(r"(\d+\.\d+\.\d+)", raw_version)
        if m:
            tag_version = m.group(1)
        else:
            tag_version = raw_version

        print(f"Determined xARPES version from __init__: {raw_version} "
              f"(using tag version '{tag_version}').")
    else:
        print("Warning: xarpes.__version__ is not defined; will skip "
        "tag-based download and try the main branch only.")

    # --- Build refs and use for–else to try them in order -------------------
    repo_parts = repo_url.replace("https://github.com/", "").rstrip("/")

    refs_to_try = []
    if tag_version is not None:
        refs_to_try.append(f"tags/v{tag_version}")  # version-matched examples
    refs_to_try.append("heads/main")                # fallback: latest examples

    response = None
    for ref in refs_to_try:
        zip_url = f"https://github.com/{repo_parts}/archive/refs/{ref}.zip"
        print(f"Attempting to download examples from '{ref}':\n  {zip_url}")
        response = requests.get(zip_url)

        if response.status_code == 200:
            if ref.startswith("tags/"):
                print(f"Successfully downloaded examples from tagged release "
                      f"'v{tag_version}'.")
            else:
                print("Tagged release not available; using latest examples "
                "from the 'main' branch instead.")
            break
        else:
            print("Failed to download from this ref. HTTP status code: "
                  f"{response.status_code}")
    else:
        # for–else: only executed if we never hit 'break'
        print("Error: could not download examples from any ref "
              f"(tried: {', '.join(refs_to_try)}).")
        return 1

    # At this point, 'response' holds a successful download
    zip_file_bytes = io.BytesIO(response.content)

    # --- Extract into a temporary directory to avoid polluting CWD ----------
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_file_bytes, "r") as zip_ref:
            zip_ref.extractall(tmpdir)
            # First member gives us the top-level directory in the archive,
            # typically something like 'xARPES-0.3.3/' or 'xARPES-main/'.
            first_member = zip_ref.namelist()[0]

        top_level_dir = first_member.split("/")[0]
        main_folder_path = os.path.join(tmpdir, top_level_dir)
        examples_path = os.path.join(main_folder_path, "examples")

        if not os.path.exists(examples_path):
            print("Error: downloaded archive does not contain an 'examples' "
            "directory.")
            return 1

        # Move the 'examples' directory to the target location in the CWD
        shutil.move(examples_path, final_examples_path)
        print(f"'examples' subdirectory moved to {final_examples_path}")

        # Convert all .Rmd files in the examples directory to .ipynb
        # and delete the .Rmd files
        for dirpath, dirnames, filenames in os.walk(final_examples_path):
            for filename in filenames:
                if filename.endswith(".Rmd"):
                    full_path = os.path.join(dirpath, filename)
                    jupytext.write(
                        jupytext.read(full_path),
                        full_path.replace(".Rmd", ".ipynb")
                    )
                    os.remove(full_path)  # Deletes .Rmd file afterwards
                    print(f"Converted and deleted {full_path}")

    # Temporary directory is cleaned up automatically
    print("Cleaned up temporary files.")
    return 0


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