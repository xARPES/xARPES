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
    r"""The error function used inside the fit_least_squares function.

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


def fit_least_squares(p0, xdata, ydata, function, resolution=None, yerr=None,
                *extra_args, bounds=None):
    r"""Least-squares fit using `scipy.optimize.least_squares`.

    Default behavior is Levenberg–Marquardt (`method="lm"`) when unbounded.
    If `bounds` is provided, switches to trust-region reflective (`"trf"`).

    Returns (pfit, pcov) in the same style as the old `leastsq` wrapper.
    """
    from scipy.optimize import least_squares

    if yerr is None:
        yerr = np.ones_like(ydata)

    def _residuals(p):
        return error_function(
            p, xdata, ydata, function, resolution, yerr, extra_args
        )

    if bounds is None:
        res = least_squares(_residuals, p0, method="lm")
    else:
        res = least_squares(_residuals, p0, method="trf", bounds=bounds)

    pfit = res.x

    m = len(ydata)
    n = pfit.size

    if (m > n) and res.jac is not None and res.jac.size:
        resid = res.fun
        s_sq = (resid ** 2).sum() / (m - n)

        try:
            jtj = res.jac.T @ res.jac
            pcov = np.linalg.inv(jtj) * s_sq
        except np.linalg.LinAlgError:
            pcov = np.inf
    else:
        pcov = np.inf

    return pfit, pcov


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


def MEM_core(dvec, model_in, uvec, mu, alpha, wvec, V_Sigma, U,
             t_criterion, iter_max):
    r"""
    Implementation of Bryan's algorithm (not to be confused with Bryan's
    'method' for determining the Lagrange multiplier alpha. For details, see
    Eur. Biophys. J. 18, 165 (1990).
    """
    import numpy as np
    import warnings

    spectrum_in = model_in * np.exp(U @ uvec)  # Eq. 9
    alphamu = alpha + mu

    converged = False
    iter_count = 0
    while not converged and iter_count < iter_max:

        T = V_Sigma @ (U.T @ spectrum_in)  # Below Eq. 7
        gvec = V_Sigma.T @ (wvec * (T - dvec))  # Eq. 10
        M = V_Sigma.T @ (wvec[:, None] * V_Sigma)  # Above Eq. 11
        K = U.T @ (spectrum_in[:, None] * U)  # Above Eq. 11

        xi, P = np.linalg.eigh(K)  # Eq. 13
        sqrt_xi = np.sqrt(xi)
        P_sqrt_xi = P * sqrt_xi[None, :]
        A = P_sqrt_xi.T @ (M @ P_sqrt_xi)  # Between Eqs. 13 and 14
        Lambda, R = np.linalg.eigh(A)  # Eq. 14
        Y_inv = R.T @ (sqrt_xi[:, None] * P.T)  # Below Eq. 15

        # From Eq. 16:
        Y_inv_du = -(Y_inv @ (alpha * uvec + gvec)) / (alphamu + Lambda)
        d_uvec = (
            -alpha * uvec - gvec - M @ (Y_inv.T @ Y_inv_du)
        ) / alphamu  # Eq. 20

        uvec += d_uvec
        spectrum_in = model_in * np.exp(U @ uvec)  # Eq. 9

        # Convergence block: Section 2.3
        alpha_K_u = alpha * (K @ uvec)  # Skipping the minus sign twice
        K_g = K @ gvec
        tcon = (
            2 * np.linalg.norm(alpha_K_u + K_g)**2
            / (np.linalg.norm(alpha_K_u) + np.linalg.norm(K_g))**2
        )
        converged = (tcon < t_criterion)

        iter_count += 1

    if not converged:
        with warnings.catch_warnings():
            warnings.simplefilter("always", RuntimeWarning)
            warnings.warn(
                f"MEM_core did not converge within iter_max={iter_max} "
                f"(performed {iter_count} iterations).",
                category=RuntimeWarning,
                stacklevel=2,
            )

    return spectrum_in, uvec


def bose_einstein(omega, k_BT):
    """Bose-Einstein distribution n_B(omega) for k_BT > 0 and omega >= 0."""
    x_over = np.log(np.finfo(float).max)  # ~709.78 for float64

    x = omega / k_BT

    out = np.empty_like(omega, dtype=float)

    momega0 = (omega == 0)
    if np.any(momega0):
        out[momega0] = np.inf

    mpos_big = (x > x_over) & (omega != 0)
    if np.any(mpos_big):
        out[mpos_big] = 0.0

    mnorm = (omega != 0) & ~mpos_big
    if np.any(mnorm):
        out[mnorm] = 1.0 / np.expm1(x[mnorm])

    return out


def fermi(omega, k_BT):
    """Fermi-Dirac distribution f(omega) for k_BT > 0 and omega >= 0.
    Could potentially be made a core block of the FermiDirac distribution."""
    x_over = np.log(np.finfo(float).max)  # ~709.78 for float64

    x = omega / k_BT
    out = np.empty_like(omega, dtype=float)

    mover = x > x_over
    out[mover] = 0.0

    mnorm = ~mover
    y = np.exp(-x[mnorm])
    out[mnorm] = y / (1.0 + y)

    return out


def create_kernel_function(enel, omega, k_BT):
    r"""Kernel function. Eq. 17 from https://arxiv.org/abs/2508.13845.

    Returns
    -------
    K : ndarray, complex
        Shape (enel.size, omega.size) if enel and omega are 1D.
    """
    from scipy.special import digamma

    enel = enel[:, None]     # (Ne, 1)
    omega = omega[None, :]   # (1, Nw)

    denom = 2.0 * np.pi * k_BT

    K = (digamma(0.5 - 1j * (enel - omega) / denom)
         - digamma(0.5 - 1j * (enel + omega) / denom)
         - 2j * np.pi * (bose_einstein(omega, k_BT) + 0.5))

    return K


def singular_value_decomposition(kernel, sigma_svd):
    r"""
    Some papers use kernel = U Sigma V^T; we follow Bryan's algorithm.
    """
    V, Sigma, U_transpose = np.linalg.svd(kernel)
    U = U_transpose.T
    Sigma = Sigma[Sigma > sigma_svd]
    s_reduced = Sigma.size
    V = V[:, :s_reduced]
    U = U[:, :s_reduced]
    V_Sigma = V * Sigma[None, :]
    
    uvec = np.zeros(s_reduced)
    
    print('Dimensionality has been reduced from a matrix of rank ' + str(min(kernel.shape)) +
          ' to ' + str(int(s_reduced)) + ' in the singular space.')
          
    return V_Sigma, U, uvec


def create_model_function(omega, omega_I, omega_M, omega_S, h_n):
    r"""Piecewise model m_n(omega) defined on the omega grid.

    Implements the piecewise definition in the figure, interpreting
    omega_min/max as omega.min()/omega.max().

    Parameters
    ----------
    omega : ndarray
        Frequency grid (assumed sorted, but only min/max are used).
    omega_I : float
        ω_n^I
    omega_M : float
        ω_n^M
    omega_S : float
        ω_n^S
    h_n : float
        h_n in the prefactor m_n(omega) = 2 h_n * ( ... ).

    Returns
    -------
    model : ndarray
        m_n(omega) evaluated on the omega grid.
    """
    w_min = omega.min()
    w_max = omega.max()

    if omega_I <= 0:
        raise ValueError("omega_I must be > 0.")
    denom = w_max + omega_S - omega_M
    if denom == 0:
        raise ValueError("omega_max + omega_S - omega_M must be nonzero.")

    w_I_half = 0.5 * omega_I
    w_mid = 0.5 * (w_max + omega_S + omega_M)

    domains = np.empty_like(omega)

    m1 = (omega >= w_min) & (omega < w_I_half)
    domains[m1] = (omega[m1] / omega_I) ** 2

    m2 = (omega >= w_I_half) & (omega < omega_I)
    domains[m2] = 0.5 - (omega[m2] / omega_I - 1.0) ** 2

    m3 = (omega >= omega_I) & (omega < omega_M)
    domains[m3] = 0.5

    m4 = (omega >= omega_M) & (omega < w_mid)
    domains[m4] = 0.5 - ((omega[m4] - omega_M) / denom) ** 2

    m5 = (omega >= w_mid) & (omega <= w_max)
    domains[m5] = ((omega[m5] - omega_M) / denom - 1.0) ** 2

    return 2.0 * h_n * domains


def chi2kink_logistic(x, a, b, c, d):
    """Four-parameter logistic (scaled sigmoid), evaluated stably.

    Parameters
    ----------
    x : array_like
        Input values.
    a : float
        Lower asymptote.
    b : float
        Amplitude (upper - lower).
    c : float
        Midpoint (inflection point).
    d : float
        Slope parameter (steepness).

    Returns
    -------
    phi : ndarray
        Logistic curve evaluated at x.
    """
    z = d * (x - c)

    phi = np.empty_like(z, dtype=float)

    mpos = z >= 0
    if np.any(mpos):
        phi[mpos] = a + b / (1.0 + np.exp(-z[mpos]))

    mneg = ~mpos
    if np.any(mneg):
        expz = np.exp(z[mneg])
        phi[mneg] = a + b * expz / (1.0 + expz)

    return phi