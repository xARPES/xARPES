# Copyright (C) 2025 xARPES Developers
# This program is free software under the terms of the GNU GPLv3 license.

# get_ax_fig_plt and add_fig_kwargs originate from pymatgen/util/plotting.py.
# Copyright (C) 2011-2024 Shyue Ping Ong and the pymatgen Development Team
# Pymatgen is released under the MIT License.

# See also abipy/tools/plotting.py.
# Copyright (C) 2021 Matteo Giantomassi and the AbiPy Group
# AbiPy is free software under the terms of the GNU GPLv2 license.

"""File containing the band map class."""

import numpy as np
from .plotting import get_ax_fig_plt, add_fig_kwargs
from .constants import PREF, KILO, K_B

class SelfEnergy:
    r"""Self-energy"""

    def __init__(self, ekin_range, hnuminPhi, energy_resolution,
                 temperature, label, properties, parameters):
        # core read-only state
        self._ekin_range = ekin_range
        self._hnuminPhi = hnuminPhi
        self._energy_resolution = energy_resolution
        self._temperature = temperature
        self._label = label

        # accept either a dict or a single-element list of dicts
        if isinstance(properties, list):
            if len(properties) == 1:
                properties = properties[0]
            else:
                raise ValueError("`properties` must be a dict or a single " \
                "dict in a list.")

        # single source of truth for all params (+ their *_sigma)
        self._properties = dict(properties or {})
        self._class = self._properties.get("_class", None)

        # ---- enforce supported classes at construction
        if self._class not in ("SpectralLinear", "SpectralQuadratic"):
            raise ValueError(
                f"Unsupported spectral class '{self._class}'. "
                "Only 'SpectralLinear' or 'SpectralQuadratic' are allowed."
            )

        # grab user parameters
        self._parameters = dict(parameters or {})
        self._fermi_wavevector = self._parameters.get("fermi_wavevector")
        self._fermi_velocity   = self._parameters.get("fermi_velocity")
        self._bare_mass        = self._parameters.get("bare_mass")
        self._side             = self._parameters.get("side", None)

        # ---- class-specific parameter constraints
        if self._class == "SpectralLinear" and (self._bare_mass is not None):
            raise ValueError("`bare_mass` cannot be set for SpectralLinear.")
        if self._class == "SpectralQuadratic" and (self._fermi_velocity is not None):
            raise ValueError("`fermi_velocity` cannot be set for SpectralQuadratic.")

        if self._side is not None and self._side not in ("left", "right"):
            raise ValueError("`side` must be 'left' or 'right' if provided.")
        if self._side is not None:
            self._parameters["side"] = self._side

        # convenience attributes (read from properties)
        self._amplitude        = self._properties.get("amplitude")
        self._amplitude_sigma  = self._properties.get("amplitude_sigma")
        self._peak             = self._properties.get("peak")
        self._peak_sigma       = self._properties.get("peak_sigma")
        self._broadening       = self._properties.get("broadening")
        self._broadening_sigma = self._properties.get("broadening_sigma")
        self._center_wavevector = self._properties.get("center_wavevector")

        # lazy caches
        self._peak_positions = None
        self._peak_positions_sigma = None
        self._real = None
        self._real_sigma = None
        self._imag = None
        self._imag_sigma = None

    def _check_mass_velocity_exclusivity(self):
        """Ensure that fermi_velocity and bare_mass are not both set."""
        if (self._fermi_velocity is not None) and (self._bare_mass is not None):
            raise ValueError(
                "Cannot set both `fermi_velocity` and `bare_mass`: "
                "choose one physical parametrization (SpectralLinear or SpectralQuadratic)."
            )

    # ---------------- core read-only axes ----------------
    @property
    def ekin_range(self):
        return self._ekin_range

    @property
    def enel_range(self):
        if self._ekin_range is None:
            return None
        hnp = 0.0 if self._hnuminPhi is None else self._hnuminPhi
        return np.asarray(self._ekin_range) - hnp

    @property
    def hnuminPhi(self):
        return self._hnuminPhi
    
    @property
    def energy_resolution(self):
        """Energy resolution associated with the self-energy."""
        return self._energy_resolution
    
    @property
    def temperature(self):
        """Temperature associated with the self-energy [K]."""
        return self._temperature

    # ---------------- identifiers ----------------
    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, x):
        self._label = x

    # ---------------- exported user parameters ----------------
    @property
    def parameters(self):
        """Dictionary with user-supplied parameters (read-only view)."""
        return self._parameters

    @property
    def side(self):
        """Optional side selector: 'left' or 'right'."""
        return self._side

    @side.setter
    def side(self, x):
        if x is not None and x not in ("left", "right"):
            raise ValueError("`side` must be 'left' or 'right' if provided.")
        self._side = x
        if x is not None:
            self._parameters["side"] = x
        else:
            self._parameters.pop("side", None)
        # affects sign of peak_positions and thus `real`
        self._peak_positions = None
        self._real = None
        self._real_sigma = None
        self._mdc_maxima = None

    @property
    def fermi_wavevector(self):
        """Optional k_F; can be set later."""
        return self._fermi_wavevector

    @fermi_wavevector.setter
    def fermi_wavevector(self, x):
        self._fermi_wavevector = x
        self._parameters["fermi_wavevector"] = x
        # invalidate dependent cache
        self._real = None
        self._real_sigma = None

    @property
    def fermi_velocity(self):
        """Optional v_F; can be set later."""
        return self._fermi_velocity

    @fermi_velocity.setter
    def fermi_velocity(self, x):
        if self._class == "SpectralQuadratic":
            raise ValueError("`fermi_velocity` cannot be set for" \
            " SpectralQuadratic.")
        self._fermi_velocity = x
        self._parameters["fermi_velocity"] = x
        # invalidate dependents
        self._imag = None; self._imag_sigma = None
        self._real = None; self._real_sigma = None

    @property
    def bare_mass(self):
        """Optional bare mass; used by SpectralQuadratic formulas."""
        return self._bare_mass

    @bare_mass.setter
    def bare_mass(self, x):
        if self._class == "SpectralLinear":
            raise ValueError("`bare_mass` cannot be set for SpectralLinear.")
        self._bare_mass = x
        self._parameters["bare_mass"] = x
        # invalidate dependents
        self._imag = None; self._imag_sigma = None
        self._real = None; self._real_sigma = None

    # ---------------- optional fit parameters (convenience) ----------------
    @property
    def amplitude(self):
        return self._amplitude
    
    @amplitude.setter
    def amplitude(self, x):
        self._amplitude = x
        self._properties["amplitude"] = x

    @property
    def amplitude_sigma(self):
        return self._amplitude_sigma
    
    @amplitude_sigma.setter
    def amplitude_sigma(self, x):
        self._amplitude_sigma = x
        self._properties["amplitude_sigma"] = x

    @property
    def peak(self):
        return self._peak
    
    @peak.setter
    def peak(self, x):
        self._peak = x
        self._properties["peak"] = x
        # invalidate dependent cache
        self._peak_positions = None
        self._real = None
        self._mdc_maxima = None

    @property
    def peak_sigma(self):
        return self._peak_sigma
    
    @peak_sigma.setter
    def peak_sigma(self, x):
        self._peak_sigma = x
        self._properties["peak_sigma"] = x
        self._peak_positions_sigma = None
        self._real_sigma = None

    @property
    def broadening(self):
        return self._broadening
    
    @broadening.setter
    def broadening(self, x):
        self._broadening = x
        self._properties["broadening"] = x
        self._imag = None

    @property
    def broadening_sigma(self):
        return self._broadening_sigma
    
    @broadening_sigma.setter
    def broadening_sigma(self, x):
        self._broadening_sigma = x
        self._properties["broadening_sigma"] = x
        self._imag_sigma = None

    @property
    def center_wavevector(self):
        """Read-only center wavevector (SpectralQuadratic, if present)."""
        return self._center_wavevector
    
    # ---------------- derived outputs ----------------
    @property
    def peak_positions(self):
        r"""k_parallel = peak * dtor * sqrt(ekin_range / PREF) (lazy)."""
        if self._peak_positions is None:
            if self._peak is None or self._ekin_range is None:
                return None
            if self._class == "SpectralQuadratic":
                if self._side is None:
                    raise AttributeError(
                        "For SpectralQuadratic, set `side` ('left'/'right') "
                        "before accessing peak_positions and quantities that "
                        "depend on the latter."
                    )
                kpar_mag = np.sqrt(self._ekin_range / PREF) * \
                    np.sin(np.deg2rad(np.abs(self._peak)))
                self._peak_positions = (-1.0 if self._side == "left" \
                                        else 1.0) * kpar_mag
            else:
                self._peak_positions = np.sqrt(self._ekin_range / PREF) \
                * np.sin(np.deg2rad(self._peak))
        return self._peak_positions
    

    @property
    def peak_positions_sigma(self):
        r"""Std. dev. of k_parallel (lazy)."""
        if self._peak_positions_sigma is None:
            if self._peak_sigma is None or self._ekin_range is None:
                return None
            self._peak_positions_sigma = (np.sqrt(self._ekin_range / PREF) \
                * np.abs(np.cos(np.deg2rad(self._peak))) \
                * np.deg2rad(self._peak_sigma))
        return self._peak_positions_sigma

    @property
    def imag(self):
        r"""-Σ'' (lazy)."""
        if self._imag is None:
            if self._broadening is None or self._ekin_range is None:
                return None
            if self._class == "SpectralLinear":
                if self._fermi_velocity is None:
                    raise AttributeError("Cannot compute `imag` "
                    "(SpectralLinear): set `fermi_velocity` first.")
                self._imag = np.abs(self._fermi_velocity) * np.sqrt(self._ekin_range \
                     / PREF) * self._broadening
            else:
                if self._bare_mass is None:
                    raise AttributeError("Cannot compute `imag` "
                    "(SpectralQuadratic): set `bare_mass` first.")
                self._imag = (self._ekin_range * self._broadening) \
                / np.abs(self._bare_mass)
        return self._imag

    @property
    def imag_sigma(self):
        r"""Std. dev. of -Σ'' (lazy)."""
        if self._imag_sigma is None:
            if self._broadening_sigma is None or self._ekin_range is None:
                return None
            if self._class == "SpectralLinear":
                if self._fermi_velocity is None:
                    raise AttributeError("Cannot compute `imag_sigma` "
                    "(SpectralLinear): set `fermi_velocity` first.")
                self._imag_sigma = np.abs(self._fermi_velocity) * \
                    np.sqrt(self._ekin_range / PREF) * self._broadening_sigma
            else:
                if self._bare_mass is None:
                    raise AttributeError("Cannot compute `imag_sigma` "
                    "(SpectralQuadratic): set `bare_mass` first.")
                self._imag_sigma = (self._ekin_range * \
                            self._broadening_sigma) / np.abs(self._bare_mass)
        return self._imag_sigma

    @property
    def real(self):
        r"""Σ' (lazy)."""
        if self._real is None:
            if self._peak is None or self._ekin_range is None:
                return None
            if self._class == "SpectralLinear":
                if self._fermi_velocity is None or self._fermi_wavevector is None:
                    raise AttributeError("Cannot compute `real` "
                    "(SpectralLinear): set `fermi_velocity` and " \
                    "`fermi_wavevector` first.")
                self._real = self.enel_range - self._fermi_velocity * \
                    (self.peak_positions - self._fermi_wavevector)
            else:
                if self._bare_mass is None or self._fermi_wavevector is None:
                    raise AttributeError("Cannot compute `real` "
                    "(SpectralQuadratic): set `bare_mass` and " \
                    "`fermi_wavevector` first.")
                self._real = self.enel_range - (PREF / \
                    self._bare_mass) * (self.peak_positions**2 \
                    - self._fermi_wavevector**2)
        return self._real

    @property
    def real_sigma(self):
        r"""Std. dev. of Σ' (lazy)."""
        if self._real_sigma is None:
            if self._peak_sigma is None or self._ekin_range is None:
                return None
            if self._class == "SpectralLinear":
                if self._fermi_velocity is None:
                    raise AttributeError("Cannot compute `real_sigma` "
                    "(SpectralLinear): set `fermi_velocity` first.")
                self._real_sigma = np.abs(self._fermi_velocity) * self.peak_positions_sigma
            else: 
                if self._bare_mass is None or self._fermi_wavevector is None:
                    raise AttributeError("Cannot compute `real_sigma` "
                    "(SpectralQuadratic): set `bare_mass` and " \
                    "`fermi_wavevector` first.")
                self._real_sigma = 2 * PREF * self.peak_positions_sigma \
                    * np.abs(self.peak_positions / self._bare_mass)
        return self._real_sigma

    @property
    def mdc_maxima(self):
        """
        MDC maxima (lazy).

        SpectralLinear:
            identical to peak_positions

        SpectralQuadratic:
            peak_positions + center_wavevector
        """
        if getattr(self, "_mdc_maxima", None) is None:
            if self.peak_positions is None:
                return None

            if self._class == "SpectralLinear":
                self._mdc_maxima = self.peak_positions
            elif self._class == "SpectralQuadratic":
                self._mdc_maxima = (
                    self.peak_positions + self._center_wavevector
                )

        return self._mdc_maxima
    
    def _se_legend_labels(self):
        """Return (real_label, imag_label) for legend with safe subscripts."""
        se_label = getattr(self, "label", None)

        if se_label is None:
            real_label = r"$\Sigma'(E)$"
            imag_label = r"$-\Sigma''(E)$"
            return real_label, imag_label

        safe_label = str(se_label).replace("_", r"\_")

        # If the label is empty after conversion, fall back
        if safe_label == "":
            real_label = r"$\Sigma'(E)$"
            imag_label = r"$-\Sigma''(E)$"
            return real_label, imag_label

        real_label = rf"$\Sigma_{{\mathrm{{{safe_label}}}}}'(E)$"
        imag_label = rf"$-\Sigma_{{\mathrm{{{safe_label}}}}}''(E)$"

        return real_label, imag_label

    @add_fig_kwargs
    def plot_real(self, ax=None, **kwargs):
        r"""Plot the real part Σ' of the self-energy as a function of E-μ.

        Parameters
        ----------
        ax : Matplotlib-Axes or None
            Axis to plot on. Created if not provided by the user.
        **kwargs :
            Additional keyword arguments passed to ``ax.errorbar``.

        Returns
        -------
        fig : Matplotlib-Figure
            Figure containing the Σ'(E) plot.
        """
        from . import settings_parameters as xprs

        ax, fig, plt = get_ax_fig_plt(ax=ax)

        x = self.enel_range
        y = self.real
        y_sigma = self.real_sigma

        real_label, _ = self._se_legend_labels()
        kwargs.setdefault("label", real_label)

        if y_sigma is not None:
            if np.isnan(y_sigma).any():
                print(
                    "Warning: some Σ'(E) uncertainty values are missing. "
                    "Error bars omitted at those energies."
                )
            kwargs.setdefault("yerr", xprs.sigma_confidence * y_sigma)

        ax.errorbar(x, y, **kwargs)
        ax.set_xlabel(r"$E-\mu$ (eV)")
        ax.set_ylabel(r"$\Sigma'(E)$ (eV)")
        ax.legend()

        return fig

    @add_fig_kwargs
    def plot_imag(self, ax=None, **kwargs):
        r"""Plot the imaginary part -Σ'' of the self-energy vs. E-μ.

        Parameters
        ----------
        ax : Matplotlib-Axes or None
            Axis to plot on. Created if not provided by the user.
        **kwargs :
            Additional keyword arguments passed to ``ax.errorbar``.

        Returns
        -------
        fig : Matplotlib-Figure
            Figure containing the -Σ''(E) plot.
        """
        from . import settings_parameters as xprs

        ax, fig, plt = get_ax_fig_plt(ax=ax)

        x = self.enel_range
        y = self.imag
        y_sigma = self.imag_sigma

        _, imag_label = self._se_legend_labels()
        kwargs.setdefault("label", imag_label)

        if y_sigma is not None:
            if np.isnan(y_sigma).any():
                print(
                    "Warning: some -Σ''(E) uncertainty values are missing. "
                    "Error bars omitted at those energies."
                )
            kwargs.setdefault("yerr", xprs.sigma_confidence * y_sigma)

        ax.errorbar(x, y, **kwargs)
        ax.set_xlabel(r"$E-\mu$ (eV)")
        ax.set_ylabel(r"$-\Sigma''(E)$ (eV)")
        ax.legend()

        return fig

    @add_fig_kwargs
    def plot_both(self, ax=None, **kwargs):
        r"""Plot Σ'(E) and -Σ''(E) vs. E-μ on the same axis."""
        from . import settings_parameters as xprs

        ax, fig, plt = get_ax_fig_plt(ax=ax)

        x = self.enel_range
        real = self.real
        imag = self.imag
        real_sigma = self.real_sigma
        imag_sigma = self.imag_sigma

        real_label, imag_label = self._se_legend_labels()

        # --- plot Σ'
        kw_real = dict(kwargs)
        if real_sigma is not None:
            if np.isnan(real_sigma).any():
                print(
                    "Warning: some Σ'(E) uncertainty values are missing. "
                    "Error bars omitted at those energies."
                )
            kw_real.setdefault("yerr", xprs.sigma_confidence * real_sigma)
        kw_real.setdefault("label", real_label)
        ax.errorbar(x, real, **kw_real)

        # --- plot -Σ''
        kw_imag = dict(kwargs)
        if imag_sigma is not None:
            if np.isnan(imag_sigma).any():
                print(
                    "Warning: some -Σ''(E) uncertainty values are missing. "
                    "Error bars omitted at those energies."
                )
            kw_imag.setdefault("yerr", xprs.sigma_confidence * imag_sigma)
        kw_imag.setdefault("label", imag_label)
        ax.errorbar(x, imag, **kw_imag)

        ax.set_xlabel(r"$E-\mu$ (eV)")
        ax.set_ylabel(r"$\Sigma'(E),\ -\Sigma''(E)$ (eV)")
        ax.legend()

        return fig


    def extract_a2f(self, omega_min, omega_max, omega_num, omega_I, omega_M,
                    omega_S, h_n, method='chi2kink', parts='both',
                    alpha_min=1.0, alpha_max=9.0, alpha_num=10.0, mu=1.0,
                    a_guess=1.0, b_guess=2.5, c_guess=3.0, d_guess=1.5,
                    f_chi_squared=None, impurity_scattering=0.0,
                    ecut_left=0.0, ecut_right=None, sigma_svd=1e-4,
                    t_criterion=1e-8, iter_max=1e4):
        r"""
        Extract Eliashberg function α²F(ω) from the self-energy. While working
        with band maps and MDCs is more intuitive in eV, the self-energy
        extraction is performed in eV.

        Parameters
        ----------
        ecut_left : float, optional
            Energy cutoff applied on the high–binding-energy side (left-hand
            side of the energy axis), in meV. Binding energies are negative, so
             this value is interpreted relative to the most negative energy in
             ``self.enel_range``. Any data points with energies smaller than
            ``min(self.enel_range) + ecut_left`` are excluded from the
            analysis. Must be non-negative. Defaults to 0.0, meaning no
            exclusion on the high–binding-energy side.

        ecut_right : float or None, optional
            Energy cutoff applied on the low–binding-energy side (right-hand
            side of the energy axis), in meV, measured relative to the
            chemical potential (zero energy). Any data points with energies
            larger than ``-ecut_right`` are excluded. If ``None`` (default),
            the cutoff is set equal to the energy resolution
            ``self.energy_resolution`` (in eV). Negative values are allowed,
            leading to retention of data above the chemical potential.

        """

        if parts not in {"both", "real", "imag"}:
            raise ValueError(
                "parts must be one of {'both', 'real', 'imag'}"
            )

        if ecut_left < 0.0:
            raise ValueError("ecut_left must be >= 0.0")
        ecut_left_eV = ecut_left / KILO
        
        if ecut_right is None:
            ecut_right_eV = self.energy_resolution
        else:
            ecut_right_eV = ecut_right / KILO

        if f_chi_squared is None:
            f_chi_squared = 2.5 if parts == "both" else 2.0

        if method != "chi2kink":
            raise NotImplementedError(
                "Only method='chi2kink' is currently implemented."
            )

        from . import (create_model_function, create_kernel_function,
                    singular_value_decomposition, MEM_core)

        omega_range = np.linspace(omega_min, omega_max, omega_num)

        model = create_model_function(omega_range, omega_I, omega_M, omega_S,
                                       h_n)
        delta_omega = (omega_max - omega_min) / omega_num
        model_in = model * delta_omega

        k_BT = K_B * self.temperature * KILO
        energies_eV = self.enel_range

        Emin = np.min(energies_eV)
        Elow = Emin + ecut_left_eV

        Ehigh = -ecut_right_eV
        mE = (energies_eV >= Elow) & (energies_eV <= Ehigh)

        if not np.any(mE):
            raise ValueError(
                "Energy cutoffs removed all points; adjust ecut_left/right."
            )

        energies = energies_eV[mE] * KILO

        kernel = create_kernel_function(energies, omega_range, k_BT)

        if parts == "both":
            real = self.real[mE] * KILO
            real_sigma = self.real_sigma[mE] * KILO
            imag = self.imag[mE] * KILO - impurity_scattering
            imag_sigma = self.imag_sigma[mE] * KILO
            dvec = np.concatenate((real, imag))
            wvec = np.concatenate((real_sigma**(-2), imag_sigma**(-2)))
            kernel = np.concatenate((np.real(kernel), -np.imag(kernel)))

        elif parts == "real":
            real = self.real[mE] * KILO
            real_sigma = self.real_sigma[mE] * KILO
            dvec = real
            wvec = real_sigma**(-2)
            kernel = np.real(kernel)

        else:  # parts == "imag"
            imag = self.imag[mE] * KILO - impurity_scattering
            imag_sigma = self.imag_sigma[mE] * KILO
            dvec = imag
            wvec = imag_sigma**(-2)
            kernel = -np.imag(kernel)

        V_Sigma, U, uvec = singular_value_decomposition(kernel, sigma_svd)

        if method == "chi2kink":
            spectrum_in = self._chi2kink_a2f(
                dvec, model_in, uvec, mu, wvec, V_Sigma, U, alpha_min,
                alpha_max, alpha_num, a_guess, b_guess, c_guess, d_guess,
                f_chi_squared, t_criterion, iter_max, MEM_core
            )

        spectrum = spectrum_in * omega_num / omega_max
        return spectrum, model
    
    def _chi2kink_a2f(self, dvec, model_in, uvec, mu, wvec, V_Sigma, U,
                            alpha_min, alpha_max, alpha_num, a_guess, b_guess,
                            c_guess, d_guess, f_chi_squared, t_criterion, 
                            iter_max, MEM_core):
        r"""Compute MEM spectrum using the chi2kink alpha-selection procedure.

        Returns
        -------
        spectrum_in : ndarray
            Selected spectrum from MEM_core evaluated at the chi2kink alpha.
        """
        from . import (fit_leastsq, chi2kink_logistic)

        alpha_range = np.logspace(alpha_min, alpha_max, alpha_num)
        chi_squared = np.empty_like(alpha_range, dtype=float)

        for i, alpha in enumerate(alpha_range):
            spectrum_in, uvec = MEM_core(dvec, model_in, uvec, mu, alpha, 
                wvec, V_Sigma, U,  t_criterion, iter_max)
            
            T = V_Sigma @ (U.T @ spectrum_in)
            chi_squared[i] = wvec @ ((T - dvec) ** 2)

        log_alpha = np.log10(alpha_range)
        log_chi_squared = np.log10(chi_squared)

        p0 = np.array([a_guess, b_guess, c_guess, d_guess], dtype=float)
        pfit, pcov = fit_leastsq(
            p0, log_alpha, log_chi_squared, chi2kink_logistic
        )

        cout = pfit[2]
        dout = pfit[3]
        alpha_select = 10 ** (cout - f_chi_squared / dout)

        spectrum_in, uvec = MEM_core(dvec, model_in, uvec, mu, alpha_select, 
                        wvec, V_Sigma, U, t_criterion, iter_max)

        return spectrum_in


class CreateSelfEnergies:
    r"""
    Thin container for self-energies with leaf-aware utilities.
    All items are assumed to be leaf self-energy objects with
    a `.label` attribute for identification.
    """

    def __init__(self, self_energies):
        self.self_energies = self_energies

    # ------ Basic container protocol ------
    def __call__(self):
        return self.self_energies

    @property
    def self_energies(self):
        return self._self_energies

    @self_energies.setter
    def self_energies(self, x):
        self._self_energies = x

    def __iter__(self):
        return iter(self.self_energies)

    def __getitem__(self, index):
        return self.self_energies[index]

    def __setitem__(self, index, value):
        self.self_energies[index] = value

    def __len__(self):
        return len(self.self_energies)

    def __deepcopy__(self, memo):
        import copy
        return type(self)(copy.deepcopy(self.self_energies, memo))

    # ------ Label-based utilities ------
    def get_by_label(self, label):
        r"""
        Return the self-energy object with the given label.

        Parameters
        ----------
        label : str
            Label of the self-energy to retrieve.

        Returns
        -------
        obj : SelfEnergy
            The corresponding self-energy instance.

        Raises
        ------
        KeyError
            If no self-energy with the given label exists.
        """
        for se in self.self_energies:
            if getattr(se, "label", None) == label:
                return se
        raise KeyError(
            f"No self-energy with label {label!r} found in container."
        )

    def labels(self):
        r"""
        Return a list of all labels.
        """
        return [getattr(se, "label", None) for se in self.self_energies]

    def as_dict(self):
        r"""
        Return a {label: self_energy} dictionary for convenient access.
        """
        return {se.label: se for se in self.self_energies}