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
            self._imag = self._compute_imag()
        return self._imag


    @property
    def imag_sigma(self):
        r"""Std. dev. of -Σ'' (lazy)."""
        if self._imag_sigma is None:
            if self._broadening_sigma is None or self._ekin_range is None:
                return None
            self._imag_sigma = self._compute_imag_sigma()
        return self._imag_sigma


    @property
    def real(self):
        r"""Σ' (lazy)."""
        if self._real is None:
            if self._peak is None or self._ekin_range is None:
                return None
            self._real = self._compute_real()
        return self._real


    @property
    def real_sigma(self):
        r"""Std. dev. of Σ' (lazy)."""
        if self._real_sigma is None:
            if self._peak_sigma is None or self._ekin_range is None:
                return None
            self._real_sigma = self._compute_real_sigma()
        return self._real_sigma


    def _compute_imag(self, fermi_velocity=None, mbare=None):
        r"""Compute -Σ'' without touching caches."""
        if self._broadening is None or self._ekin_range is None:
            return None

        ekin = np.asarray(self._ekin_range)
        broad = self._broadening

        if self._class == "SpectralLinear":
            vF = self._fermi_velocity if fermi_velocity is None else fermi_velocity
            if vF is None:
                raise AttributeError(
                    "Cannot compute `imag` (SpectralLinear): set `fermi_velocity` "
                    "first."
                )
            return np.abs(vF) * np.sqrt(ekin / PREF) * broad

        mb = self._bare_mass if mbare is None else mbare
        if mb is None:
            raise AttributeError(
                "Cannot compute `imag` (SpectralQuadratic): set `bare_mass` first."
            )
        return (ekin * broad) / np.abs(mb)


    def _compute_imag_sigma(self, fermi_velocity=None, mbare=None):
        r"""Compute std. dev. of -Σ'' without touching caches."""
        if self._broadening_sigma is None or self._ekin_range is None:
            return None

        ekin = np.asarray(self._ekin_range)
        broad_sigma = self._broadening_sigma

        if self._class == "SpectralLinear":
            vF = self._fermi_velocity if fermi_velocity is None else fermi_velocity
            if vF is None:
                raise AttributeError(
                    "Cannot compute `imag_sigma` (SpectralLinear): set "
                    "`fermi_velocity` first."
                )
            return np.abs(vF) * np.sqrt(ekin / PREF) * broad_sigma

        mb = self._bare_mass if mbare is None else mbare
        if mb is None:
            raise AttributeError(
                "Cannot compute `imag_sigma` (SpectralQuadratic): set `bare_mass` "
                "first."
            )
        return (ekin * broad_sigma) / np.abs(mb)


    def _compute_real(self, fermi_velocity=None, fermi_wavevector=None, mbare=None):
        r"""Compute Σ' without touching caches."""
        if self._peak is None or self._ekin_range is None:
            return None

        enel = self.enel_range
        kpar = self.peak_positions
        if kpar is None:
            return None

        if self._class == "SpectralLinear":
            vF = self._fermi_velocity if fermi_velocity is None else fermi_velocity
            kF = (self._fermi_wavevector if fermi_wavevector is None
                else fermi_wavevector)
            if vF is None or kF is None:
                raise AttributeError(
                    "Cannot compute `real` (SpectralLinear): set `fermi_velocity` "
                    "and `fermi_wavevector` first."
                )
            return enel - vF * (kpar - kF)

        mb = self._bare_mass if mbare is None else mbare
        kF = (self._fermi_wavevector if fermi_wavevector is None
            else fermi_wavevector)
        if mb is None or kF is None:
            raise AttributeError(
                "Cannot compute `real` (SpectralQuadratic): set `bare_mass` and "
                "`fermi_wavevector` first."
            )
        return enel - (PREF / mb) * (kpar**2 - kF**2)


    def _compute_real_sigma(self, fermi_velocity=None, fermi_wavevector=None,
                            mbare=None):
        r"""Compute std. dev. of Σ' without touching caches."""
        if self._peak_sigma is None or self._ekin_range is None:
            return None

        kpar_sigma = self.peak_positions_sigma
        if kpar_sigma is None:
            return None

        if self._class == "SpectralLinear":
            vF = self._fermi_velocity if fermi_velocity is None else fermi_velocity
            if vF is None:
                raise AttributeError(
                    "Cannot compute `real_sigma` (SpectralLinear): set "
                    "`fermi_velocity` first."
                )
            return np.abs(vF) * kpar_sigma

        mb = self._bare_mass if mbare is None else mbare
        kF = (self._fermi_wavevector if fermi_wavevector is None
            else fermi_wavevector)
        if mb is None or kF is None:
            raise AttributeError(
                "Cannot compute `real_sigma` (SpectralQuadratic): set `bare_mass` "
                "and `fermi_wavevector` first."
            )

        kpar = self.peak_positions
        if kpar is None:
            return None
        return 2.0 * PREF * kpar_sigma * np.abs(kpar / mb)
    

    def _evaluate_self_energy_arrays(self, fermi_velocity=None, fermi_wavevector=None,
                                    mbare=None):
        r"""Evaluate Σ' / -Σ'' and 1σ uncertainties without mutating caches."""
        real = self._compute_real(
            fermi_velocity=fermi_velocity, fermi_wavevector=fermi_wavevector,
            mbare=mbare,
        )
        real_sigma = self._compute_real_sigma(
            fermi_velocity=fermi_velocity, fermi_wavevector=fermi_wavevector,
            mbare=mbare,
        )
        imag = self._compute_imag(fermi_velocity=fermi_velocity, mbare=mbare)
        imag_sigma = self._compute_imag_sigma(
            fermi_velocity=fermi_velocity, mbare=mbare
        )
        return real, real_sigma, imag, imag_sigma


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
                    omega_S, method='chi2kink', parts='both',
                    alpha_min=1.0, alpha_max=9.0, alpha_num=10.0, mu=1.0,
                    a_guess=1.0, b_guess=2.5, c_guess=3.0, d_guess=1.5,
                    ecut_left=0.0, ecut_right=None, f_chi_squared=None, 
                    W=None, power=2, h_n=0.1, impurity_magnitude=0.0, 
                    lambda_el=0.0, sigma_svd=1e-4, t_criterion=1e-8, 
                    iter_max=1e4):
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
        
        delta_omega = (omega_max - omega_min) / (omega_num - 1)
        model_in = model * delta_omega

        energies_eV = self.enel_range

        Emin = np.min(energies_eV)
        Elow = Emin + ecut_left_eV
        Ehigh = -ecut_right_eV
        mE = (energies_eV >= Elow) & (energies_eV <= Ehigh)

        if not np.any(mE):
            raise ValueError(
                "Energy cutoffs removed all points; adjust ecut_left/right."
            )

        energies_eV_masked = energies_eV[mE]
        energies = energies_eV_masked * KILO

        k_BT = K_B * self.temperature * KILO

        kernel = create_kernel_function(energies, omega_range, k_BT)

        if lambda_el:
            if W is None:
                if self._class == "SpectralQuadratic":
                    W = (
                        PREF * self._fermi_wavevector**2
                        / (2.0 * self._bare_mass)
                    ) * KILO
                else:
                    raise ValueError(
                        "lambda_el was provided, but W is None. For a "
                        "linearised band (SpectralLinear), you must also "
                        "provide W in meV: the electron–electron interaction scale."
                    )

            energies_el = energies_eV_masked * KILO
            real_el, imag_el = self._el_el_self_energy(
                energies_el, k_BT, lambda_el, W, power
            )
        else:
            real_el = 0.0
            imag_el = 0.0

        if parts == "both":
            real = self.real[mE] * KILO - real_el
            real_sigma = self.real_sigma[mE] * KILO
            imag = self.imag[mE] * KILO - impurity_magnitude - imag_el
            imag_sigma = self.imag_sigma[mE] * KILO
            dvec = np.concatenate((real, imag))
            wvec = np.concatenate((real_sigma**(-2), imag_sigma**(-2)))
            kernel = np.concatenate((np.real(kernel), -np.imag(kernel)))

        elif parts == "real":
            real = self.real[mE] * KILO - real_el
            real_sigma = self.real_sigma[mE] * KILO
            dvec = real
            wvec = real_sigma**(-2)
            kernel = np.real(kernel)

        else:  # parts == "imag"
            imag = self.imag[mE] * KILO - impurity_magnitude - imag_el
            imag_sigma = self.imag_sigma[mE] * KILO
            dvec = imag
            wvec = imag_sigma**(-2)
            kernel = -np.imag(kernel)

        V_Sigma, U, uvec = singular_value_decomposition(kernel, sigma_svd)

        if method == "chi2kink":
            spectrum_in, _ = self._chi2kink_a2f(dvec, model_in, uvec, mu, wvec,
                V_Sigma, U, alpha_min, alpha_max, alpha_num, a_guess, b_guess,
                 c_guess, d_guess, f_chi_squared, t_criterion, iter_max, 
                 MEM_core)

        spectrum = spectrum_in * omega_num / omega_max
        return spectrum, model
    

    def bayesian_loop(self, omega_min, omega_max, omega_num, omega_I, omega_M,
                    omega_S, method='chi2kink', parts='both',
                    alpha_min=1.0, alpha_max=9.0, alpha_num=10.0, mu=1.0,
                    a_guess=1.0, b_guess=2.5, c_guess=3.0, d_guess=1.5,
                    f_chi_squared=None, W=None, power=2,
                    ecut_left=0.0, ecut_right=None, sigma_svd=1e-4,
                    t_criterion=1e-8, iter_max=1e4,
                    h_n=1.0, impurity_magnitude=0.0, lambda_el=0.0,
                    fermi_velocity=None, fermi_wavevector=None):
        r"""TBD

        Stub for a Bayesian outer loop that will call extract_a2f().
        Currently accepts the same arguments as extract_a2f and does nothing.
        """

        if self._class == "SpectralLinear":
            if fermi_velocity is None:
                if hasattr(self, "fermi_velocity"):
                    fermi_velocity = self.fermi_velocity
                else:
                    raise ValueError("SpectralLinear optimisation requires an "
                    "initial fermi_velocity to be provided.")
                
            if fermi_wavevector is None:
                if hasattr(self, "fermi_wavevector"):
                    fermi_wavevector = self.fermi_wavevector
                else:
                    raise ValueError("SpectralLinear optimisation requires an "
                    "initial fermi_wavevector to be provided.")

        optimisation_parameters = {"h_n": h_n,
        "impurity_magnitude": impurity_magnitude,
        "lambda_el": lambda_el,
        "fermi_velocity": fermi_velocity,
        "fermi_wavevector": fermi_wavevector}

        cost = cost_function(optimisation_parameters)

        # T = V_Sigma @ (U.T @ spectrum_in)
        # chi_squared = wvec @ ((T - dvec) ** 2)

        # mask = (spectrum_in > 0.0) & (model_in > 0.0)
        # information_entropy = (np.sum(spectrum_in[mask] - model_in[mask]) 
        #                     - np.sum(spectrum_in[mask] * 
        #                              np.log(spectrum_in[mask] / model_in[mask])))
        
        # cost_function = (
        #     0.5 * chi_squared
        #     - alpha_select * information_entropy
        #     + 0.5 * np.sum(np.log(2.0 * np.pi / wvec))
        #     - 0.5 * spectrum_in.size * np.log(alpha_select)
        # )

        return cost


    def _cost_function(self, optimisation_parameters):
        r"""TBD

        Negative log-posterior cost function for Bayesian optimisation.
        """
        required = {"h_n", "impurity_magnitude", "lambda_el"}
        missing = required.difference(optimisation_parameters)
        if missing:
            raise ValueError(
                f"Missing optimisation parameters: {sorted(missing)}"
            )

        h_n = optimisation_parameters["h_n"]
        impurity_magnitude = optimisation_parameters["impurity_magnitude"]
        lambda_el = optimisation_parameters["lambda_el"]

        if self._class == "SpectralLinear":
            required_lin = {"fermi_velocity", "fermi_wavevector"}
            missing_lin = required_lin.difference(optimisation_parameters)
            if missing_lin:
                raise ValueError(
                    "SpectralLinear requires optimisation_parameters to include "
                    f"{sorted(missing_lin)}."
                )

            fermi_velocity = optimisation_parameters["fermi_velocity"]
            fermi_wavevector = optimisation_parameters["fermi_wavevector"]

        pass

    @staticmethod
    def _chi2kink_a2f(dvec, model_in, uvec, mu, wvec, V_Sigma, U,
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

        return spectrum_in, alpha_select


    @staticmethod
    def _el_el_self_energy(enel_range, k_BT, lambda_el, W, power):
        """Electron–electron contribution to the self-energy."""
        x = enel_range / W
        denom = 1.0 - (np.pi * k_BT / W) ** 2

        if denom == 0.0:
            raise ZeroDivisionError(
                "Invalid parameters: 1 - (π k_BT / W)^2 = 0."
            )

        pref = lambda_el / (W * denom)

        if power == 2:
            real_el = pref * x * ((np.pi * k_BT) ** 2 - W ** 2) / (1.0 + x ** 2)
            imag_el = (pref * (enel_range ** 2 + (np.pi * k_BT) ** 2) 
                       / (1.0 + x ** 2))

        elif power == 4:
            num = (
                (np.pi * k_BT) ** 2 * (1.0 + x ** 2)
                - W ** 2 * (1.0 - x ** 2)
            )
            real_el = pref * x * num / (1.0 + x ** 4)
            imag_el = (pref * np.sqrt(2.0) * (enel_range ** 2 + (np.pi * k_BT)
                        ** 2) / ( 1.0 + x ** 4))
        else:
            raise ValueError("El-el coupling not implemented for given power.")

        return real_el, imag_el


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