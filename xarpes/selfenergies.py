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

    def __init__(self, energy_range, hnuminPhi, energy_resolution,
                 temperature, label, properties, parameters):
        # core read-only state
        self._energy_range = (None if energy_range is None
                              else np.asarray(energy_range))
        self._hnuminPhi = hnuminPhi

        if self._hnuminPhi is None:
            # MDC export may be electron-energy-leading when hnuminPhi is unknown.
            self._enel_range = self._energy_range
            self._ekin_range = None
        else:
            self._ekin_range = self._energy_range
            self._enel_range = (None if self._ekin_range is None
                                else np.asarray(self._ekin_range) - self._hnuminPhi)
        self._energy_resolution = energy_resolution
        self._temperature = temperature
        self._label = label

        # accept either a dict or a single-element list of dicts
        if isinstance(properties, list):
            if len(properties) == 1:
                properties = properties[0]
            else:
                raise ValueError(
                    "`properties` must be a dict or a single dict in a list."
                )

        # single source of truth for all params (+ their *_sigma)
        self._properties = dict(properties or {})
        self._class = self._properties.get("_class", None)

        # ---- enforce supported classes at construction
        if self._class not in ("SpectralLinear", "SpectralQuadratic",
                               "MomentumQuadratic"):
            raise ValueError(
                f"Unsupported spectral class '{self._class}'. "
                "Only 'SpectralLinear', 'SpectralQuadratic', or "
                "'MomentumQuadratic' are allowed."
            )

        # grab user parameters
        self._parameters = dict(parameters or {})
        self._fermi_wavevector = self._parameters.get("fermi_wavevector")
        self._fermi_velocity   = self._parameters.get("fermi_velocity")
        self._bare_mass        = self._parameters.get("bare_mass")
        self._side             = self._parameters.get("side", None)
        self._abscissa_type    = self._parameters.get("abscissa_type", None)

        # ---- class-specific parameter constraints
        if self._class == "SpectralLinear" and (self._bare_mass is not None):
            raise ValueError("`bare_mass` cannot be set for SpectralLinear.")
        if self._class == "SpectralQuadratic" and (self._fermi_velocity is not None):
            raise ValueError(
                "`fermi_velocity` cannot be set for SpectralQuadratic."
                )

        if self._side is not None and self._side not in ("left", "right"):
            raise ValueError("`side` must be 'left' or 'right' if provided.")
        if self._side is not None:
            self._parameters["side"] = self._side

        if self._abscissa_type is not None and self._abscissa_type not in ("angle", "momenta"):
            raise ValueError("`abscissa_type` must be 'angle' or 'momenta' if provided.")
        if self._abscissa_type is not None:
            self._parameters["abscissa_type"] = self._abscissa_type

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

        # lazy caches for α²F(ω) extraction results
        self._a2f_spectrum = None
        self._a2f_model = None
        self._a2f_omega_range = None
        self._a2f_aval_select = None
        self._a2f_cost = None

    def _check_mass_velocity_exclusivity(self):
        """Ensure that fermi_velocity and bare_mass are not both set."""
        if (self._fermi_velocity is not None) and (self._bare_mass is not None):
            raise ValueError(
                "Cannot set both `fermi_velocity` and  `bare_mass`: choose one "
                "physical parametrization (SpectralLinear or SpectralQuadratic)."
            )

    # ---------------- core read-only axes ----------------
    @property
    def ekin_range(self):
        return self._ekin_range

    @property
    def enel_range(self):
        return self._enel_range

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
    def abscissa_type(self):
        """Abscissa type used during MDC fitting/export: 'angle' or 'momenta'."""
        return self._abscissa_type

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
            raise ValueError(
                "`fermi_velocity` cannot be set for SpectralQuadratic."
                )
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
            raise ValueError(
                "`bare_mass` cannot be set for SpectralLinear."
                )
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
            if self._peak is None:
                return None
            if self._class != "MomentumQuadratic" and self._ekin_range is None:
                return None
            if self._class == "SpectralQuadratic":
                if self._side is None:
                    raise AttributeError(
                        "For SpectralQuadratic, set `side` ('left'/'right') "
                        "before accessing peak_positions and quantities that "
                        "depend on the latter."
                    )
                kpar_mag = (
                    np.sqrt(self._ekin_range / PREF)
                    * np.sin(np.deg2rad(np.abs(self._peak)))
                )
                self._peak_positions = ((-1.0 if self._side == "left"
                                        else 1.0) * kpar_mag)
            elif self._class == "MomentumQuadratic":
                self._peak_positions = np.asarray(self._peak)
            else:
                self._peak_positions = (
                    np.sqrt(self._ekin_range / PREF)
                    * np.sin(np.deg2rad(self._peak))
                )
        return self._peak_positions
    

    @property
    def peak_positions_sigma(self):
        r"""Std. dev. of k_parallel (lazy)."""
        if self._peak_positions_sigma is None:
            if self._peak_sigma is None:
                return None
            if self._class == "MomentumQuadratic":
                self._peak_positions_sigma = np.asarray(self._peak_sigma)
            else:
                if self._ekin_range is None:
                    return None
                self._peak_positions_sigma = (
                    np.sqrt(self._ekin_range / PREF)
                    * np.abs(np.cos(np.deg2rad(self._peak)))
                    * np.deg2rad(self._peak_sigma)
                )
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
    
    @property
    def a2f_spectrum(self):
        """Cached α²F(ω) spectrum from last extraction (or None)."""
        return self._a2f_spectrum

    @property
    def a2f_model(self):
        """Cached MEM model spectrum from last extraction (or None)."""
        return self._a2f_model

    @property
    def a2f_omega_range(self):
        """Cached ω grid for the last extraction (or None)."""
        return self._a2f_omega_range

    @property
    def a2f_aval_select(self):
        """Cached selected a-value from last extraction (or None)."""
        return self._a2f_aval_select

    @property
    def a2f_cost(self):
        """Cached cost from last bayesian_loop (or None)."""
        return self._a2f_cost


    def _compute_imag(self, fermi_velocity=None, bare_mass=None):
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

        if self._class == "SpectralQuadratic":
            mb = self._bare_mass if bare_mass is None else bare_mass
            if mb is None:
                raise AttributeError(
                    "Cannot compute `imag` (SpectralQuadratic): set `bare_mass` "
                    "first."
                )
            return (ekin * broad) / np.abs(mb)

        if self._class == "MomentumQuadratic":
            mb = self._bare_mass if bare_mass is None else bare_mass
            if mb is None:
                raise AttributeError(
                    "Cannot compute `imag` (MomentumQuadratic): set `bare_mass` "
                    "first."
                )
            return (PREF * broad) / np.abs(mb)

        raise NotImplementedError(
            f"_compute_imag is not implemented for spectral class '{self._class}'."
        )


    def _compute_imag_sigma(self, fermi_velocity=None, bare_mass=None):
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

        if self._class == "SpectralQuadratic":
            mb = self._bare_mass if bare_mass is None else bare_mass
            if mb is None:
                raise AttributeError(
                    "Cannot compute `imag_sigma` (SpectralQuadratic): set "
                    "`bare_mass` first."
                )
            return (ekin * broad_sigma) / np.abs(mb)

        if self._class == "MomentumQuadratic":
            mb = self._bare_mass if bare_mass is None else bare_mass
            if mb is None:
                raise AttributeError(
                    "Cannot compute `imag_sigma` (MomentumQuadratic): set "
                    "`bare_mass` first."
                )
            return (PREF * broad_sigma) / np.abs(mb)

        raise NotImplementedError(
            f"_compute_imag_sigma is not implemented for spectral class "
            f"'{self._class}'."
        )


    def _compute_real(self, fermi_velocity=None, fermi_wavevector=None,
                    bare_mass=None):
        r"""Compute Σ' without touching caches."""
        if self._peak is None:
            return None

        enel = self.enel_range
        if enel is None:
            return None

        kpar = self.peak_positions
        if self._class != "MomentumQuadratic" and kpar is None:
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

        if self._class in ("SpectralQuadratic", "MomentumQuadratic"):
            mb = self._bare_mass if bare_mass is None else bare_mass
            kF = (self._fermi_wavevector if fermi_wavevector is None
                else fermi_wavevector)
            if mb is None or kF is None:
                raise AttributeError(
                    f"Cannot compute `real` ({self._class}): set `bare_mass` "
                    "and `fermi_wavevector` first."
                )
            return enel - (PREF / mb) * (kpar**2 - kF**2)

        raise NotImplementedError(
            f"_compute_real is not implemented for spectral class '{self._class}'."
        )

    def _compute_real_sigma(self, fermi_velocity=None, fermi_wavevector=None,
                            bare_mass=None):
        r"""Compute std. dev. of Σ' without touching caches."""
        if self._peak_sigma is None:
            return None

        kpar_sigma = self.peak_positions_sigma
        if self._class != "MomentumQuadratic" and kpar_sigma is None:
            return None

        if self._class == "SpectralLinear":
            vF = self._fermi_velocity if fermi_velocity is None else fermi_velocity
            if vF is None:
                raise AttributeError(
                    "Cannot compute `real_sigma` (SpectralLinear): set "
                    "`fermi_velocity` first."
                )
            return np.abs(vF) * kpar_sigma

        if self._class in ("SpectralQuadratic", "MomentumQuadratic"):
            mb = self._bare_mass if bare_mass is None else bare_mass
            kF = (self._fermi_wavevector if fermi_wavevector is None
                else fermi_wavevector)
            if mb is None or kF is None:
                raise AttributeError(
                    f"Cannot compute `real_sigma` ({self._class}): set "
                    "`bare_mass` and `fermi_wavevector` first."
                )

            kpar = self.peak_positions
            if kpar is None:
                return None
            return 2.0 * PREF * np.abs(kpar_sigma) * np.abs(kpar / mb)

        raise ValueError(
            f"Unsupported spectral class '{self._class}' in "
            "_compute_real_sigma."
        )


    def _evaluate_self_energy_arrays(self, fermi_velocity=None, fermi_wavevector=None,
                                    bare_mass=None):
        r"""Evaluate Σ' / -Σ'' and 1σ uncertainties without mutating caches."""
        real = self._compute_real(
            fermi_velocity=fermi_velocity,
            fermi_wavevector=fermi_wavevector,
            bare_mass=bare_mass,
        )
        real_sigma = self._compute_real_sigma(
            fermi_velocity=fermi_velocity,
            fermi_wavevector=fermi_wavevector,
            bare_mass=bare_mass,
        )
        imag = self._compute_imag(fermi_velocity=fermi_velocity, bare_mass=bare_mass)
        imag_sigma = self._compute_imag_sigma(
            fermi_velocity=fermi_velocity, bare_mass=bare_mass
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

        MomentumQuadratic:
            peak_positions + center_wavevector
        """
        if getattr(self, "_mdc_maxima", None) is None:
            if self.peak_positions is None:
                return None

            if self._class == "SpectralLinear":
                self._mdc_maxima = self.peak_positions
            elif self._class in ("SpectralQuadratic", "MomentumQuadratic"):
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
    
    def _a2f_legend_labels(self):
        """Return (a2f_label, model_label) for legend with safe subscripts."""
        se_label = getattr(self, "label", None)

        if se_label is None:
            return r"$\alpha^2F(\omega)$", r"$m(\omega)$"

        safe_label = str(se_label).replace("_", r"\_")
        if safe_label == "":
            return r"$\alpha^2F(\omega)$", r"$m(\omega)$"

        a2f_label = rf"$\alpha^2F_{{\mathrm{{{safe_label}}}}}(\omega)$"
        model_label = rf"$m_{{\mathrm{{{safe_label}}}}}(\omega)$"
        return a2f_label, model_label

    @add_fig_kwargs
    def plot_real(self, ax=None, scale="eV", resolution_range="absent", **kwargs):
        """Plot the real part Σ' of the self-energy as a function of E-μ.

        Parameters
        ----------
        ax : Matplotlib-Axes or None
            Axis to plot on. Created if not provided by the user.
        scale : {"eV", "meV"}
            Units for both axes. If "meV", x and y (and yerr) are multiplied by
            `KILO`.
        resolution_range : {"absent", "applied"}
            If "applied", removes points with E-μ <= energy_resolution (around
            the chemical potential). The energy resolution is taken from
            ``self.energy_resolution`` (in eV) and scaled consistently with `scale`.
        **kwargs :
            Additional keyword arguments passed to ``ax.errorbar``.

        Returns
        -------
        fig : Matplotlib-Figure
            Figure containing the Σ'(E) plot.
        """
        from . import settings_parameters as xprs

        ax, fig, plt = get_ax_fig_plt(ax=ax)

        if scale not in ("eV", "meV"):
            raise ValueError("scale must be either 'eV' or 'meV'.")
        if resolution_range not in ("absent", "applied"):
            raise ValueError("resolution_range must be 'absent' or 'applied'.")

        factor = KILO if scale == "meV" else 1.0

        x = self.enel_range
        y = self.real
        y_sigma = self.real_sigma

        if x is not None:
            x = factor * np.asarray(x, dtype=float)
        if y is not None:
            y = factor * np.asarray(y, dtype=float)

        if resolution_range == "applied" and x is not None and y is not None:
            res = self.energy_resolution
            if res is not None:
                keep = np.abs(x) > (factor * float(res))
                x = x[keep]
                y = y[keep]
                if y_sigma is not None:
                    y_sigma = np.asarray(y_sigma, dtype=float)[keep]

        real_label, _ = self._se_legend_labels()
        kwargs.setdefault("label", real_label)

        if y_sigma is not None:
            if np.isnan(y_sigma).any():
                print(
                    "Warning: some Σ'(E) uncertainty values are missing. "
                    "Error bars omitted at those energies."
                )
            kwargs.setdefault("yerr", xprs.sigma_confidence * factor * y_sigma)

        ax.errorbar(x, y, **kwargs)

        x_unit = "meV" if scale == "meV" else "eV"
        ax.set_xlabel(rf"$E-\mu$ ({x_unit})")
        ax.set_ylabel(rf"$\Sigma'(E)$ ({x_unit})")
        ax.legend()

        return fig


    @add_fig_kwargs
    def plot_imag(self, ax=None, scale="eV", resolution_range="absent", **kwargs):
        """Plot the imaginary part -Σ'' of the self-energy vs. E-μ.

        Parameters
        ----------
        ax : Matplotlib-Axes or None
            Axis to plot on. Created if not provided by the user.
        scale : {"eV", "meV"}
            Units for both axes. If "meV", x and y (and yerr) are multiplied by
            `KILO`.
        resolution_range : {"absent", "applied"}
            If "applied", removes points with E-μ <= energy_resolution (around
            the chemical potential). The energy resolution is taken from
            ``self.energy_resolution`` (in eV) and scaled consistently with `scale`.
        **kwargs :
            Additional keyword arguments passed to ``ax.errorbar``.

        Returns
        -------
        fig : Matplotlib-Figure
            Figure containing the -Σ''(E) plot.
        """
        from . import settings_parameters as xprs

        ax, fig, plt = get_ax_fig_plt(ax=ax)

        if scale not in ("eV", "meV"):
            raise ValueError("scale must be either 'eV' or 'meV'.")
        if resolution_range not in ("absent", "applied"):
            raise ValueError("resolution_range must be 'absent' or 'applied'.")

        factor = KILO if scale == "meV" else 1.0

        x = self.enel_range
        y = self.imag
        y_sigma = self.imag_sigma

        if x is not None:
            x = factor * np.asarray(x, dtype=float)
        if y is not None:
            y = factor * np.asarray(y, dtype=float)

        if resolution_range == "applied" and x is not None and y is not None:
            res = self.energy_resolution
            if res is not None:
                keep = np.abs(x) > (factor * float(res))
                x = x[keep]
                y = y[keep]
                if y_sigma is not None:
                    y_sigma = np.asarray(y_sigma, dtype=float)[keep]

        _, imag_label = self._se_legend_labels()
        kwargs.setdefault("label", imag_label)

        if y_sigma is not None:
            if np.isnan(y_sigma).any():
                print(
                    "Warning: some -Σ''(E) uncertainty values are missing. "
                    "Error bars omitted at those energies."
                )
            kwargs.setdefault("yerr", xprs.sigma_confidence * factor * y_sigma)

        ax.errorbar(x, y, **kwargs)

        x_unit = "meV" if scale == "meV" else "eV"
        ax.set_xlabel(rf"$E-\mu$ ({x_unit})")
        ax.set_ylabel(rf"$-\Sigma''(E)$ ({x_unit})")
        ax.legend()

        return fig


    @add_fig_kwargs
    def plot_both(self, ax=None, scale="eV", resolution_range="absent", **kwargs):
        """Plot Σ'(E) and -Σ''(E) vs. E-μ on the same axis.

        Parameters
        ----------
        ax : Matplotlib-Axes or None
            Axis to plot on. Created if not provided by the user.
        scale : {"eV", "meV"}
            Units for both axes. If "meV", x, y, and yerr are multiplied by
            `KILO`.
        resolution_range : {"absent", "applied"}
            If "applied", removes points with \-μ <= energy_resolution (around
            the chemical potential). The energy resolution is taken from
            ``self.energy_resolution`` (in eV) and scaled consistently with `scale`.
        **kwargs :
            Additional keyword arguments passed to ``ax.errorbar``.
        """
        from . import settings_parameters as xprs

        ax, fig, plt = get_ax_fig_plt(ax=ax)

        if scale not in ("eV", "meV"):
            raise ValueError("scale must be either 'eV' or 'meV'.")
        if resolution_range not in ("absent", "applied"):
            raise ValueError("resolution_range must be 'absent' or 'applied'.")

        factor = KILO if scale == "meV" else 1.0

        x = self.enel_range
        real = self.real
        imag = self.imag
        real_sigma = self.real_sigma
        imag_sigma = self.imag_sigma

        if x is not None:
            x = factor * np.asarray(x, dtype=float)
        if real is not None:
            real = factor * np.asarray(real, dtype=float)
        if imag is not None:
            imag = factor * np.asarray(imag, dtype=float)

        if resolution_range == "applied" and x is not None:
            res = self.energy_resolution
            if res is not None:
                keep = np.abs(x) > (factor * float(res))
                x = x[keep]
                if real is not None:
                    real = real[keep]
                if imag is not None:
                    imag = imag[keep]
                if real_sigma is not None:
                    real_sigma = np.asarray(real_sigma, dtype=float)[keep]
                if imag_sigma is not None:
                    imag_sigma = np.asarray(imag_sigma, dtype=float)[keep]

        real_label, imag_label = self._se_legend_labels()

        kw_real = dict(kwargs)
        if real_sigma is not None:
            if np.isnan(real_sigma).any():
                print(
                    "Warning: some Σ'(E) uncertainty values are missing. "
                    "Error bars omitted at those energies."
                )
            kw_real.setdefault("yerr", xprs.sigma_confidence * factor * real_sigma)
        kw_real.setdefault("label", real_label)
        ax.errorbar(x, real, **kw_real)

        kw_imag = dict(kwargs)
        if imag_sigma is not None:
            if np.isnan(imag_sigma).any():
                print(
                    "Warning: some -Σ''(E) uncertainty values are missing. "
                    "Error bars omitted at those energies."
                )
            kw_imag.setdefault("yerr", xprs.sigma_confidence * factor * imag_sigma)
        kw_imag.setdefault("label", imag_label)
        ax.errorbar(x, imag, **kw_imag)

        x_unit = "meV" if scale == "meV" else "eV"
        ax.set_xlabel(rf"$E-\mu$ ({x_unit})")
        ax.set_ylabel(rf"$\Sigma'(E),\ -\Sigma''(E)$ ({x_unit})")
        ax.legend()

        return fig

    @add_fig_kwargs
    def plot_a2f(self, ax=None, abscissa="forward", **kwargs):
        ax, fig, plt = get_ax_fig_plt(ax=ax)

        xlim_in = ax.get_xlim()
        ylim_in = ax.get_ylim()

        if abscissa not in ("forward", "reversed"):
            raise ValueError("abscissa must be either 'forward' or 'reversed'.")

        omega = self.a2f_omega_range
        spectrum = self.a2f_spectrum

        if omega is None or spectrum is None:
            raise AttributeError(
                "No cached α²F(ω) spectrum found. Run `extract_a2f()` or "
                "`bayesian_loop()` first."
            )

        if abscissa == "reversed":
            omega = -omega[::-1]
            spectrum = spectrum[::-1]

        a2f_label, _ = self._a2f_legend_labels()
        kwargs.setdefault("label", a2f_label)
        ax.plot(omega, spectrum, **kwargs)

        ax.set_xlabel(r"$\omega$ (meV)")
        ax.set_ylabel(r"$\alpha^2F(\omega)$ (-)")

        self._apply_spectra_axis_defaults(ax, omega, abscissa, xlim_in, ylim_in)

        ax.legend()
        return fig

    @add_fig_kwargs
    def plot_model(self, ax=None, abscissa="forward", **kwargs):
        ax, fig, plt = get_ax_fig_plt(ax=ax)

        xlim_in = ax.get_xlim()
        ylim_in = ax.get_ylim()

        if abscissa not in ("forward", "reversed"):
            raise ValueError("abscissa must be either 'forward' or 'reversed'.")

        omega = self.a2f_omega_range
        model = self.a2f_model

        if omega is None or model is None:
            raise AttributeError(
                "No cached model spectrum found. Run `extract_a2f()` or "
                "`bayesian_loop()` first."
            )

        if abscissa == "reversed":
            omega = -omega[::-1]
            model = model[::-1]

        _, model_label = self._a2f_legend_labels()
        kwargs.setdefault("label", model_label)
        ax.plot(omega, model, **kwargs)

        ax.set_xlabel(r"$\omega$ (meV)")
        ax.set_ylabel(r"$m(\omega)$ (-)")

        self._apply_spectra_axis_defaults(ax, omega, abscissa, xlim_in, ylim_in)

        ax.legend()
        return fig
    
    @add_fig_kwargs
    def plot_spectra(self, ax=None, abscissa="forward", **kwargs):
        ax, fig, plt = get_ax_fig_plt(ax=ax)

        xlim_in = ax.get_xlim()
        ylim_in = ax.get_ylim()

        if abscissa not in ("forward", "reversed"):
            raise ValueError("abscissa must be either 'forward' or 'reversed'.")

        omega = self.a2f_omega_range
        spectrum = self.a2f_spectrum
        model = self.a2f_model

        if omega is None or spectrum is None or model is None:
            raise AttributeError(
                "No cached spectra found. Run `extract_a2f()` or `bayesian_loop()` "
                "first."
            )

        if abscissa == "reversed":
            omega = -omega[::-1]
            spectrum = spectrum[::-1]
            model = model[::-1]

        kw_a2f = dict(kwargs)
        kw_model = dict(kwargs)
        a2f_label, model_label = self._a2f_legend_labels()
        kw_a2f.setdefault("label", a2f_label)
        kw_model.setdefault("label", model_label)

        ax.plot(omega, spectrum, **kw_a2f)
        ax.plot(omega, model, **kw_model)

        self._apply_spectra_axis_defaults(ax, omega, abscissa, xlim_in, ylim_in)

        ax.set_xlabel(r"$\omega$ (meV)")
        ax.set_ylabel(r"$\alpha^2F_n(\omega),~m_n(\omega)~(-)$")
        ax.legend()

        return fig

    @staticmethod
    def _apply_spectra_axis_defaults(ax, omega, abscissa, xlim_in, ylim_in):
        """Apply default spectra x-range and y-min, without stomping overrides.

        Defaults are applied only if the incoming axis limits were Matplotlib's
        defaults (0, 1), i.e. the caller did not pre-set them.
        """
        if abscissa not in ("forward", "reversed"):
            raise ValueError("abscissa must be either 'forward' or 'reversed'.")

        omega_max = float(np.max(np.abs(omega)))

        # --- X defaults (only if user did not pre-set xlim)
        x0, x1 = xlim_in
        x_is_default = np.isclose(x0, 0.0) and np.isclose(x1, 1.0)
        if x_is_default:
            if abscissa == "forward":
                ax.set_xlim(0.0, omega_max)
            else:
                ax.set_xlim(-omega_max, 0.0)

        # --- Y default: set only the bottom to 0 (only if user did not pre-set)
        y0, y1 = ylim_in
        y_is_default = np.isclose(y0, 0.0) and np.isclose(y1, 1.0)
        if y_is_default:
            ax.set_ylim(bottom=0.0)


    @add_fig_kwargs
    def extract_a2f(self, *, omega_min, omega_max, omega_num, omega_I, omega_M,
                    mem=None, ax=None, **mem_kwargs):
        r"""
        Extract Eliashberg function α²F(ω) from the self-energy. While working
        with band maps and MDCs is more intuitive in eV, the self-energy
        extraction is performed in eV.

        Parameters
        ----------
        ax : Matplotlib-Axes or None
            Axis to plot on. Created if not provided by the user. (Not used yet;
            reserved for future plotting.)

        Returns
        -------
        spectrum : ndarray
            Extracted α²F(ω).
        model : ndarray
            MEM model spectrum.
        omega_range : ndarray
            ω grid used for the extraction.
        aval_select : float
            Selected a-value returned by the chi2kink procedure.
        """
        from . import settings_parameters as xprs

        # Reserve the plot API now; not used yet, but this matches xARPES style.
        ax, fig, plt = get_ax_fig_plt(ax=ax)

        mem_cfg = self._merge_defaults(xprs.mem_defaults, mem, mem_kwargs)

        method = mem_cfg["method"]
        parts = mem_cfg["parts"]
        iter_max = int(mem_cfg["iter_max"])
        aval_min = float(mem_cfg["aval_min"])
        aval_max = float(mem_cfg["aval_max"])
        aval_num = int(mem_cfg["aval_num"])
        ecut_left = float(mem_cfg["ecut_left"])
        ecut_right = mem_cfg["ecut_right"]
        omega_S = float(mem_cfg["omega_S"])
        f_chi_squared = mem_cfg["f_chi_squared"]
        sigma_svd = float(mem_cfg["sigma_svd"])
        t_criterion = float(mem_cfg["t_criterion"])
        mu = float(mem_cfg["mu"])
        g_guess = float(mem_cfg["g_guess"])
        b_guess = float(mem_cfg["b_guess"])
        c_guess = float(mem_cfg["c_guess"])
        d_guess = float(mem_cfg["d_guess"])
        power = int(mem_cfg["power"])
        lambda_el = float(mem_cfg["lambda_el"])
        impurity_magnitude = float(mem_cfg["impurity_magnitude"])
        W = mem_cfg.get("W", None)

        if omega_S < 0.0:
            raise ValueError("omega_S must be >= 0.")
        if f_chi_squared is None:
            f_chi_squared = 2.5 if parts == "both" else 2.0
        else:
            f_chi_squared = float(f_chi_squared)
        if d_guess <= 0.0:
            raise ValueError(
                "chi2kink requires d_guess > 0 to fix the logistic sign "
                "ambiguity."
            )

        h_n = mem_cfg.get("h_n", None)
        if h_n is None:
            raise ValueError(
                "`optimisation_parameters` must include 'h_n' for cost evaluation."
            )

        from . import (create_model_function, create_kernel_function,
                       singular_value_decomposition, MEM_core)

        omega_range = np.linspace(omega_min, omega_max, omega_num)
        model = create_model_function(omega_range, omega_I, omega_M, omega_S, h_n)

        delta_omega = (omega_max - omega_min) / (omega_num - 1)
        model_in = model * delta_omega

        energies_eV = self.enel_range

        ecut_left_eV = ecut_left / KILO
        if ecut_right is None:
            ecut_right_eV = self.energy_resolution
        else:
            ecut_right_eV = float(ecut_right) / KILO

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
                    W = (PREF * self._fermi_wavevector**2 / self._bare_mass) * KILO
                else:
                    raise ValueError(
                        "lambda_el was provided, but W is None. For a linearised "
                        "band (SpectralLinear), you must also provide W in meV: "
                        "the electron–electron interaction  scale."
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
            H = np.concatenate((np.real(kernel), -np.imag(kernel)))
        elif parts == "real":
            real = self.real[mE] * KILO - real_el
            real_sigma = self.real_sigma[mE] * KILO
            dvec = real
            wvec = real_sigma**(-2)
            H = np.real(kernel)
        else:  # parts == "imag"
            imag = self.imag[mE] * KILO - impurity_magnitude - imag_el
            imag_sigma = self.imag_sigma[mE] * KILO
            dvec = imag
            wvec = imag_sigma**(-2)
            H = -np.imag(kernel)

        V_Sigma, U, uvec = singular_value_decomposition(H, sigma_svd)

        if method == "chi2kink":
            (spectrum_in, aval_select, fit_curve, guess_curve,
            chi2kink_result) = self._chi2kink_a2f(
                dvec, model_in, uvec, mu, wvec, V_Sigma, U, aval_min,
                aval_max, aval_num, g_guess, b_guess, c_guess, d_guess,
                f_chi_squared, t_criterion, iter_max, MEM_core
            )
        else:
            raise NotImplementedError(
                f"extract_a2f does not support method='{method}'."
            )

        # --- Plot on ax: always raw chi2 + guess; add fit only if success ---
        aval_range = chi2kink_result["aval_range"]
        aval0 = float(aval_range[0])
        x_plot = np.log10(aval_range / aval0)
        y_chi2 = chi2kink_result["log_chi_squared"]

        ax.set_xlabel(r"log$_{10}(a)$ (-)")
        ax.set_ylabel(r"log$_{10}(\chi^2)$ (-)")

        ax.plot(x_plot, y_chi2, label="data")
        ax.plot(x_plot, guess_curve, label="guess")

        if chi2kink_result["success"]:
            ax.plot(x_plot, fit_curve, label="fit")
            ax.axvline(
                np.log10(aval_select / aval0),
                linestyle="--",
                label=r"$a_{\rm sel}$",
            )
        ax.legend()

        # Abort extraction if fit failed (you asked to terminate in that case)
        if not chi2kink_result["success"]:
            raise RuntimeError(
                "chi2kink logistic fit failed; aborting extract_a2f after "
                "plotting chi2 and guess."
            )

        # From here on, we know spectrum_in and aval_select exist
        spectrum = spectrum_in * omega_num / omega_max

        self._a2f_spectrum = spectrum
        self._a2f_model = model
        self._a2f_omega_range = omega_range
        self._a2f_aval_select = aval_select
        self._a2f_cost = None

        return fig, spectrum, model, omega_range, aval_select
    

    def bayesian_loop(self, *, omega_min, omega_max, omega_num, omega_I, 
                    omega_M, fermi_velocity=None, 
                    fermi_wavevector=None, bare_mass=None, vary=(),
                    opt_method="Nelder-Mead", opt_options=None,
                    mem=None, loop=None, **mem_kwargs):
        r"""
        Bayesian outer loop calling `_cost_function()`.

        If `vary` is non-empty, runs a SciPy optimization over the selected
        parameters in `vary`.

        Supported entries in `vary` depend on `self._class`:

        - Common: "fermi_wavevector", "impurity_magnitude", "lambda_el", "h_n"
        - SpectralLinear: additionally "fermi_velocity"
        - SpectralQuadratic: additionally "bare_mass"

        Notes
        -----
        **Convergence behaviour**

        By default, convergence is controlled by a *custom patience criterion*:
        the optimization terminates when the absolute difference between the
        current cost and the best cost seen so far is smaller than `tole` for
        `converge_iters` consecutive iterations.

        To instead rely on SciPy's native convergence criteria (e.g. Nelder–Mead
        `xatol` / `fatol`), disable the custom criterion by setting
        `converge_iters=0` or `tole=None`. In that case, SciPy termination options
        supplied via `opt_options` are used.

        Parameters
        ----------
        opt_options : dict, optional
            Options passed directly to `scipy.optimize.minimize`. These are only
            used for convergence if the custom criterion is disabled (see Notes).
        """

        fermi_velocity, fermi_wavevector, bare_mass = self._prepare_bare(
            fermi_velocity, fermi_wavevector, bare_mass)
        
        vary = tuple(vary) if vary is not None else ()

        allowed = {"fermi_wavevector", "impurity_magnitude", "lambda_el", "h_n"}

        if self._class == "SpectralLinear":
            allowed.add("fermi_velocity")
        elif self._class == "SpectralQuadratic" or self._class == "MomentumQuadratic":
            allowed.add("bare_mass")
        else:
            raise NotImplementedError(
                f"bayesian_loop does not support spectral class '{self._class}'."
            )
        
        unknown = set(vary).difference(allowed)
        if unknown:
            raise ValueError(
                f"Unsupported entries in vary: {sorted(unknown)}. "
                f"Allowed: {sorted(allowed)}."
            )
                        
        omega_num = int(omega_num)
        if omega_num < 2:
            raise ValueError("omega_num must be an integer >= 2.")

        from . import settings_parameters as xprs

        mem_cfg = self._merge_defaults(xprs.mem_defaults, mem, mem_kwargs)

        parts = mem_cfg["parts"]
        sigma_svd = float(mem_cfg["sigma_svd"])
        ecut_left = float(mem_cfg["ecut_left"])
        ecut_right = mem_cfg["ecut_right"]
        omega_S = float(mem_cfg["omega_S"])
        imp0 = float(mem_cfg["impurity_magnitude"])
        lae0 = float(mem_cfg["lambda_el"])
        h_n0 = float(mem_cfg["h_n"])
        h_n_min = float(mem_cfg.get("h_n_min", 1e-8))

        loop_overrides = {
            key: val for key, val in mem_kwargs.items()
            if (val is not None) and (key in xprs.loop_defaults)
        }
        loop_cfg = self._merge_defaults(xprs.loop_defaults, loop, loop_overrides)

        tole = float(loop_cfg["tole"])
        converge_iters = int(loop_cfg["converge_iters"])
        opt_iter_max = int(loop_cfg["opt_iter_max"])
        scale_vF = float(loop_cfg["scale_vF"])
        scale_mb = float(loop_cfg["scale_mb"])
        scale_imp = float(loop_cfg["scale_imp"])
        scale_kF = float(loop_cfg["scale_kF"])
        scale_lambda_el = float(loop_cfg["scale_lambda_el"])
        scale_hn = float(loop_cfg["scale_hn"])

        rollback_steps = int(loop_cfg.get("rollback_steps"))
        max_retries = int(loop_cfg.get("max_retries"))
        relative_best = float(loop_cfg.get("relative_best"))
        min_steps_for_regression = int(loop_cfg.get("min_steps_for_regression"))

        if rollback_steps < 0:
            raise ValueError("rollback_steps must be >= 0.")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0.")
        if relative_best <= 0.0:
            raise ValueError("relative_best must be > 0.")
        if min_steps_for_regression < 0:
            raise ValueError("min_steps_for_regression must be >= 0.")

        vF0 = float(fermi_velocity) if fermi_velocity is not None else None
        kF0 = float(fermi_wavevector) if fermi_wavevector is not None else None
        mb0 = float(bare_mass) if bare_mass is not None else None

        if lae0 < 0.0:
            raise ValueError("Initial lambda_el must be >= 0.")
        if imp0 < 0.0:
            raise ValueError("Initial impurity_magnitude must be >= 0.")
        if omega_S < 0.0:
            raise ValueError("omega_S must be >= 0.")
        if h_n_min <= 0.0:
            raise ValueError("h_n_min must be > 0.")
        if h_n0 < h_n_min:
            raise ValueError(
                f"Initial h_n ({h_n0:g}) must be >= h_n_min ({h_n_min:g})."
            )
        if kF0 is None:
            raise ValueError(
                "bayesian_loop requires an initial fermi_wavevector."
                )
        if self._class == "SpectralLinear" and vF0 is None:
            raise ValueError(
                "bayesian_loop requires an initial fermi_velocity."
                )
        if self._class == "SpectralQuadratic" and mb0 is None:
            raise ValueError("bayesian_loop requires an initial bare_mass.")
        
        from scipy.optimize import minimize
        from . import create_kernel_function, singular_value_decomposition
          
        ecut_left = float(mem_cfg["ecut_left"])
        ecut_right = mem_cfg["ecut_right"]

        ecut_left_eV = ecut_left / KILO
        if ecut_right is None:
            ecut_right_eV = self.energy_resolution
        else:
            ecut_right_eV = float(ecut_right) / KILO

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
        omega_range = np.linspace(omega_min, omega_max, omega_num)

        kernel_raw = create_kernel_function(energies, omega_range, k_BT)

        if parts == "both":
            kernel_used = np.concatenate((np.real(kernel_raw), -np.imag(kernel_raw)))
        elif parts == "real":
            kernel_used = np.real(kernel_raw)
        else:  # parts == "imag"
            kernel_used = -np.imag(kernel_raw)

        V_Sigma, U, uvec0 = singular_value_decomposition(kernel_used, sigma_svd)

        _precomp = {
            "omega_range": omega_range,
            "mE": mE,
            "energies_eV_masked": energies_eV_masked,
            "V_Sigma": V_Sigma,
            "U": U,
            "uvec0": uvec0,
            "ecut_left": ecut_left,
            "ecut_right": ecut_right,
        }
        
        def _reflect_min(xi, p0, p_min, scale):
            """Map R -> [p_min, +inf) using linear reflection around p_min."""
            return p_min + np.abs((float(p0) - p_min) + scale * float(xi))

        def _unpack_params(x):
            params = {}

            i = 0
            for name in vary:
                xi = float(x[i])

                if name == "fermi_velocity":
                    if vF0 is None:
                        raise ValueError("Cannot vary fermi_velocity: no "
                        "initial vF provided.")
                    params["fermi_velocity"] = vF0 + scale_vF * xi

                elif name == "bare_mass":
                    if mb0 is None:
                        raise ValueError("Cannot vary bare_mass: no initial "
                        "bare_mass provided.")
                    params["bare_mass"] = mb0 + scale_mb * xi

                elif name == "fermi_wavevector":
                    if kF0 is None:
                        raise ValueError(
                            "Cannot vary fermi_wavevector: no initial kF "
                            "provided."
                        )
                    params["fermi_wavevector"] = kF0 + scale_kF * xi
                    
                elif name == "impurity_magnitude":
                    params["impurity_magnitude"] = _reflect_min(xi, imp0, 0.0, scale_imp)

                elif name == "lambda_el":
                    params["lambda_el"] = _reflect_min(xi, lae0, 0.0, scale_lambda_el)

                elif name == "h_n":
                    params["h_n"] = _reflect_min(xi, h_n0, h_n_min, scale_hn)

                i += 1

            params.setdefault("fermi_wavevector", kF0)
            params.setdefault("impurity_magnitude", imp0)
            params.setdefault("lambda_el", lae0)
            params.setdefault("h_n", h_n0)

            if self._class == "SpectralLinear":
                params.setdefault("fermi_velocity", vF0)
            elif self._class == "SpectralQuadratic":
                params.setdefault("bare_mass", mb0)

            return params

        def _evaluate_cost(params):
            optimisation_parameters = {
                "h_n": params["h_n"],
                "impurity_magnitude": params["impurity_magnitude"],
                "lambda_el": params["lambda_el"],
                "fermi_wavevector": params["fermi_wavevector"],
            }

            if self._class == "SpectralLinear":
                optimisation_parameters["fermi_velocity"] = params["fermi_velocity"]
            elif self._class == "SpectralQuadratic" or self._class == "MomentumQuadratic":
                optimisation_parameters["bare_mass"] = params["bare_mass"]
            else:
                raise NotImplementedError(
                    f"_evaluate_cost does not support class '{self._class}'."
                )

            return self._cost_function(
                optimisation_parameters=optimisation_parameters,
                omega_min=omega_min, omega_max=omega_max, omega_num=omega_num,
                omega_I=omega_I, omega_M=omega_M, mem_cfg=mem_cfg, 
                _precomp=_precomp
                )

        last = {"cost": None, "spectrum": None, "model": None, "aval": None}

        iter_counter = {"n": 0}

        class ConvergenceException(RuntimeError):
            """Raised when optimisation has converged successfully."""

        class RegressionException(RuntimeError):
            """Raised when optimizer regresses toward the initial guess."""

        if converge_iters is None:
            converge_iters = 0
        converge_iters = int(converge_iters)

        if tole is not None:
            tole = float(tole)
            if tole < 0.0:
                raise ValueError("tole must be >= 0.")
        if converge_iters < 0:
            raise ValueError("converge_iters must be >= 0.")

        # Track best solution seen across all obj calls (not just last).
        best_global = {
            "x": None,
            "params": None,
            "cost": np.inf,
            "spectrum": None,
            "model": None,
            "aval": None,
        }

        history = []

        # Cache most recent evaluation so the callback can read a cost without
        # forcing an extra objective evaluation.
        last_x = {"x": None}
        last_cost = {"cost": None}
        initial_cost = {"cost": None}

        iter_counter = {"n": 0}

        def _clean_params(params):
            """Convert NumPy scalar values to plain Python scalars."""
            out = {}
            for key, val in params.items():
                if isinstance(val, np.generic):
                    out[key] = float(val)
                else:
                    out[key] = val
            return out

        def obj(x):
            import warnings

            iter_counter["n"] += 1

            params = _unpack_params(x)

            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                try:
                    cost, spectrum, model, aval_select = _evaluate_cost(params)
                except RuntimeWarning as exc:
                    raise ValueError(f"RuntimeWarning during cost eval: {exc}") from exc
            cost_f = float(cost)

            history.append(
                {
                    "x": np.array(x, dtype=float, copy=True),
                    "params": _clean_params(params),
                    "cost": cost_f,
                    "spectrum": spectrum,
                    "model": model,
                    "aval": float(aval_select),
                }
            )

            last["cost"] = cost_f
            last["spectrum"] = spectrum
            last["model"] = model
            last["aval"] = float(aval_select)

            last_x["x"] = np.array(x, dtype=float, copy=True)
            last_cost["cost"] = cost_f

            if initial_cost["cost"] is None:
                initial_cost["cost"] = cost_f

            if cost_f < best_global["cost"]:
                best_global["x"] = np.array(x, dtype=float, copy=True)
                best_global["cost"] = cost_f
                best_global["params"] = _clean_params(params)
                best_global["spectrum"] = spectrum
                best_global["model"] = model
                best_global["aval"] = float(aval_select)

            msg = [f"Iter {iter_counter['n']:4d} | cost = {cost: .4e}"]
            for key in sorted(params):
                msg.append(f"{key}={params[key]:.8g}")
            print(" | ".join(msg))

            return cost_f
        
        class TerminationCallback:
            def __init__(self, tole, converge_iters,
                         min_steps_for_regression):
                self.tole = None if tole is None else float(tole)
                self.converge_iters = int(converge_iters)
                self.min_steps_for_regression = int(
                    min_steps_for_regression
                )
                self.iter_count = 0
                self.call_count = 0

            def __call__(self, xk):
                self.call_count += 1

                if self.tole is None or self.converge_iters <= 0:
                    return

                current = last_cost["cost"]
                if current is None:
                    return

                best_cost = float(best_global["cost"])
                if np.isfinite(best_cost):
                    if abs(current - best_cost) < self.tole:
                        self.iter_count += 1
                    else:
                        self.iter_count = 0

                if self.iter_count >= self.converge_iters:
                    raise ConvergenceException(
                        "Converged: |cost-best| < "
                        f"{self.tole:g} for "
                        f"{self.converge_iters} iterations."
                    )

                if self.call_count < self.min_steps_for_regression:
                    return

                init_cost = initial_cost["cost"]
                if init_cost is None:
                    return

                current = float(current)
                init_cost = float(init_cost)

                if not np.isfinite(best_cost):
                    return

                if (
                    abs(current - init_cost) * relative_best
                    < abs(current - best_cost)
                ):
                    raise RegressionException(
                        "Regression toward initial guess detected."
                    )

        callback = TerminationCallback(
            tole=tole,
            converge_iters=converge_iters,
            min_steps_for_regression=min_steps_for_regression,
        )

        if not vary:
            params = _unpack_params(np.zeros(0, dtype=float))
            cost, spectrum, model, aval_select = _evaluate_cost(params)
            return cost, spectrum, model, aval_select

        x0 = np.zeros(len(vary), dtype=float)

        options = {} if opt_options is None else dict(opt_options)
        options.setdefault("maxiter", int(opt_iter_max))

        use_patience = (tole is not None) and (int(converge_iters) > 0)
        if use_patience:
            options.pop("xatol", None)
            options.pop("fatol", None)

        retry_count = 0
        res = None

        while retry_count <= max_retries:
            best = {
                "x": None,
                "params": None,
                "cost": np.inf,
                "spectrum": None,
                "model": None,
                "aval": None,
            }
            last_x["x"] = None
            last_cost["cost"] = None
            initial_cost["cost"] = None
            iter_counter["n"] = 0
            history.clear()

            callback = TerminationCallback(
                tole=tole,
                converge_iters=converge_iters,
                min_steps_for_regression=min_steps_for_regression,
            )

            try:
                res = minimize(
                    obj,
                    x0,
                    method=opt_method,
                    options=options,
                    callback=callback,
                )
                break

            except ConvergenceException as exc:
                print(str(exc))
                res = None
                break

            except RegressionException as exc:
                print(f"{exc} Rolling back {rollback_steps} steps.")
                retry_count += 1

                if rollback_steps <= 0 or not history:
                    continue

                back = min(int(rollback_steps), len(history))
                x0 = np.array(history[-back]["x"], dtype=float, copy=True)
                continue

            except ValueError as exc:
                print(f"ValueError encountered: {exc}. Rolling back.")
                retry_count += 1

                if rollback_steps <= 0 or not history:
                    continue

                back = min(int(rollback_steps), len(history))
                x0 = np.array(history[-back]["x"], dtype=float, copy=True)
                continue

        if retry_count > max_retries:
            print("Max retries reached. Parameters may not be optimal.")

        if best_global["params"] is None:
            params = _unpack_params(x0)
            cost, spectrum, model, aval_select = _evaluate_cost(params)
        else:
            params = best_global["params"]
            cost = best_global["cost"]
            spectrum = best_global["spectrum"]
            model = best_global["model"]
            aval_select = best_global["aval"]

        args = ", ".join(
            f"{key}={params[key]:.10g}" if isinstance(params[key], float)
            else f"{key}={params[key]}"
            for key in sorted(params)
        )
        print("Optimised parameters:")
        print(args)

        # store inside class methods
        self._a2f_spectrum = spectrum
        self._a2f_model = model
        self._a2f_omega_range = omega_range
        self._a2f_aval_select = aval_select
        self._a2f_cost = cost

        return spectrum, model, omega_range, aval_select, cost, params


    @staticmethod
    def _merge_defaults(defaults, override_dict=None, override_kwargs=None):
        """Merge defaults with dict + kwargs overrides (kwargs win)."""
        cfg = dict(defaults)

        if override_dict is not None:
            cfg.update(dict(override_dict))

        if override_kwargs is not None:
            cfg.update({k: v for k, v in override_kwargs.items() if v is not None})

        return cfg

    def _prepare_bare(self, fermi_velocity, fermi_wavevector, bare_mass):
        """Validate class-compatible band parameters and infer missing defaults.

        Enforces:
            - SpectralLinear: bare_mass must be None; vF and kF must be available.
            - SpectralQuadratic: fermi_velocity must be None; bare_mass and kF must
            be available.

        Returns
        -------
        fermi_velocity : float or None
            Initial vF (Linear) or None (Quadratic).
        fermi_wavevector : float
            Initial kF.
        bare_mass : float or None
            Initial bare mass (Quadratic) or None (Linear).
        """
        if self._class == "SpectralLinear":
            if bare_mass is not None:
                raise ValueError(
                    "SpectralLinear bayesian_loop does not accept " 
                    "`bare_mass`. Provide `fermi_velocity` instead."
                )

            if fermi_velocity is None:
                fermi_velocity = getattr(self, "fermi_velocity", None)
                if fermi_velocity is None:
                    raise ValueError(
                        "SpectralLinear optimisation requires an initial "
                        "fermi_velocity to be provided."
                    )

            if fermi_wavevector is None:
                fermi_wavevector = getattr(self, "fermi_wavevector", None)
                if fermi_wavevector is None:
                    raise ValueError(
                        "SpectralLinear optimisation requires an initial "
                        "fermi_wavevector to be provided."
                    )

            return float(fermi_velocity), float(fermi_wavevector), None

        elif self._class == "SpectralQuadratic" or self._class == "MomentumQuadratic":
            if fermi_velocity is not None:
                raise ValueError(
                    "SpectralQuadratic bayesian_loop does not accept "
                    "`fermi_velocity`. Provide `bare_mass` instead."
                )

            if bare_mass is None:
                bare_mass = getattr(self, "_bare_mass", None)
                if bare_mass is None:
                    raise ValueError(
                        "SpectralQuadratic optimisation requires an initial "
                        "bare_mass to be provided."
                    )

            if fermi_wavevector is None:
                fermi_wavevector = getattr(self, "fermi_wavevector", None)
                if fermi_wavevector is None:
                    raise ValueError(
                        "SpectralQuadratic optimisation requires an initial "
                        "fermi_wavevector to be provided."
                    )

            return None, float(fermi_wavevector), float(bare_mass)

        else:
            raise NotImplementedError(
            f"_prepare_bare is not implemented for spectral class "
            "'{self._class}'.")
        

    def _cost_function(self, *, optimisation_parameters, omega_min, omega_max,
                    omega_num, omega_I, omega_M, mem_cfg, _precomp):
        r"""TBD

        Negative log-posterior cost function for Bayesian optimisation.

        This mirrors `extract_a2f()` but recomputes the self-energy arrays for the
        candidate optimisation parameters instead of using cached `self.real/imag`.

        Parameters
        ----------
        optimisation_parameters : dict
            Must include at least keys: "h_n", "impurity_magnitude", "lambda_el".
            For SpectralLinear, must also include "fermi_velocity" and
            "fermi_wavevector". For SpectralQuadratic, "bare_mass" is optional
            (falls back to `self._bare_mass` if present).

        Returns
        -------
        cost : float
            Negative log-posterior evaluated at the selected a-value.
        spectrum : ndarray
            Rescaled α²F(ω) spectrum (same scaling convention as `extract_a2f()`).
        model : ndarray
            The model spectrum used by MEM (same as `extract_a2f()`).
        aval_select : float
            The selected a-value returned by `_chi2kink_a2f`.
        """

        required = {"h_n", "impurity_magnitude", "lambda_el"}
        missing = required.difference(optimisation_parameters)
        if missing:
            raise ValueError(
                f"Missing optimisation parameters: {sorted(missing)}"
            )
        
        parts = mem_cfg["parts"]
        method = mem_cfg["method"]
        aval_min = float(mem_cfg["aval_min"])
        aval_max = float(mem_cfg["aval_max"])
        aval_num = int(mem_cfg["aval_num"])
        omega_S = float(mem_cfg["omega_S"])

        mu = float(mem_cfg["mu"])
        g_guess = float(mem_cfg["g_guess"])
        b_guess = float(mem_cfg["b_guess"])
        c_guess = float(mem_cfg["c_guess"])
        d_guess = float(mem_cfg["d_guess"])

        f_chi_squared = mem_cfg["f_chi_squared"]
        power = int(mem_cfg["power"])
        W = mem_cfg.get("W", None)
        t_criterion = float(mem_cfg["t_criterion"])
        iter_max = int(mem_cfg["iter_max"])

        if f_chi_squared is None:
            f_chi_squared = 2.5 if parts == "both" else 2.0
        else:
            f_chi_squared = float(f_chi_squared)

        if d_guess <= 0.0:
            raise ValueError(
                "chi2kink requires d_guess > 0 to fix the logistic sign ambiguity."
            )

        if parts not in {"both", "real", "imag"}:
            raise ValueError("parts must be one of {'both', 'real', 'imag'}")

        if method != "chi2kink":
            raise NotImplementedError(
                "Only method='chi2kink' is currently implemented."
            )

        impurity_magnitude = float(optimisation_parameters["impurity_magnitude"])
        lambda_el = float(optimisation_parameters["lambda_el"])
        h_n = float(optimisation_parameters["h_n"])

        fermi_velocity = None
        fermi_wavevector = None
        bare_mass = None

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

        elif self._class == "SpectralQuadratic" or self._class == "MomentumQuadratic":
            if "fermi_wavevector" not in optimisation_parameters:
                raise ValueError(
                    "SpectralQuadratic requires optimisation_parameters to include "
                    "'fermi_wavevector'."
                )
            fermi_wavevector = optimisation_parameters["fermi_wavevector"]

            bare_mass = optimisation_parameters.get("bare_mass", None)
            if bare_mass is None:
                bare_mass = getattr(self, "_bare_mass", None)

        else:
            raise NotImplementedError(
                f"_cost_function does not support class '{self._class}'."
            )

        from . import create_model_function, MEM_core

        if f_chi_squared is None:
            f_chi_squared = 2.5 if parts == "both" else 2.0

        if _precomp is None:
            raise ValueError(
                "_precomp is None in _cost_function. Pass the precomputed" 
                " kernel/SVD bundle from bayesian_loop."
            )

        omega_range = _precomp["omega_range"]
        mE = _precomp["mE"]
        energies_eV_masked = _precomp["energies_eV_masked"]

        V_Sigma = _precomp["V_Sigma"]
        U = _precomp["U"]
        uvec = np.array(_precomp["uvec0"], copy=True)

        if f_chi_squared is None:
            f_chi_squared = 2.5 if parts == "both" else 2.0

        model = create_model_function(omega_range, omega_I, omega_M, omega_S, h_n)

        delta_omega = (omega_max - omega_min) / (omega_num - 1)
        model_in = model * delta_omega

        k_BT = K_B * self.temperature * KILO

        if lambda_el:
            if W is None:
                if self._class == "SpectralQuadratic" or self._class == "MomentumQuadratic":
                    if fermi_wavevector is None or bare_mass is None:
                        raise ValueError(
                            "lambda_el is nonzero, but W is None and cannot be "
                            "inferred. Provide W (meV), or pass both "
                            "`fermi_wavevector` and `bare_mass`."
                        )
                    W = (PREF * fermi_wavevector**2 / bare_mass) * KILO
                else:
                    raise ValueError(
                        "lambda_el was provided, but W is None. For "
                        "SpectralLinear you must provide W in meV."
                    )

            energies_el = energies_eV_masked * KILO
            real_el, imag_el = self._el_el_self_energy(
                energies_el, k_BT, lambda_el, W, power
            )
        else:
            real_el = 0.0
            imag_el = 0.0

        real, real_sigma, imag, imag_sigma = self._evaluate_self_energy_arrays(
            fermi_velocity=fermi_velocity,
            fermi_wavevector=fermi_wavevector,
            bare_mass=bare_mass,
        )
        if real is None or imag is None:
            raise ValueError(
                "Cannot compute self-energy arrays for cost evaluation. "
                "Ensure the required band parameters and peak/broadening " \
                "inputs are set.")

        real_m = real[mE] * KILO - real_el
        imag_m = imag[mE] * KILO - impurity_magnitude - imag_el

        if parts == "both":
            real_sig_m = real_sigma[mE] * KILO
            imag_sig_m = imag_sigma[mE] * KILO
            dvec = np.concatenate((real_m, imag_m))
            wvec = np.concatenate((real_sig_m**(-2), imag_sig_m**(-2)))
        elif parts == "real":
            real_sig_m = real_sigma[mE] * KILO
            dvec = real_m
            wvec = real_sig_m**(-2)
        else:
            imag_sig_m = imag_sigma[mE] * KILO
            dvec = imag_m
            wvec = imag_sig_m**(-2)

        (spectrum_in, aval_select, fit_curve, guess_curve,
            chi2kink_result) = self._chi2kink_a2f(
            dvec, model_in, uvec, mu, wvec, V_Sigma, U, aval_min, aval_max,
            aval_num, g_guess, b_guess, c_guess, d_guess, f_chi_squared,
            t_criterion, iter_max, MEM_core,
        )

        T = V_Sigma @ (U.T @ spectrum_in)
        chi_squared = wvec @ ((T - dvec) ** 2)

        mask = (spectrum_in > 0.0) & (model_in > 0.0)
        if not np.any(mask):
            raise ValueError(
                "Invalid spectrum/model for entropy: no positive entries "
                "after MEM."
            )

        information_entropy = (
            np.sum(spectrum_in[mask] - model_in[mask])
            - np.sum(
                spectrum_in[mask]
                * np.log(spectrum_in[mask] / model_in[mask])
            )
        )

        cost = (0.5 * chi_squared
            - aval_select * information_entropy
            + 0.5 * np.sum(np.log(2.0 * np.pi / wvec))
            - 0.5 * spectrum_in.size * np.log(aval_select))

        spectrum = spectrum_in * omega_num / omega_max

        return (cost, spectrum, model, aval_select)


    @staticmethod
    def _chi2kink_a2f(dvec, model_in, uvec, mu, wvec, V_Sigma, U, aval_min,
                    aval_max, aval_num, g_guess, b_guess, c_guess, d_guess,
                    f_chi_squared, t_criterion, iter_max, MEM_core, *,
                    plot=None):
        r"""
        Compute MEM spectrum using the chi2-kink aval-selection procedure.

        Notes
        -----
        This routine contains extensive logic to detect failure modes of the
        chi2-kink logistic fit, including (but not limited to):

        - non-finite or non-positive chi² values,
        - lack of meaningful parameter updates relative to the initial guess,
        - absence of improvement in the residual sum of squares,
        - numerical instabilities or overflows in the logistic model,
        - invalid or non-finite aval selection.

        Despite these safeguards, it is **not possible to guarantee** that all
        failure modes are detected in a nonlinear least-squares problem.
        Consequently, a reported success should be interpreted as a *necessary*
        but *not sufficient* condition for physical or numerical reliability.

        Callers **must** inspect the returned ``success`` flag (contained in
        ``chi2kink_result``) before using the fitted curve, selected aval, or
        MEM spectrum. When ``success`` is False, the returned quantities are
        limited to those required for diagnostic plotting only.
        """
        from . import fit_least_squares, chi2kink_logistic

        aval_range = np.logspace(aval_min, aval_max, int(aval_num))
        chi_squared = np.empty_like(aval_range, dtype=float)

        for i, aval in enumerate(aval_range):
            spectrum_in, uvec = MEM_core(
                dvec, model_in, uvec, mu, aval, wvec, V_Sigma, U,
                t_criterion, iter_max
            )
            T = V_Sigma @ (U.T @ spectrum_in)
            chi_squared[i] = wvec @ ((T - dvec) ** 2)

        if (not np.all(np.isfinite(chi_squared))) or np.any(chi_squared <= 0.0):
            raise ValueError(
                "chi_squared contains non-finite or non-positive values."
            )

        log_aval = np.log10(aval_range)
        log_chi_squared = np.log10(chi_squared)

        p0 = np.array([g_guess, b_guess, c_guess, d_guess], dtype=float)
        pfit, pcov, lsq_success = fit_least_squares(
            p0, log_aval, log_chi_squared, chi2kink_logistic
        )

        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            guess_curve = chi2kink_logistic(log_aval, *p0)

        # Start from the necessary requirement: least_squares must say success
        success = bool(lsq_success)

        # If the guess itself blows up, we can't trust anything
        if not np.all(np.isfinite(guess_curve)):
            success = False

        fit_curve_tmp = None
        if success:
            pfit = np.asarray(pfit, dtype=float)

            if np.allclose(pfit, p0, rtol=1e-12, atol=0.0):
                success = False
            else:
                with np.errstate(over="ignore", invalid="ignore",
                                 divide="ignore"):
                    fit_curve_tmp = chi2kink_logistic(log_aval, *pfit)

                if not np.all(np.isfinite(fit_curve_tmp)):
                    success = False
                else:
                    r0 = guess_curve - log_chi_squared
                    r1 = fit_curve_tmp - log_chi_squared
                    sse0 = float(r0 @ r0)
                    sse1 = float(r1 @ r1)

                    tol = 1e-12 * max(1.0, sse0)
                    if (not np.isfinite(sse1)) or (sse1 >= sse0 - tol):
                        success = False

        aval_select = None
        fit_curve = None
        spectrum_out = None

        if success:
            fit_curve = fit_curve_tmp

            cout = float(pfit[2])
            dout = float(pfit[3])
            exp10 = cout - float(f_chi_squared) / dout

            if (not np.isfinite(exp10)) or (exp10 < -308.0) or (exp10 > 308.0):
                success = False
                fit_curve = None
            else:
                with np.errstate(over="raise", invalid="raise"):
                    aval_select = float(np.power(10.0, exp10))

                spectrum_out, uvec = MEM_core(
                    dvec, model_in, uvec, mu, aval_select, wvec, V_Sigma, U,
                    t_criterion, iter_max
                )

        chi2kink_result = {
            "aval_range": aval_range,
            "chi_squared": chi_squared,
            "log_aval": log_aval,
            "log_chi_squared": log_chi_squared,
            "p0": p0,
            "pfit": pfit,
            "pcov": pcov,
            "success": bool(success),
            "aval_select": aval_select,
        }

        return spectrum_out, aval_select, fit_curve, guess_curve, chi2kink_result


    @staticmethod
    def _trimmed_stdout(print_lines):
        """Optionally tee stdout+stderr and replace final output with head/tail."""
        from contextlib import contextmanager

        @contextmanager
        def _ctx():
            if print_lines is None:
                yield
                return

            n = int(print_lines)
            import sys
            import io

            class _Tee:
                def __init__(self, out_stream, err_stream, nlines):
                    self.out_stream = out_stream
                    self.err_stream = err_stream
                    self.nlines = int(nlines)
                    self.buf = io.StringIO()

                def write(self, text):
                    # Jupyter sometimes calls write with empty strings
                    if text is None:
                        return
                    if self.nlines > 0:
                        self.out_stream.write(text)
                        self.out_stream.flush()
                    self.buf.write(text)

                def flush(self):
                    self.out_stream.flush()

                def err_write(self, text):
                    if text is None:
                        return
                    if self.nlines > 0:
                        self.err_stream.write(text)
                        self.err_stream.flush()
                    self.buf.write(text)

                def err_flush(self):
                    self.err_stream.flush()

                def cleaned(self):
                    lines = self.buf.getvalue().splitlines()
                    if self.nlines <= 0:
                        return ""

                    if len(lines) <= 2 * self.nlines:
                        return "\n".join(lines)

                    head = lines[:self.nlines]
                    tail = lines[-self.nlines:]
                    omitted = len(lines) - 2 * self.nlines
                    mid = f"... ({omitted} lines omitted) ..."
                    return "\n".join(head + [mid] + tail)

            stdout_orig = sys.stdout
            stderr_orig = sys.stderr

            tee = _Tee(stdout_orig, stderr_orig, n)

            class _ErrProxy:
                def write(self, text):
                    tee.err_write(text)

                def flush(self):
                    tee.err_flush()

            try:
                sys.stdout = tee
                sys.stderr = _ErrProxy()
                yield
            finally:
                sys.stdout = stdout_orig
                sys.stderr = stderr_orig

                try:
                    from IPython.display import clear_output
                    clear_output(wait=True)
                except Exception:
                    pass

                cleaned = tee.cleaned()
                if cleaned:
                    print(cleaned)
                else:
                    # Avoid a "blank cell" surprise.
                    print("(output trimmed; nothing captured from stdout/stderr)")

        return _ctx()


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
            raise ValueError(
                "El-el coupling has not yet been implemented for the given " \
                "power."
                )

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