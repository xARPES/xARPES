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
from igor2 import binarywave
from .plotting import get_ax_fig_plt, add_fig_kwargs
from .functions import fit_least_squares, extend_function
from .distributions import FermiDirac, Linear
from .constants import PREF

class BandMap:
    """
    Band map container for ARPES intensity data.

    A `BandMap` stores a two-dimensional ARPES intensity map together with
    its angular axis and a single energy axis (either kinetic or binding
    energy). Conversion between kinetic and binding energy is handled
    internally via the work-function offset ``hnuminPhi`` when available.

    Notes
    -----
    Users are encouraged to construct instances via
    :meth:`from_ibw_file` or :meth:`from_np_arrays`. The ``__init__``
    method expects fully initialized, canonical NumPy arrays and performs
    no file I/O.

    The intensity array is assumed to have shape
    ``(n_energy, n_angle)``, consistent with all downstream operations
    (plotting, MDC extraction, Fermi-edge fitting).

    See Also
    --------
    BandMap.from_ibw_file
    BandMap.from_np_arrays
    """

    @classmethod
    def from_ibw_file(cls, datafile, transpose=False, flip_ekin=False,
                       flip_angles=False, **kwargs):
        """
        Construct a `BandMap` from an IGOR Binary Wave (``.ibw``) file.

        The IGOR wave header is used to reconstruct the angular and
        kinetic-energy axes. The resulting instance uses kinetic energy
        (`ekin`) as the explicit energy axis.

        Parameters
        ----------
        datafile : path-like
            Path to the ``.ibw`` file.
        transpose : bool, optional
            If True, transpose the loaded intensity array and swap the
            associated axis metadata accordingly.
        flip_ekin : bool, optional
            If True, reverse the kinetic-energy axis (first dimension).
        flip_angles : bool, optional
            If True, reverse the angle axis (second dimension).
        **kwargs
            Additional keyword arguments forwarded to
            :class:`BandMap.__init__`.

        Returns
        -------
        BandMap
            New instance constructed from the file contents.

        Raises
        ------
        ValueError
            If the dimensions reported in the file header do not match
            the shape of the stored intensity array.
        """
        data = binarywave.load(datafile)
        intensities = data['wave']['wData']

        fnum, anum = data['wave']['wave_header']['nDim'][0:2]
        fstp, astp = data['wave']['wave_header']['sfA'][0:2]
        fmin, amin = data['wave']['wave_header']['sfB'][0:2]

        if intensities.shape != (fnum, anum):
            raise ValueError('nDim and shape of wData do not match.')

        if transpose:
            intensities = intensities.T
            fnum, anum = anum, fnum
            fstp, astp = astp, fstp
            fmin, amin = amin, fmin

        if flip_ekin:
            intensities = intensities[::-1, :]

        if flip_angles:
            intensities = intensities[:, ::-1]

        angles = np.linspace(amin, amin + (anum - 1) * astp, anum)
        ekin = np.linspace(fmin, fmin + (fnum - 1) * fstp, fnum)

        return cls(intensities=intensities, angles=angles, ekin=ekin, 
                   **kwargs)

    @classmethod
    def from_np_arrays(cls, intensities=None, angles=None, ekin=None, 
                       enel=None, **kwargs):
        """
        Construct a `BandMap` directly from NumPy arrays.

        Exactly one of `ekin` or `enel` must be provided and will become the
        authoritative energy axis. The other energy axis may be derived
        later if the work-function offset ``hnuminPhi`` is known.

        Parameters
        ----------
        intensities : array-like
            ARPES intensity map with shape ``(n_energy, n_angle)`` [counts].
        angles : array-like
            Angular axis values with shape ``(n_angle,)`` [degree].
        ekin : array-like, optional
            Kinetic-energy axis values with shape ``(n_energy,)`` [eV].
        enel : array-like, optional
            Binding-energy axis values with shape ``(n_energy,)`` [eV].
        **kwargs
            Additional keyword arguments forwarded to
            :class:`BandMap.__init__`.

        Returns
        -------
        BandMap
            New instance constructed from the provided arrays.

        Raises
        ------
        ValueError
            If `intensities` or `angles` is missing, or if both (or neither)
            of `ekin` and `enel` are provided.
        """
        if intensities is None or angles is None:
            raise ValueError('Please provide intensities and angles.')
        if (ekin is None) == (enel is None):
            raise ValueError('Provide exactly one of ekin or enel.')
        return cls(intensities=intensities, angles=angles, ekin=ekin, enel=enel,
                   **kwargs)

    def __init__(self, intensities=None, angles=None, ekin=None, enel=None,
                 energy_resolution=None, angle_resolution=None,
                 temperature=None, hnuminPhi=None, hnuminPhi_std=None):
        """
        Initialize a `BandMap` from canonical arrays and metadata.

        A `BandMap` represents a two-dimensional ARPES intensity map defined on
        an angular axis and a single authoritative energy axis. Exactly one of
        `ekin` (kinetic energy) or `enel` (binding energy) must be provided.
        The non-authoritative energy axis may be derived automatically when the
        work-function offset ``hnuminPhi = h\\nu - \\Phi`` has been set with
        the Fermi-edge fitting.

        Parameters
        ----------
        intensities : array-like
            ARPES intensity map with shape ``(n_energy, n_angle)`` [counts].
        angles : array-like
            Angular axis values with shape ``(n_angle,)`` [degree].
        ekin : array-like, optional
            Kinetic-energy axis values with shape ``(n_energy,)`` [eV]. If 
            provided, `ekin` becomes the authoritative energy axis.
        enel : array-like, optional
            Binding-energy axis values with shape ``(n_energy,)`` [eV]. If 
            provided, `enel` becomes the authoritative energy axis.
        energy_resolution : float, optional
            Energy resolution of the measurement, [eV].
        angle_resolution : float, optional
            Angular resolution of the measurement [degree].
        temperature : float, optional
            Sample temperature [K].
        hnuminPhi : float, optional
            Photon energy minus the work function, ``h\\nu - \\Phi`` [eV]. When
            provided, this value is used to convert between kinetic and binding
            energy via ``enel = ekin - hnuminPhi``.
        hnuminPhi_std : float, optional
            One-sigma standard deviation of `hnuminPhi` [eV].

        Notes
        -----
        Exactly one of `ekin` or `enel` must be provided at initialization.
        Attempting to set both (or neither) raises a `ValueError`.

        The energy axis provided at initialization becomes *authoritative*.
        After initialization:
        - If `ekin` is authoritative, setting `enel` raises an
          `AttributeError`.
        - If `enel` is authoritative, setting `ekin` raises an
          `AttributeError`.
        - Updating `hnuminPhi` updates the non-authoritative energy axis when
          possible.

        No copying or validation of array shapes is performed beyond basic
        presence checks; consistency of dimensions is assumed.

        Raises
        ------
        ValueError
            If required arrays are missing, or if both (or neither) of `ekin`
            and `enel` are provided.
        """

        # --- Required arrays ------------------------------------------------
        if intensities is None:
            raise ValueError('Please provide intensities.')
        if angles is None:
            raise ValueError('Please provide angles.')

        self.intensities = intensities
        self.angles = angles

        # --- Initialize energy axes (raw slots) ----------------------------
        self._ekin = None
        self._enel = None

        # Apply user overrides or file ekin
        if ekin is not None and enel is not None:
            raise ValueError('Provide only one of ekin or enel, not both.')

        if ekin is not None:
            self._ekin = ekin
        elif enel is not None:
            self._enel = enel
        else:
            raise ValueError('Please provide datafile, ekin, or enel.')

        # Scalars / metadata
        self.energy_resolution = energy_resolution
        self.angle_resolution = angle_resolution
        self.temperature = temperature

        # Work-function combo and its std
        self._hnuminPhi = None
        self._hnuminPhi_std = None
        self.hnuminPhi = hnuminPhi
        self.hnuminPhi_std = hnuminPhi_std

        # --- 1) Track which axis is authoritative --------------------------
        self._ekin_explicit = ekin is not None
        self._enel_explicit = enel is not None

        # --- 2) Derive missing axis if possible ----------------------------
        if self._ekin is None and self._enel is not None \
                and self._hnuminPhi is not None:
            self._ekin = self._enel + self._hnuminPhi
        if self._enel is None and self._ekin is not None \
                and self._hnuminPhi is not None:
            self._enel = self._ekin - self._hnuminPhi

    # -------------------- Properties: data arrays ---------------------------
    @property
    def intensities(self):
        return self._intensities

    @intensities.setter
    def intensities(self, x):
        self._intensities = x

    @property
    def angles(self):
        return self._angles

    @angles.setter
    def angles(self, x):
        self._angles = x

    # -------------------- 3) Resolution / temperature ----------------------
    @property
    def angle_resolution(self):
        return self._angle_resolution

    @angle_resolution.setter
    def angle_resolution(self, x):
        self._angle_resolution = x

    @property
    def energy_resolution(self):
        return self._energy_resolution

    @energy_resolution.setter
    def energy_resolution(self, x):
        self._energy_resolution = x

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, x):
        self._temperature = x

    # -------------------- 4) Sync ekin / enel / hnuminPhi ------------------
    @property
    def ekin(self):
        if self._ekin is None and self._enel is not None \
                and self._hnuminPhi is not None:
            return self._enel + self._hnuminPhi
        return self._ekin

    @ekin.setter
    def ekin(self, x):
        if getattr(self, "_enel_explicit", False):
            raise AttributeError('enel is explicit; set hnuminPhi instead.')
        self._ekin = x
        self._ekin_explicit = True
        if not getattr(self, "_enel_explicit", False) \
                and self._hnuminPhi is not None and x is not None:
            self._enel = x - self._hnuminPhi

    @property
    def enel(self):
        if self._enel is None and self._ekin is not None \
                and self._hnuminPhi is not None:
            return self._ekin - self._hnuminPhi
        return self._enel

    @enel.setter
    def enel(self, x):
        if getattr(self, "_ekin_explicit", False):
            raise AttributeError('ekin is explicit; set hnuminPhi instead.')
        self._enel = x
        self._enel_explicit = True
        if not getattr(self, "_ekin_explicit", False) \
                and self._hnuminPhi is not None and x is not None:
            self._ekin = x + self._hnuminPhi
            
    @property
    def hnuminPhi(self):
        r"""Returns the photon energy minus the work function in eV if it has
        been set, either during instantiation, with the setter, or by fitting
        the Fermi-Dirac distribution to the integrated weight.

        Returns
        -------
        hnuminPhi : float, None
            Kinetic energy minus the work function [eV]

        """
        return self._hnuminPhi

    @hnuminPhi.setter
    def hnuminPhi(self, x):
        r"""TBD
        """
        self._hnuminPhi = x
        # Re-derive the non-explicit axis if possible
        if not getattr(self, "_ekin_explicit", False) \
                and self._enel is not None and x is not None:
            self._ekin = self._enel + x
        if not getattr(self, "_enel_explicit", False) \
                and self._ekin is not None and x is not None:
            self._enel = self._ekin - x

    @property
    def hnuminPhi_std(self):
        r"""Returns standard deviation of the photon energy minus the work
        function in eV.

        Returns
        -------
        hnuminPhi_std : float
            Standard deviation of energy minus the work function [eV]
            
        """
        return self._hnuminPhi_std

    @hnuminPhi_std.setter
    def hnuminPhi_std(self, x):
        r"""Manually sets the standard deviation of photon energy minus the
        work function in eV.

        Parameters
        ----------
        hnuminPhi_std : float
            Standard deviation of energy minus the work function [eV]

        """
        self._hnuminPhi_std = x

    def shift_angles(self, shift):
        r"""
        Shifts the angles by the specified amount in degrees. Used to shift
        from the detector angle to the material angle.

        Parameters
        ----------
        shift : float
            Angular shift [degrees]

        """
        self.angles = self.angles + shift
        
    def mdc_set(self, angle_min, angle_max, energy_value=None,
                energy_range=None):
        r"""Return a set of momentum distribution curves (MDCs).

        This method extracts MDCs from the stored ARPES intensity map from a
        specified angular interval and either selecting a single energy slice
        or an energy window.

        Parameters
        ----------
        angle_min : float
            Minimum angle of the integration interval [degrees].
        angle_max : float
            Maximum angle of the integration interval [degrees].
        energy_value : float, optional
            Energy value [same units as ``self.enel``] at which a single MDC
            is extracted. Exactly one of ``energy_value`` or ``energy_range``
            must be provided.
        energy_range : array-like, optional
            Energy interval [same units as ``self.enel``] over which MDCs are
            extracted. Exactly one of ``energy_value`` or ``energy_range``
            must be provided.

        Returns
        -------
        mdcs : ndarray
            Extracted MDC intensities. Shape is ``(n_angles,)`` when a single
            ``energy_value`` is provided, or ``(n_energies, n_angles)`` when
            an ``energy_range`` is provided.
        angle_range : ndarray
            Angular values corresponding to the MDCs [degrees].
        angle_resolution : float
            Angular resolution associated with the MDCs.
        energy_resolution : float
            Energy resolution associated with the MDCs.
        temperature: float
            Temperature associated with the band map [K].
        energy_range : ndarray or float
            Energy value (scalar) or energy array corresponding to the MDCs.
        hnuminPhi : float
            Photon-energy-related offset propagated from the BandMap.

        Raises
        ------
        ValueError
            If neither or both of ``energy_value`` and ``energy_range`` are
            provided.

        """

        if (energy_value is None and energy_range is None) or \
        (energy_value is not None and energy_range is not None):
            raise ValueError('Please provide either energy_value or ' +
            'energy_range.')

        angle_min_index = np.abs(self.angles - angle_min).argmin()
        angle_max_index = np.abs(self.angles - angle_max).argmin()
        angle_range_out = self.angles[angle_min_index:angle_max_index + 1]

        if energy_value is not None:
            energy_index = np.abs(self.enel - energy_value).argmin()
            enel_range_out = self.enel[energy_index]
            mdcs = self.intensities[energy_index,
                   angle_min_index:angle_max_index + 1]

        if energy_range is not None:
            energy_indices = np.where((self.enel >= np.min(energy_range))
                                      & (self.enel <= np.max(energy_range))) \
                                        [0]
            enel_range_out = self.enel[energy_indices]
            mdcs = self.intensities[energy_indices,
                                    angle_min_index:angle_max_index + 1]

        return (mdcs, angle_range_out, self.angle_resolution, 
                self.energy_resolution, self.temperature, enel_range_out, self.hnuminPhi)

    @add_fig_kwargs
    def plot(self, abscissa='momentum', ordinate='electron_energy',
             self_energies=None, ax=None, markersize=None,
             plot_dispersions='none', **kwargs):
        r"""
        Plot the band map. Optionally overlay a collection of self-energies,
        e.g. a CreateSelfEnergies instance or any iterable of self-energy
        objects. Self-energies are *not* stored internally; they are used
        only for this plotting call.

        When self-energies are present and ``abscissa='momentum'``, their
        MDC maxima are overlaid with 95 % confidence intervals.

        The `plot_dispersions` argument controls bare-band plotting:

        - "full"   : use the full momentum range of the map (default)
        - "none"   : do not plot bare dispersions
        - "kink"   : for each self-energy, use the min/max of its own
          momentum range (typically its MDC maxima), with
          `len(self.angles)` points.
        - "domain" : for SpectralQuadratic, use only the left or right
          domain relative to `center_wavevector`, based on the self-energy
          attribute `side` ("left" / "right"); for other cases this behaves
          as "full".
        """
        import warnings
        from . import settings_parameters as xprs

        plot_disp_mode = plot_dispersions
        valid_disp_modes = ('full', 'none', 'kink', 'domain')
        if plot_disp_mode not in valid_disp_modes:
            raise ValueError(
                f"Invalid plot_dispersions '{plot_disp_mode}'. "
                f"Valid options: {valid_disp_modes}."
            )

        valid_abscissa = ('angle', 'momentum')
        valid_ordinate = ('kinetic_energy', 'electron_energy')

        if abscissa not in valid_abscissa:
            raise ValueError(
                f"Invalid abscissa '{abscissa}'. "
                f"Valid options: {valid_abscissa}"
            )
        if ordinate not in valid_ordinate:
            raise ValueError(
                f"Invalid ordinate '{ordinate}'. "
                f"Valid options: {valid_ordinate}"
            )

        if self_energies is not None:

            # MDC maxima are defined in momentum space, not angle space
            if abscissa == 'angle':
                raise ValueError(
                    "MDC maxima cannot be plotted against angles; they are "
                    "defined in momentum space. Use abscissa='momentum' "
                    "when passing self-energies."
                )

        ax, fig, plt = get_ax_fig_plt(ax=ax)

        Angl, Ekin = np.meshgrid(self.angles, self.ekin)

        if abscissa == 'angle':
            ax.set_xlabel('Angle ($\\degree$)')
            if ordinate == 'kinetic_energy':
                mesh = ax.pcolormesh(
                    Angl, Ekin, self.intensities,
                    shading='auto',
                    cmap=plt.get_cmap('bone').reversed())
                ax.set_ylabel('$E_{\\mathrm{kin}}$ (eV)')
            elif ordinate == 'electron_energy':
                Enel = Ekin - self.hnuminPhi
                mesh = ax.pcolormesh(
                    Angl, Enel, self.intensities,
                    shading='auto',
                    cmap=plt.get_cmap('bone').reversed())
                ax.set_ylabel('$E-\\mu$ (eV)')

        elif abscissa == 'momentum':
            ax.set_xlabel(r'$k_{//}$ ($\mathrm{\AA}^{-1}$)')

            with warnings.catch_warnings(record=True) as wlist:
                warnings.filterwarnings(
                    "always",
                    message=(
                        "The input coordinates to pcolormesh are "
                        "interpreted as cell centers, but are not "
                        "monotonically increasing or decreasing."
                    ),
                    category=UserWarning,
                )

                Mome = np.sqrt(Ekin / PREF) * np.sin(np.deg2rad(Angl))
                mome_min = np.min(Mome)
                mome_max = np.max(Mome)
                full_disp_momenta = np.linspace(
                    mome_min, mome_max, len(self.angles)
                )

                if ordinate == 'kinetic_energy':
                    mesh = ax.pcolormesh(
                        Mome, Ekin, self.intensities,
                        shading='auto',
                        cmap=plt.get_cmap('bone').reversed())
                    ax.set_ylabel('$E_{\\mathrm{kin}}$ (eV)')
                elif ordinate == 'electron_energy':
                    Enel = Ekin - self.hnuminPhi
                    mesh = ax.pcolormesh(
                        Mome, Enel, self.intensities,
                        shading='auto',
                        cmap=plt.get_cmap('bone').reversed())
                    ax.set_ylabel('$E-\\mu$ (eV)')

                y_lims = ax.get_ylim()

            if any("cell centers" in str(w.message) for w in wlist):
                warnings.warn(
                    "Conversion from angle to momenta causes warping of the "
                    "cell centers. \n Cell edges of the mesh plot may look "
                    "irregular.",
                    UserWarning,
                    stacklevel=2,
                )

        if abscissa == 'momentum' and self_energies is not None:
            for self_energy in self_energies:

                mdc_maxima = getattr(self_energy, "mdc_maxima", None)

                # If this self-energy doesn't contain maxima, don't plot
                if mdc_maxima is None:
                    continue

                # Reserve a colour from the axes cycle for this self-energy,
                # and use it consistently for MDC maxima and dispersion.
                line_color = ax._get_lines.get_next_color()

                peak_sigma = getattr(
                    self_energy, "peak_positions_sigma", None
                )
                xerr = xprs.sigma_confidence * peak_sigma if peak_sigma is \
                    not None else None

                if ordinate == 'kinetic_energy':
                    y_vals = self_energy.ekin_range
                else:
                    y_vals = self_energy.enel_range

                x_vals = mdc_maxima
                label = getattr(self_energy, "label", None)

                # First plot the MDC maxima, using the reserved colour
                if xerr is not None:
                    ax.errorbar(
                        x_vals, y_vals, xerr=xerr, fmt='o',
                        linestyle='', label=label,
                        markersize=markersize,
                        color=line_color, ecolor=line_color,
                    )
                else:
                    ax.plot(
                        x_vals, y_vals, linestyle='',
                        marker='o', label=label,
                        markersize=markersize,
                        color=line_color,
                    )

                # Bare-band dispersion for SpectralLinear / SpectralQuadratic
                spec_class = getattr(
                    self_energy, "_class",
                    self_energy.__class__.__name__,
                )

                if (plot_disp_mode != 'none'
                        and spec_class in ("SpectralLinear",
                                           "SpectralQuadratic")):

                    # Determine momentum grid for the dispersion
                    if plot_disp_mode == 'kink':
                        x_arr = np.asarray(x_vals)
                        mask = np.isfinite(x_arr)
                        if not np.any(mask):
                            # No valid k-points to define a range
                            continue
                        k_min = np.min(x_arr[mask])
                        k_max = np.max(x_arr[mask])
                        disp_momenta = np.linspace(
                            k_min, k_max, len(self.angles)
                        )
                    elif (plot_disp_mode == 'domain'
                          and spec_class == "SpectralQuadratic"):
                        side = getattr(self_energy, "side", None)
                        if side == 'left':
                            disp_momenta = np.linspace(
                                mome_min, self_energy.center_wavevector,
                                len(self.angles)
                            )
                        elif side == 'right':
                            disp_momenta = np.linspace(
                                self_energy.center_wavevector, mome_max,
                                len(self.angles)
                            )
                        else:
                            # Fallback: no valid side, use full range
                            disp_momenta = full_disp_momenta
                    else:
                        # 'full' or 'domain' for SpectralLinear
                        disp_momenta = full_disp_momenta

                    # --- Robust parameter checks before computing base_disp ---
                    if spec_class == "SpectralLinear":
                        fermi_vel = getattr(
                            self_energy, "fermi_velocity", None
                        )
                        fermi_k = getattr(
                            self_energy, "fermi_wavevector", None
                        )
                        if fermi_vel is None or fermi_k is None:
                            missing = []
                            if fermi_vel is None:
                                missing.append("fermi_velocity")
                            if fermi_k is None:
                                missing.append("fermi_wavevector")
                            raise TypeError(
                                "Cannot plot bare dispersion for "
                                "SpectralLinear: "
                                f"{', '.join(missing)} is None."
                            )

                        base_disp = (
                            fermi_vel * (disp_momenta - fermi_k)
                        )

                    else:  # SpectralQuadratic
                        bare_mass = getattr(
                            self_energy, "bare_mass", None
                        )
                        center_k = getattr(
                            self_energy, "center_wavevector", None
                        )
                        fermi_k = getattr(
                            self_energy, "fermi_wavevector", None
                        )

                        missing = []
                        if bare_mass is None:
                            missing.append("bare_mass")
                        if center_k is None:
                            missing.append("center_wavevector")
                        if fermi_k is None:
                            missing.append("fermi_wavevector")

                        if missing:
                            raise TypeError(
                                "Cannot plot bare dispersion for "
                                "SpectralQuadratic: "
                                f"{', '.join(missing)} is None."
                            )

                        dk = disp_momenta - center_k
                        base_disp = PREF * (dk ** 2 - fermi_k ** 2) / bare_mass
                    # --- end parameter checks and base_disp construction ---

                    if ordinate == 'electron_energy':
                        disp_vals = base_disp
                    else:  # kinetic energy
                        disp_vals = base_disp + self.hnuminPhi

                    band_label = getattr(self_energy, "label", None)
                    if band_label is not None:
                        band_label = f"{band_label} (bare)"

                    ax.plot(
                        disp_momenta, disp_vals,
                        label=band_label,
                        linestyle='--',
                        color=line_color,
                    )

            handles, labels = ax.get_legend_handles_labels()
            if any(labels):
                ax.legend()

            ax.set_ylim(y_lims)

        plt.colorbar(mesh, ax=ax, label='counts (-)')
        return fig
    
    @add_fig_kwargs
    def fit_fermi_edge(self, hnuminPhi_guess, background_guess=0.0,
                       integrated_weight_guess=1.0, angle_min=-np.inf,
                       angle_max=np.inf, ekin_min=-np.inf,
                       ekin_max=np.inf, ax=None, **kwargs):
        r"""Fits the Fermi edge of the band map and plots the result.
        Also sets hnuminPhi, the kinetic energy minus the work function in eV.
        The fitting includes an energy convolution with an abscissa range
        expanded by 5 times the energy resolution standard deviation.

        Parameters
        ----------
        hnuminPhi_guess : float
            Initial guess for kinetic energy minus the work function [eV]
        background_guess : float
            Initial guess for background intensity [counts]
        integrated_weight_guess : float
            Initial guess for integrated spectral intensity [counts]
        angle_min : float
            Minimum angle of integration interval [degrees]
        angle_max : float
            Maximum angle of integration interval [degrees]
        ekin_min : float
            Minimum kinetic energy of integration interval [eV]
        ekin_max : float
            Maximum kinetic energy of integration interval [eV]
        ax : Matplotlib-Axes / NoneType
            Axis for plotting the Fermi edge on. Created if not provided by
            the user.

        Other parameters
        ----------------
        **kwargs : dict, optional
            Additional arguments passed on to add_fig_kwargs.

        Returns
        -------
        fig : Matplotlib-Figure
            Figure containing the Fermi edge fit

        """
        from scipy.ndimage import gaussian_filter

        ax, fig, plt = get_ax_fig_plt(ax=ax)

        min_angle_index = np.argmin(np.abs(self.angles - angle_min))
        max_angle_index = np.argmin(np.abs(self.angles - angle_max))

        min_ekin_index = np.argmin(np.abs(self.ekin - ekin_min))
        max_ekin_index = np.argmin(np.abs(self.ekin - ekin_max))

        energy_range = self.ekin[min_ekin_index:max_ekin_index]

        trapz = np.trapz if hasattr(np, 'trapz') else np.trapezoid

        integrated_intensity = trapz(
            self.intensities[min_ekin_index:max_ekin_index,
                min_angle_index:max_angle_index], axis=1)

        fdir_initial = FermiDirac(temperature=self.temperature,
                                  hnuminPhi=hnuminPhi_guess,
                                  background=background_guess,
                                  integrated_weight=integrated_weight_guess,
                                  name='Initial guess')

        parameters = np.array(
            [hnuminPhi_guess, background_guess, integrated_weight_guess])

        extra_args = (self.temperature,)

        popt, pcov, _ = fit_least_squares(
            p0=parameters, xdata=energy_range, ydata=integrated_intensity,
            function=fdir_initial, resolution=self.energy_resolution,
            yerr=None, bounds=None, extra_args=extra_args)

        # Update hnuminPhi; automatically sets self.enel
        self.hnuminPhi = popt[0]
        self.hnuminPhi_std = np.sqrt(np.diag(pcov)[0])

        fdir_final = FermiDirac(temperature=self.temperature,
                                hnuminPhi=self.hnuminPhi, background=popt[1],
                                integrated_weight=popt[2],
                                name='Fitted result')

        ax.set_xlabel(r'$E_{\mathrm{kin}}$ (-)')
        ax.set_ylabel('Counts (-)')
        ax.set_xlim([ekin_min, ekin_max])

        ax.plot(energy_range, integrated_intensity, label='Data')

        extend, step, numb = extend_function(energy_range,
                                             self.energy_resolution)

        initial_result = gaussian_filter(fdir_initial.evaluate(extend),
                         sigma=step)[numb:-numb if numb else None]

        final_result = gaussian_filter(fdir_final.evaluate(extend),
                       sigma=step)[numb:-numb if numb else None]

        ax.plot(energy_range, initial_result, label=fdir_initial.name)
        ax.plot(energy_range, final_result, label=fdir_final.name)

        ax.legend()

        return fig


    @add_fig_kwargs
    def correct_fermi_edge(self, hnuminPhi_guess=None, background_guess=0.0,
                       integrated_weight_guess=1.0, angle_min=-np.inf,
                       angle_max=np.inf, ekin_min=-np.inf, ekin_max=np.inf,
                       slope_guess=0, offset_guess=None,
                           true_angle=0, ax=None, **kwargs):
        r"""TBD
        hnuminPhi_guess should be estimated at true angle

        Parameters
        ----------
        hnuminPhi_guess : float, optional
            Initial guess for kinetic energy minus the work function [eV].

        Other parameters
        ----------------
        **kwargs : dict, optional
            Additional arguments passed on to add_fig_kwargs.

        Returns
        -------
        fig : Matplotlib-Figure
            Figure containing the Fermi edge fit

        """
        from scipy.ndimage import map_coordinates
        from . import settings_parameters as xprs
        
        if hnuminPhi_guess is None:
            raise ValueError('Please provide an initial guess for ' +
                             'hnuminPhi.')
 
        # Here some loop where it fits all the Fermi edges
        angle_min_index = np.abs(self.angles - angle_min).argmin()
        angle_max_index = np.abs(self.angles - angle_max).argmin()
        
        ekin_min_index = np.abs(self.ekin - ekin_min).argmin()
        ekin_max_index = np.abs(self.ekin - ekin_max).argmin()
  
        Intensities = self.intensities[ekin_min_index:ekin_max_index + 1,
                                       angle_min_index:angle_max_index + 1]
        angle_range = self.angles[angle_min_index:angle_max_index + 1]
        energy_range = self.ekin[ekin_min_index:ekin_max_index + 1]
        
        nmps = np.zeros_like(angle_range, dtype=float)
        stds = np.zeros_like(angle_range, dtype=float)
        
        hnuminPhi_left = hnuminPhi_guess - (true_angle - angle_min) \
        * slope_guess
  
        fdir_initial = FermiDirac(temperature=self.temperature,
                      hnuminPhi=hnuminPhi_left,
                      background=background_guess,
                      integrated_weight=integrated_weight_guess,
                      name='Initial guess')
        
        parameters = np.array(
                [hnuminPhi_left, background_guess, integrated_weight_guess])
        
        extra_args = (self.temperature,)
 
        for indx in range(angle_max_index - angle_min_index + 1):
            edge = Intensities[:, indx]
            
            parameters, pcov, _ = fit_least_squares(
                p0=parameters, xdata=energy_range, ydata=edge,
                function=fdir_initial, resolution=self.energy_resolution,
                yerr=None, bounds=None, extra_args=extra_args)

            nmps[indx] = parameters[0]
            stds[indx] = np.sqrt(np.diag(pcov)[0])
        
        # Offset at true angle if not set before
        if offset_guess is None:    
            offset_guess = hnuminPhi_guess - slope_guess * true_angle 
            
        parameters = np.array([offset_guess, slope_guess])
        
        lin_fun = Linear(offset_guess, slope_guess, 'Linear')
                    
        popt, pcov, _ = fit_least_squares(p0=parameters, xdata=angle_range, 
                        ydata=nmps, function=lin_fun, resolution=None,
                                 yerr=stds, bounds=None)

        linsp = lin_fun(angle_range, popt[0], popt[1])

        # Update hnuminPhi; automatically sets self.enel
        self.hnuminPhi = lin_fun(true_angle, popt[0], popt[1])
        self.hnuminPhi_std = np.sqrt(true_angle**2 * pcov[1, 1] + pcov[0, 0] 
                                     + 2 * true_angle * pcov[0, 1])
                    
        Angl, Ekin = np.meshgrid(self.angles, self.ekin)

        ax, fig, plt = get_ax_fig_plt(ax=ax)
        
        ax.set_xlabel('Angle ($\degree$)')
        ax.set_ylabel('$E_{\mathrm{kin}}$ (eV)')
        mesh = ax.pcolormesh(Angl, Ekin, self.intensities,
                       shading='auto', cmap=plt.get_cmap('bone').reversed(),
                             zorder=1)

        ax.errorbar(angle_range, nmps, yerr=xprs.sigma_confidence * stds, zorder=1)
        ax.plot(angle_range, linsp, zorder=2)
        
        cbar = plt.colorbar(mesh, ax=ax, label='counts (-)')
        
        # Fermi-edge correction
        rows, cols = self.intensities.shape
        shift_values = popt[1] * self.angles / (self.ekin[0] - self.ekin[1])
        row_coords = np.arange(rows).reshape(-1, 1) - shift_values
        col_coords = np.arange(cols).reshape(1, -1).repeat(rows, axis=0)
        self.intensities = map_coordinates(self.intensities, 
                [row_coords, col_coords], order=1)
                                  
        return fig