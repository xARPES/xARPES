# Copyright (C) 2025 xARPES Developers
# This program is free software under the terms of the GNU GPLv3 license.

# get_ax_fig_plt and add_fig_kwargs originate from pymatgen/util/plotting.py.
# Copyright (C) 2011-2024 Shyue Ping Ong and the pymatgen Development Team
# Pymatgen is released under the MIT License.

# See also abipy/tools/plotting.py.
# Copyright (C) 2021 Matteo Giantomassi and the AbiPy Group
# AbiPy is free software under the terms of the GNU GPLv2 license.

"""File containing all the spectral quantities."""

import numpy as np
from igor2 import binarywave
from .plotting import get_ax_fig_plt, add_fig_kwargs
from .functions import fit_leastsq, extend_function
from .distributions import FermiDirac, Linear
from .constants import KILO, PREF

class BandMap:
    r"""
    Class for the band map from the ARPES experiment.

    Parameters
    ----------
    datafile : str, optional
        Path to an IGOR binary wave file.
    intensities : ndarray, optional
        2D array of intensities [E, angle].
    angles : ndarray, optional
        1D array of emission angles in degrees.
    ekin : ndarray, optional
        1D array of kinetic energies in eV.
    enel : ndarray, optional
        1D array of electron energies in eV.
    energy_resolution : float, optional
        Energy-resolution standard deviation [eV].
    angle_resolution : float, optional
        Angular-resolution standard deviation [deg].
    temperature : float, optional
        Sample temperature [K].
    hnuminPhi : float, optional
        Photon energy minus work function [eV].
    hnuminPhi_std : float, optional
        Standard deviation on ``hnuminPhi`` [eV].
    transpose : bool, optional
        If True, transpose the input data.
    flip_ekin : bool, optional
        If True, flip the energy axis.
    flip_angles : bool, optional
        If True, flip the angle axis.

    Attributes
    ----------
    intensities : ndarray
        2D intensity map [energy, angle].
    angles : ndarray
        Emission angles in degrees.
    ekin : ndarray
        Kinetic-energy axis in eV.
    enel : ndarray
        Electron-energy axis in eV.
    hnuminPhi : float or None
        Photon energy minus work function.
    hnuminPhi_std : float or None
        Standard deviation on ``hnuminPhi``.

    """

    def __init__(self, datafile=None, intensities=None, angles=None,
                 ekin=None, enel=None, energy_resolution=None,
                 angle_resolution=None, temperature=None, hnuminPhi=None,
                 hnuminPhi_std=None, transpose=False, flip_ekin=False,
                 flip_angles=False):

        # --- IO / file load -------------------------------------------------
        if datafile is not None:
            data = binarywave.load(datafile)
            self.intensities = data['wave']['wData']

            fnum, anum = data['wave']['wave_header']['nDim'][0:2]
            fstp, astp = data['wave']['wave_header']['sfA'][0:2]
            fmin, amin = data['wave']['wave_header']['sfB'][0:2]

            if self.intensities.shape != (fnum, anum):
                raise ValueError('nDim and shape of wData do not match.')

            if transpose:
                self.intensities = self.intensities.T
                fnum, anum = anum, fnum
                fstp, astp = astp, fstp
                fmin, amin = amin, fmin

            if flip_ekin:
                self.intensities = self.intensities[::-1, :]

            if flip_angles:
                self.intensities = self.intensities[:, ::-1]

            self.angles = np.linspace(amin, amin + (anum - 1) * astp, anum)
            file_ekin = np.linspace(fmin, fmin + (fnum - 1) * fstp, fnum)
        else:
            file_ekin = None

        # --- Required arrays if not using datafile -------------------------
        if intensities is not None:
            self.intensities = intensities
        elif datafile is None:
            raise ValueError('Please provide datafile or intensities.')

        if angles is not None:
            self.angles = angles
        elif datafile is None:
            raise ValueError('Please provide datafile or angles.')

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
        elif file_ekin is not None:
            self._ekin = file_ekin
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
        self._ekin_explicit = ekin is not None or (file_ekin is not None
                                                   and enel is None)
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
        r"""Returns a set of MDCs. Documentation is to be further completed.
        
        Parameters
        ----------
        angle_min : float
            Minimum angle of integration interval [degrees]
        angle_max : float
            Maximum angle of integration interval [degrees]

        Returns
        -------
        angle_range : ndarray
            Array of size n containing the angular values
        energy_range : ndarray
            Array of size m containing the energy values
        mdcs : ndarray
            Array of size n x m containing the MDC intensities

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

        if energy_range:
            energy_indices = np.where((self.enel >= np.min(energy_range))
                                      & (self.enel <= np.max(energy_range))) \
                                        [0]
            enel_range_out = self.enel[energy_indices]
            mdcs = self.intensities[energy_indices,
                                    angle_min_index:angle_max_index + 1]

        return mdcs, angle_range_out, self.angle_resolution, \
        enel_range_out, self.hnuminPhi

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

        integrated_intensity = np.trapz(
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

        popt, pcov = fit_leastsq(
        parameters, energy_range, integrated_intensity, fdir_initial,
        self.energy_resolution, None, *extra_args)

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
        hnuminPhi_guess should be estimate at true angle

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
        
        angle_shape = angle_range.shape
        nmps = np.zeros(angle_shape)
        stds = np.zeros(angle_shape)
        
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
            
            parameters, pcov = fit_leastsq(
            parameters, energy_range, edge, fdir_initial,
            self.energy_resolution, None, *extra_args)

            nmps[indx] = parameters[0]
            stds[indx] = np.sqrt(np.diag(pcov)[0])
        
        # Offset at true angle if not set before
        if offset_guess is None:    
            offset_guess = hnuminPhi_guess - slope_guess * true_angle 
            
        parameters = np.array([offset_guess, slope_guess])
        
        lin_fun = Linear(offset_guess, slope_guess, 'Linear')
                    
        popt, pcov = fit_leastsq(parameters, angle_range, nmps, lin_fun, None,
                                 stds)

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


class MDCs:
    r"""
    Container for momentum distribution curves (MDCs) and their fits.

    This class stores the MDC intensity maps, angular and energy grids, and
    the aggregated fit results produced by :meth:`fit_selection`.

    Parameters
    ----------
    intensities : ndarray
        MDC intensity data. Typically a 2D array with shape
        ``(n_energy, n_angle)`` or a 1D array for a single curve.
    angles : ndarray
        Angular grid corresponding to the MDCs [degrees].
    angle_resolution : float
        Angular step size or effective angular resolution [degrees].
    enel : ndarray or float
        Electron binding energies of the MDC slices [eV].
        Can be a scalar for a single MDC.
    hnuminPhi : float
        Photon energy minus work function, used to convert ``enel`` to
        kinetic energy [eV].

    Attributes
    ----------
    intensities : ndarray
        MDC intensity data (same object as passed to the constructor).
    angles : ndarray
        Angular grid [degrees].
    angle_resolution : float
        Angular step size or resolution [degrees].
    enel : ndarray or float
        Electron binding energies [eV], as given at construction.
    ekin : ndarray or float
        Kinetic energies [eV], computed as ``enel + hnuminPhi``.
    hnuminPhi : float
        Photon energy minus work function [eV].
    ekin_range : ndarray
        Kinetic-energy values of the slices that were actually fitted.
        Set by :meth:`fit_selection`.
    individual_properties : dict
        Nested mapping of fitted parameters and their uncertainties for each
        component and each energy slice. Populated by :meth:`fit_selection`.

    Notes
    -----
    After calling :meth:`fit_selection`, :attr:`individual_properties` has the
    structure::

        {
            label: {
                class_name: {
                    'label': label,
                    '_class': class_name,
                    param:       [values per energy slice],
                    param_sigma: [1σ per slice or None],
                    ...
                }
            }
        }

    where ``param`` is typically one of ``'offset'``, ``'slope'``,
    ``'amplitude'``, ``'peak'``, ``'broadening'``, and ``param_sigma`` stores
    the corresponding uncertainty for each slice.
    
    """

    def __init__(self, intensities, angles, angle_resolution, enel, hnuminPhi):
        # Core input data (read-only)
        self._intensities = intensities
        self._angles = angles
        self._angle_resolution = angle_resolution
        self._enel = enel
        self._hnuminPhi = hnuminPhi

        # Derived attributes (populated by fit_selection)
        self._ekin_range = None
        self._individual_properties = None  # combined values + sigmas

    # -------------------- Immutable physics inputs --------------------

    @property
    def angles(self):
        """Angular axis for the MDCs."""
        return self._angles

    @property
    def angle_resolution(self):
        """Angular step size (float)."""
        return self._angle_resolution

    @property
    def enel(self):
        """Photoelectron binding energies (array-like). Read-only."""
        return self._enel

    @enel.setter
    def enel(self, _):
        raise AttributeError("`enel` is read-only; set it via the constructor.")

    @property
    def hnuminPhi(self):
        """Work-function/photon-energy offset. Read-only."""
        return self._hnuminPhi

    @hnuminPhi.setter
    def hnuminPhi(self, _):
        raise AttributeError("`hnuminPhi` is read-only; set it via the constructor.")

    @property
    def ekin(self):
        """Kinetic energy array: enel + hnuminPhi (computed on the fly)."""
        return self._enel + self._hnuminPhi

    @ekin.setter
    def ekin(self, _):
        raise AttributeError("`ekin` is derived and read-only.")

    # -------------------- Data arrays --------------------

    @property
    def intensities(self):
        """2D or 3D intensity map (energy × angle)."""
        return self._intensities

    @intensities.setter
    def intensities(self, x):
        self._intensities = x

    # -------------------- Results populated by fit_selection --------------------

    @property
    def ekin_range(self):
        """Kinetic-energy slices that were fitted."""
        if self._ekin_range is None:
            raise AttributeError("`ekin_range` not yet set. Run `.fit_selection()` first.")
        return self._ekin_range

    @property
    def individual_properties(self):
        """
        Aggregated fitted parameter values and uncertainties per component.

        Returns
        -------
        dict
            Nested mapping::

                {
                    label: {
                        class_name: {
                            'label': label,
                            '_class': class_name,
                            <param>:        [values per slice],
                            <param>_sigma:  [1σ per slice or None],
                            ...
                        }
                    }
                }
        """
        if self._individual_properties is None:
            raise AttributeError(
                "`individual_properties` not yet set. Run `.fit_selection()` first."
            )
        return self._individual_properties

    def energy_check(self, energy_value):
        r"""
        """
        if np.isscalar(self.ekin):
            if  energy_value is not None:
                raise ValueError("This dataset contains only one " \
                "momentum-distribution curve; do not provide energy_value.")
            else: 
                kinergy = self.ekin
                counts = self.intensities
        else:
            if energy_value is None:
                raise ValueError("This dataset contains multiple " \
                "momentum-distribution curves. Please provide an energy_value "
                "for which to plot the MDCs.")
            else:
                energy_index = np.abs(self.enel - energy_value).argmin()
                kinergy = self.ekin[energy_index]
                counts = self.intensities[energy_index, :]

                if not (self.enel.min() <= energy_value <= self.enel.max()):
                    raise ValueError(
                        f"Selected energy_value={energy_value:.3f} "
                        f"is outside the available energy range "
                        f"({self.enel.min():.3f} – {self.enel.max():.3f}) "
                        "of the MDC collection."
                    )

        return counts, kinergy


    def plot(self, energy_value=None, energy_range=None, ax=None, **kwargs):
        """
        Interactive or static plot with optional slider and full wrapper 
        support. Behavior consistent with Jupyter and CLI based on show / 
        fig_close.
        """
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider
        import string
        import sys
        import warnings

        # Wrapper kwargs
        title = kwargs.pop("title", None)
        savefig = kwargs.pop("savefig", None)
        show = kwargs.pop("show", True)
        fig_close = kwargs.pop("fig_close", False)
        tight_layout = kwargs.pop("tight_layout", False)
        ax_grid = kwargs.pop("ax_grid", None)
        ax_annotate = kwargs.pop("ax_annotate", False)
        size_kwargs = kwargs.pop("size_kwargs", None)

        if energy_value is not None and energy_range is not None:
            raise ValueError(
                "Provide at most energy_value or energy_range, not both.")

        ax, fig, plt = get_ax_fig_plt(ax=ax)

        angles = self.angles
        energies = self.enel

        if np.isscalar(energies):
            if energy_value is not None or energy_range is not None:
                raise ValueError(
                    "This dataset contains only one momentum-distribution "
                    "curve; do not provide energy_value or energy_range."
                )

            intensities = self.intensities
            ax.scatter(angles, intensities, label="Data")
            ax.set_title(f"Energy slice: {energies * KILO:.3f} meV")

            # --- y-only autoscale, preserve x ---
            x0, x1 = ax.get_xlim()                 # keep current x-range
            ax.relim(visible_only=True)            # recompute data limits
            ax.autoscale_view(scalex=False, scaley=True)
            ax.set_xlim(x0, x1)                    # restore x (belt-and-suspenders)

        else:
            if (energy_value is not None) and (energy_range is not None):
                raise ValueError("Provide either energy_value or energy_range, not both.")

            emin, emax = energies.min(), energies.max()

            # ---- Single-slice path (no slider) ----
            if energy_value is not None:
                if energy_value < emin or energy_value > emax:
                    raise ValueError(
                        f"Requested energy_value {energy_value:.3f} eV is "
                        f"outside the available energy range "
                        f"[{emin:.3f}, {emax:.3f}] eV."
                    )
                idx = int(np.abs(energies - energy_value).argmin())
                intensities = self.intensities[idx]
                ax.scatter(angles, intensities, label="Data")
                ax.set_title(f"Energy slice: {energies[idx] * KILO:.3f} meV")

                # --- y-only autoscale, preserve x ---
                x0, x1 = ax.get_xlim()                 # keep current x-range
                ax.relim(visible_only=True)            # recompute data limits
                ax.autoscale_view(scalex=False, scaley=True)
                ax.set_xlim(x0, x1)                    # restore x (belt-and-suspenders)

            # ---- Multi-slice path (slider) ----
            else:
                if energy_range is not None:
                    e_min, e_max = energy_range
                    mask = (energies >= e_min) & (energies <= e_max)
                else:
                    mask = np.ones_like(energies, dtype=bool)

                indices = np.where(mask)[0]
                if len(indices) == 0:
                    raise ValueError("No energies found in the specified selection.")

                intensities = self.intensities[indices]

                fig.subplots_adjust(bottom=0.25)
                idx = 0
                scatter = ax.scatter(angles, intensities[idx], label="Data")
                ax.set_title(f"Energy slice: "
                             f"{energies[indices[idx]] * KILO:.3f} meV")

                # Suppress single-point slider warning (when len(indices) == 1)
                warnings.filterwarnings(
                    "ignore",
                    message="Attempting to set identical left == right",
                    category=UserWarning
                )

                slider_ax = fig.add_axes([0.2, 0.08, 0.6, 0.04])
                slider = Slider(
                    slider_ax, "Index", 0, len(indices) - 1,
                    valinit=idx, valstep=1
                )

                def update(val):
                    i = int(slider.val)
                    yi = intensities[i]

                    scatter.set_offsets(np.c_[angles, yi])

                    x0, x1 = ax.get_xlim()

                    yv = np.asarray(yi, dtype=float).ravel()
                    mask = np.isfinite(yv)
                    if mask.any():
                        y_min = float(yv[mask].min())
                        y_max = float(yv[mask].max())
                        span  = y_max - y_min
                        frac  = plt.rcParams['axes.ymargin']

                        if span <= 0 or not np.isfinite(span):
                            scale = max(abs(y_max), 1.0)
                            pad = frac * scale
                        else:
                            pad = frac * span

                        ax.set_ylim(y_min - pad, y_max + pad)

                    # Keep x unchanged
                    ax.set_xlim(x0, x1)

                    # Update title and redraw
                    ax.set_title(f"Energy slice: "
                                 f"{energies[indices[i]] * KILO:.3f} meV")
                    fig.canvas.draw_idle()

                slider.on_changed(update)
                self._slider = slider
                self._line = scatter

        ax.set_xlabel("Angle (°)")
        ax.set_ylabel("Counts (-)")
        ax.legend()
        self._fig = fig

        if size_kwargs:
            fig.set_size_inches(size_kwargs.pop("w"),
                size_kwargs.pop("h"), **size_kwargs)
        if title:
            fig.suptitle(title)
        if tight_layout:
            fig.tight_layout()
        if savefig:
            fig.savefig(savefig)
        if ax_grid is not None:
            for axis in fig.axes:
                axis.grid(bool(ax_grid))
        if ax_annotate:
            tags = string.ascii_lowercase
            for i, axis in enumerate(fig.axes):
                axis.annotate(f"({tags[i]})", xy=(0.05, 0.95),
                    xycoords="axes fraction")

        is_interactive = hasattr(sys, 'ps1') or 'ipykernel' in sys.modules
        is_cli = not is_interactive

        if show:
            if is_cli:
                plt.show()
        if fig_close:
            plt.close(fig)

        if not show and (fig_close or is_cli):
            return None
        return fig


    @add_fig_kwargs
    def visualize_guess(self, distributions, energy_value=None,
                        matrix_element=None, matrix_args=None,
                        ax=None, **kwargs):
        r"""
        """
        
        counts, kinergy = self.energy_check(energy_value)

        ax, fig, plt = get_ax_fig_plt(ax=ax)

        ax.set_xlabel('Angle ($\\degree$)')
        ax.set_ylabel('Counts (-)')
        ax.set_title(f"Energy slice: "
                     f"{(kinergy - self.hnuminPhi) * KILO:.3f} meV")
        ax.scatter(self.angles, counts, label='Data')

        final_result = self._merge_and_plot(ax=ax, 
            distributions=distributions, kinetic_energy=kinergy,
            matrix_element=matrix_element,
            matrix_args=dict(matrix_args) if matrix_args else None,
            plot_individual=True,
        )

        residual = counts - final_result
        ax.scatter(self.angles, residual, label='Residual')
        ax.legend()

        return fig
    

    def fit_selection(self, distributions, energy_value=None, energy_range=None,
            matrix_element=None, matrix_args=None, ax=None, **kwargs):
        r"""
        """
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider
        from copy import deepcopy
        import string
        import sys
        import warnings
        from lmfit import Minimizer
        from scipy.ndimage import gaussian_filter
        from .functions import construct_parameters, build_distributions, \
            residual, resolve_param_name
        
        # Wrapper kwargs
        title = kwargs.pop("title", None)
        savefig = kwargs.pop("savefig", None)
        show = kwargs.pop("show", True)
        fig_close = kwargs.pop("fig_close", False)
        tight_layout = kwargs.pop("tight_layout", False)
        ax_grid = kwargs.pop("ax_grid", None)
        ax_annotate = kwargs.pop("ax_annotate", False)
        size_kwargs = kwargs.pop("size_kwargs", None)

        ax, fig, plt = get_ax_fig_plt(ax=ax)

        energies = self.enel
        new_distributions = deepcopy(distributions)

        if energy_value is not None and energy_range is not None:
            raise ValueError(
                "Provide at most energy_value or energy_range, not both.")

        if np.isscalar(energies):
            if energy_value is not None or energy_range is not None:
                raise ValueError(
                    "This dataset contains only one momentum-distribution "
                    "curve; do not provide energy_value or energy_range."
                )
            kinergies = np.atleast_1d(self.ekin)
            intensities = np.atleast_2d(self.intensities)

        else:
            if energy_value is not None:
                if (energy_value < energies.min() or energy_value > energies.max()):
                    raise ValueError( f"Requested energy_value {energy_value:.3f} eV is " 
                                     f"outside the available energy range " 
                                     f"[{energies.min():.3f}, {energies.max():.3f}] eV." )
                idx = np.abs(energies - energy_value).argmin()
                indices = np.atleast_1d(idx)
                kinergies = self.ekin[indices]
                intensities = self.intensities[indices, :]

            elif energy_range is not None:
                e_min, e_max = energy_range
                indices = np.where((energies >= e_min) & (energies <= e_max))[0]
                if len(indices) == 0:
                    raise ValueError("No energies found in the specified energy_range.")
                kinergies = self.ekin[indices]
                intensities = self.intensities[indices, :]

            else: # Without specifying a range, all MDCs are plotted
                kinergies = self.ekin
                intensities = self.intensities

        # Final shape guard
        kinergies = np.atleast_1d(kinergies)
        intensities = np.atleast_2d(intensities)

        all_final_results = []
        all_residuals = []
        all_individual_results = [] # List of (n_individuals, n_angles)

        aggregated_properties = {}

        # map class_name -> parameter names to extract
        param_spec = {
            'Constant': ('offset',),
            'Linear': ('offset', 'slope'),
            'SpectralLinear': ('amplitude', 'peak', 'broadening'),
            'SpectralQuadratic': ('amplitude', 'peak', 'broadening'),
        }

        order = np.argsort(kinergies)[::-1]
        for idx in order:
            kinergy   = kinergies[idx]
            intensity = intensities[idx]
            if matrix_element is not None:
                parameters, element_names = construct_parameters(
                    new_distributions, matrix_args)
                new_distributions = build_distributions(new_distributions, parameters)
                mini = Minimizer(
                    residual, parameters,
                    fcn_args=(self.angles, intensity, self.angle_resolution,
                              new_distributions, kinergy, self.hnuminPhi,
                              matrix_element, element_names)
                )
            else:
                parameters = construct_parameters(new_distributions)
                new_distributions = build_distributions(new_distributions, parameters)
                mini = Minimizer(
                    residual, parameters,
                    fcn_args=(self.angles, intensity, self.angle_resolution,
                              new_distributions, kinergy, self.hnuminPhi)
                )

            outcome = mini.minimize('least_squares')

            pcov = outcome.covar

            var_names = getattr(outcome, 'var_names', None)
            if not var_names:
                var_names = [n for n, p in outcome.params.items() if p.vary]
            var_idx = {n: i for i, n in enumerate(var_names)}

            param_sigma_full = {}
            for name, par in outcome.params.items():
                sigma = None
                if pcov is not None and name in var_idx:
                    d = pcov[var_idx[name], var_idx[name]]
                    if np.isfinite(d) and d >= 0:
                        sigma = float(np.sqrt(d))
                if sigma is None:
                    s = getattr(par, 'stderr', None)
                    sigma = float(s) if s is not None else None
                param_sigma_full[name] = sigma

            # Rebuild the *fitted* distributions from optimized params
            fitted_distributions = build_distributions(new_distributions, outcome.params)

            # If using a matrix element, extract slice-specific args from the fit
            if matrix_element is not None:
                new_matrix_args = {key: outcome.params[key].value for key in matrix_args}
            else:
                new_matrix_args = None

            # individual curves (smoothed, cropped) and final sum (no plotting here)
            extend, step, numb = extend_function(self.angles, self.angle_resolution)

            total_result_ext = np.zeros_like(extend)
            indiv_rows = [] # (n_individuals, n_angles)
            individual_labels = []

            for dist in fitted_distributions:
                # evaluate each component on the extended grid
                if getattr(dist, 'class_name', None) == 'SpectralQuadratic':
                    if (getattr(dist, 'center_angle', None) is not None) and (
                        kinergy is None or self.hnuminPhi is None
                    ):
                        raise ValueError(
                            'Spectral quadratic function is defined in terms '
                            'of a center angle. Please provide a kinetic energy '
                            'and hnuminPhi.'
                        )
                    extended_result = dist.evaluate(extend, kinergy, self.hnuminPhi)
                else:
                    extended_result = dist.evaluate(extend)

                if matrix_element is not None and hasattr(dist, 'index'):
                    args = new_matrix_args or {}
                    extended_result *= matrix_element(extend, **args)

                total_result_ext += extended_result

                # smoothed & cropped individual
                individual_curve = gaussian_filter(extended_result, sigma=step)[
                    numb:-numb if numb else None
                ]
                indiv_rows.append(np.asarray(individual_curve))

                # label
                label = getattr(dist, 'label', str(dist))
                individual_labels.append(label)

                # ---- collect parameters for this distribution
                # (Aggregated over slices)
                cls = getattr(dist, 'class_name', None)
                wanted = param_spec.get(cls, ())

                # ensure dicts exist
                label_bucket = aggregated_properties.setdefault(label, {})
                class_bucket = label_bucket.setdefault(
                    cls, {'label': label, '_class': cls}
                )

                # store center_wavevector (scalar) for SpectralQuadratic
                if (
                    cls == 'SpectralQuadratic'
                    and hasattr(dist, 'center_wavevector')
                ):
                    class_bucket.setdefault(
                        'center_wavevector', dist.center_wavevector
                    )

                # ensure keys for both values and sigmas
                for pname in wanted:
                    class_bucket.setdefault(pname, [])
                    class_bucket.setdefault(f"{pname}_sigma", [])

                # append values and sigmas in the order of slices
                for pname in wanted:
                    param_key = resolve_param_name(outcome.params, label, pname)

                    if param_key is not None and param_key in outcome.params:
                        class_bucket[pname].append(outcome.params[param_key].value)
                        class_bucket[f"{pname}_sigma"].append(param_sigma_full.get(param_key, None))
                    else:
                        # Not fitted in this slice → keep the value if present on the dist, sigma=None
                        class_bucket[pname].append(getattr(dist, pname, None))
                        class_bucket[f"{pname}_sigma"].append(None)

            # final (sum) curve, smoothed & cropped
            final_result_i = gaussian_filter(total_result_ext, sigma=step)[
                numb:-numb if numb else None]
            final_result_i = np.asarray(final_result_i)

            # Residual for this slice
            residual_i = np.asarray(intensity) - final_result_i

            # Store per-slice results
            all_final_results.append(final_result_i)
            all_residuals.append(residual_i)
            all_individual_results.append(np.vstack(indiv_rows))

        # --- after the reversed-order loop, restore original (ascending) order ---
        inverse_order = np.argsort(np.argsort(kinergies)[::-1])

        # Reorder per-slice arrays/lists computed in the loop
        all_final_results[:]      = [all_final_results[i]      for i in inverse_order]
        all_residuals[:]          = [all_residuals[i]          for i in inverse_order]
        all_individual_results[:] = [all_individual_results[i] for i in inverse_order]

        # Reorder all per-slice lists in aggregated_properties
        for label_dict in aggregated_properties.values():
            for cls_dict in label_dict.values():
                for key, val in cls_dict.items():
                    if isinstance(val, list) and len(val) == len(kinergies):
                        cls_dict[key] = [val[i] for i in inverse_order]

        self._ekin_range = kinergies
        self._individual_properties = aggregated_properties

        if np.isscalar(energies):
            # One slice only: plot MDC, Fit, Residual, and Individuals
            ydata = np.asarray(intensities).squeeze()
            yfit  = np.asarray(all_final_results[0]).squeeze()
            yres  = np.asarray(all_residuals[0]).squeeze()
            yind  = np.asarray(all_individual_results[0])

            ax.scatter(self.angles, ydata, label="Data")
            # plot individuals with their labels
            for j, lab in enumerate(individual_labels or []):
                ax.plot(self.angles, yind[j], label=str(lab))
            ax.plot(self.angles, yfit, label="Fit")
            ax.scatter(self.angles, yres, label="Residual")

            ax.set_title(f"Energy slice: {energies * KILO:.3f} meV")
            ax.relim()          # recompute data limits from all artists
            ax.autoscale_view() # apply autoscaling + axes.ymargin padding

        else:
            if energy_value is not None:
                _idx = int(np.abs(energies - energy_value).argmin())
                energies_sel = np.atleast_1d(energies[_idx])
            elif energy_range is not None:
                e_min, e_max = energy_range
                energies_sel = energies[(energies >= e_min) 
                                        & (energies <= e_max)]
            else:
                energies_sel = energies

            # Number of slices must match
            n_slices = len(all_final_results)
            assert intensities.shape[0] == n_slices == len(all_residuals) \
                == len(all_individual_results), (f"Mismatch: data \
                {intensities.shape[0]}, fits {len(all_final_results)}, "
                f"residuals {len(all_residuals)}, \
                individuals {len(all_individual_results)}."
            )
            n_individuals = all_individual_results[0].shape[0] \
                if n_slices else 0

            fig.subplots_adjust(bottom=0.25)
            idx = 0

            # Initial draw (MDC + Individuals + Fit + Residual) at slice 0
            scatter = ax.scatter(self.angles, intensities[idx], label="Data")

            individual_lines = []
            if n_individuals:
                for j in range(n_individuals):
                    if individual_labels and j < len(individual_labels):
                        label = str(individual_labels[j])
                    else:
                        label = f"Comp {j}"

                    yvals = all_individual_results[idx][j]
                    line, = ax.plot(self.angles, yvals, label=label)
                    individual_lines.append(line)

            result_line, = ax.plot(self.angles, all_final_results[idx], 
                                   label="Fit")
            resid_scatter = ax.scatter(self.angles, all_residuals[idx], 
                                       label="Residual")

            # Title + limits (use only the currently shown slice)
            ax.set_title(f"Energy slice: {energies_sel[idx] * KILO:.3f} meV")
            ax.relim()             # recompute data limits from all artists
            ax.autoscale_view()    # apply autoscaling + axes.ymargin padding

            # Suppress warning when a single MDC is plotted
            warnings.filterwarnings(
                "ignore",
                message="Attempting to set identical left == right",
                category=UserWarning
            )

            # Slider over slice index (0..n_slices-1)
            slider_ax = fig.add_axes([0.2, 0.08, 0.6, 0.04])
            slider = Slider(
                slider_ax, "Index", 0, n_slices - 1,
                valinit=idx, valstep=1
            )

            def update(val):
                i = int(slider.val)
                # Update MDC points
                scatter.set_offsets(np.c_[self.angles, intensities[i]])

                # Update individuals
                if n_individuals:
                    Yi = all_individual_results[i] # (n_individuals, n_angles)
                    for j, ln in enumerate(individual_lines):
                        ln.set_ydata(Yi[j])

                # Update fit and residual
                result_line.set_ydata(all_final_results[i])
                resid_scatter.set_offsets(np.c_[self.angles, all_residuals[i]])

                ax.relim()
                ax.autoscale_view()

                # Update title and redraw
                ax.set_title(f"Energy slice: "
                             f"{energies_sel[i] * KILO:.3f} meV")
                fig.canvas.draw_idle()

            slider.on_changed(update)
            self._slider = slider
            self._line = scatter
            self._individual_lines = individual_lines
            self._result_line = result_line
            self._resid_scatter = resid_scatter

        ax.set_xlabel("Angle (°)")
        ax.set_ylabel("Counts (-)")
        ax.legend()
        self._fig = fig

        if size_kwargs:
            fig.set_size_inches(size_kwargs.pop("w"),
                size_kwargs.pop("h"), **size_kwargs)
        if title:
            fig.suptitle(title)
        if tight_layout:
            fig.tight_layout()
        if savefig:
            fig.savefig(savefig)
        if ax_grid is not None:
            for axis in fig.axes:
                axis.grid(bool(ax_grid))
        if ax_annotate:
            tags = string.ascii_lowercase
            for i, axis in enumerate(fig.axes):
                axis.annotate(f"({tags[i]})", xy=(0.05, 0.95),
                    xycoords="axes fraction")

        is_interactive = hasattr(sys, 'ps1') or 'ipykernel' in sys.modules
        is_cli = not is_interactive

        if show:
            if is_cli:
                plt.show()
        if fig_close:
            plt.close(fig)

        if not show and (fig_close or is_cli):
            return None
        return fig
    

    @add_fig_kwargs
    def fit(self, distributions, energy_value=None, matrix_element=None, 
            matrix_args=None, ax=None, **kwargs):
        r"""
        """      
        from copy import deepcopy
        from lmfit import Minimizer
        from .functions import construct_parameters, build_distributions, \
            residual
        
        counts, kinergy = self.energy_check(energy_value)

        ax, fig, plt = get_ax_fig_plt(ax=ax)

        ax.set_xlabel('Angle ($\\degree$)')
        ax.set_ylabel('Counts (-)')
        ax.set_title(f"Energy slice: "
                     f"{(kinergy - self.hnuminPhi) * KILO:.3f} meV")
        
        ax.scatter(self.angles, counts, label='Data')

        new_distributions = deepcopy(distributions)

        if matrix_element is not None:
            parameters, element_names = construct_parameters(distributions,
                                                             matrix_args)
            new_distributions = build_distributions(new_distributions, \
                                                    parameters)
            mini = Minimizer(
                residual, parameters,
                fcn_args=(self.angles, counts, self.angle_resolution,
                          new_distributions, kinergy, self.hnuminPhi,
                          matrix_element, element_names))
        else:
            parameters = construct_parameters(distributions)
            new_distributions = build_distributions(new_distributions,
                                                    parameters)
            mini = Minimizer(residual, parameters,
                fcn_args=(self.angles, counts, self.angle_resolution,
                          new_distributions, kinergy, self.hnuminPhi))

        outcome = mini.minimize('least_squares')
        pcov = outcome.covar
        
        # If matrix params were fitted, pass the fitted values to plotting
        if matrix_element is not None:
            new_matrix_args = {key: outcome.params[key].value for key in
                               matrix_args}
        else:
            new_matrix_args = None

        final_result = self._merge_and_plot(ax=ax, 
            distributions=new_distributions, kinetic_energy=kinergy,
            matrix_element=matrix_element, matrix_args=new_matrix_args,
            plot_individual=True)
        
        residual_vals = counts - final_result
        ax.scatter(self.angles, residual_vals, label='Residual')
        ax.legend()
        if matrix_element is not None:
            return fig, new_distributions, pcov, new_matrix_args
        else:
            return fig, new_distributions, pcov
        

    def _merge_and_plot(self, ax, distributions, kinetic_energy,
                        matrix_element=None, matrix_args=None,
                        plot_individual=True):
        r"""
        Evaluate distributions on the extended grid, apply optional matrix
        element, smooth, plot individuals and the summed curve.

        Returns
        -------
        final_result : np.ndarray
            Smoothed, cropped total distribution aligned with self.angles.
        """
        from scipy.ndimage import gaussian_filter

        # Build extended grid
        extend, step, numb = extend_function(self.angles, self.angle_resolution)
        total_result = np.zeros_like(extend)

        for dist in distributions:
            # Special handling for SpectralQuadratic
            if getattr(dist, 'class_name', None) == 'SpectralQuadratic':
                if (getattr(dist, 'center_angle', None) is not None) and (
                    kinetic_energy is None or self.hnuminPhi is None
                ):
                    raise ValueError(
                        'Spectral quadratic function is defined in terms '
                        'of a center angle. Please provide a kinetic energy '
                        'and hnuminPhi.'
                    )
                extended_result = dist.evaluate(extend, kinetic_energy, \
                                                self.hnuminPhi)
            else:
                extended_result = dist.evaluate(extend)

            # Optional matrix element (only for components that advertise an index)
            if matrix_element is not None and hasattr(dist, 'index'):
                args = matrix_args or {}
                extended_result *= matrix_element(extend, **args)

            total_result += extended_result

            if plot_individual and ax:
                individual = gaussian_filter(extended_result, sigma=step)\
                    [numb:-numb if numb else None]
                ax.plot(self.angles, individual, label=getattr(dist, \
                                                        'label', str(dist)))

        # Smoothed, cropped total curve aligned to self.angles
        final_result = gaussian_filter(total_result, sigma=step)[numb:-numb \
                                                            if numb else None]
        if ax:
            ax.plot(self.angles, final_result, label='Distribution sum')

        return final_result
    

    def expose_parameters(self, select_label, fermi_wavevector=None,
                          fermi_velocity=None, bare_mass=None, side=None):
        r"""
        Select and return fitted parameters for a given component label, plus a
        flat export dictionary containing values **and** 1σ uncertainties.

        Parameters
        ----------
        select_label : str
            Label to look for among the fitted distributions.
        fermi_wavevector : float, optional
            Optional Fermi wave vector to include.
        fermi_velocity : float, optional
            Optional Fermi velocity to include.
        bare_mass : float, optional
            Optional bare mass to include (used for SpectralQuadratic
            dispersions).
        side : {'left','right'}, optional
            Optional side selector for SpectralQuadratic dispersions.

        Returns
        -------
        ekin_range : np.ndarray
            Kinetic-energy grid corresponding to the selected label.
        hnuminPhi : float
            Photoelectron work-function offset.
        label : str
            Label of the selected distribution.
        selected_properties : dict or list of dict
            Nested dictionary (or list thereof) containing <param> and
            <param>_sigma arrays. For SpectralQuadratic components, a
            scalar `center_wavevector` is also present.
        exported_parameters : dict
            Flat dictionary of parameters and their uncertainties, plus
            optional Fermi quantities and `side`. For SpectralQuadratic
            components, `center_wavevector` is included and taken directly
            from the fitted distribution.
        """

        if self._ekin_range is None:
            raise AttributeError(
                "ekin_range not yet set. Run `.fit_selection()` first."
            )

        store = getattr(self, "_individual_properties", None)
        if not store or select_label not in store:
            all_labels = (sorted(store.keys())
                          if isinstance(store, dict) else [])
            raise ValueError(
                f"Label '{select_label}' not found in available labels: "
                f"{all_labels}"
            )

        # Convert lists → numpy arrays within the selected label’s classes.
        # Keep scalar center_wavevector as a scalar.
        per_class_dicts = []
        for cls, bucket in store[select_label].items():
            dct = {}
            for k, v in bucket.items():
                if k in ("label", "_class"):
                    dct[k] = v
                elif k == "center_wavevector":
                    # keep scalar as-is, do not wrap in np.asarray
                    dct[k] = v
                else:
                    dct[k] = np.asarray(v)
            per_class_dicts.append(dct)

        selected_properties = (
            per_class_dicts[0] if len(per_class_dicts) == 1 else per_class_dicts
        )

        # Flat export dict: simple keys, includes optional extras
        exported_parameters = {
            "fermi_wavevector": fermi_wavevector,
            "fermi_velocity": fermi_velocity,
            "bare_mass": bare_mass,
            "side": side,
        }

        # Collect parameters without prefixing by class. This will also include
        # center_wavevector from the fitted SpectralQuadratic class, and since
        # there is no function argument with that name, it cannot be overridden.
        if isinstance(selected_properties, dict):
            for key, val in selected_properties.items():
                if key not in ("label", "_class"):
                    exported_parameters[key] = val
        else:
            # If multiple classes, merge sequentially
            # (last overwrites same-name keys).
            for cls_bucket in selected_properties:
                for key, val in cls_bucket.items():
                    if key not in ("label", "_class"):
                        exported_parameters[key] = val

        return (self._ekin_range, self.hnuminPhi, select_label,
                selected_properties, exported_parameters)


class SelfEnergy:
    r"""Self-energy (ekin-leading; hnuminPhi/ekin are read-only)."""

    def __init__(self, ekin_range, hnuminPhi, label, properties, parameters):
        # core read-only state
        self._ekin_range = ekin_range
        self._hnuminPhi = hnuminPhi
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
        r"""k_parallel = peak * dtor * sqrt(ekin_range / pref) (lazy)."""
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