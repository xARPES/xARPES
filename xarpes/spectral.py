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
from .constants import uncr, pref, dtor, kilo

class BandMap:
    r"""Class for the band map from the ARPES experiment."""

    def __init__(self, datafile=None, intensities=None, angles=None,
                 ekin=None, enel=None, energy_resolution=None,
                 angle_resolution=None, temperature=None, hnuminphi=None,
                 hnuminphi_std=None, transpose=False, flip_ekin=False,
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
        self._hnuminphi = None
        self._hnuminphi_std = None
        self.hnuminphi = hnuminphi
        self.hnuminphi_std = hnuminphi_std

        # --- 1) Track which axis is authoritative --------------------------
        self._ekin_explicit = ekin is not None or (file_ekin is not None
                                                   and enel is None)
        self._enel_explicit = enel is not None

        # --- 2) Derive missing axis if possible ----------------------------
        if self._ekin is None and self._enel is not None \
                and self._hnuminphi is not None:
            self._ekin = self._enel + self._hnuminphi
        if self._enel is None and self._ekin is not None \
                and self._hnuminphi is not None:
            self._enel = self._ekin - self._hnuminphi

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

    # -------------------- 4) Sync ekin / enel / hnuminphi ------------------
    @property
    def ekin(self):
        if self._ekin is None and self._enel is not None \
                and self._hnuminphi is not None:
            return self._enel + self._hnuminphi
        return self._ekin

    @ekin.setter
    def ekin(self, x):
        if getattr(self, "_enel_explicit", False):
            raise AttributeError('enel is explicit; set hnuminphi instead.')
        self._ekin = x
        self._ekin_explicit = True
        if not getattr(self, "_enel_explicit", False) \
                and self._hnuminphi is not None and x is not None:
            self._enel = x - self._hnuminphi

    @property
    def enel(self):
        if self._enel is None and self._ekin is not None \
                and self._hnuminphi is not None:
            return self._ekin - self._hnuminphi
        return self._enel

    @enel.setter
    def enel(self, x):
        if getattr(self, "_ekin_explicit", False):
            raise AttributeError('ekin is explicit; set hnuminphi instead.')
        self._enel = x
        self._enel_explicit = True
        if not getattr(self, "_ekin_explicit", False) \
                and self._hnuminphi is not None and x is not None:
            self._ekin = x + self._hnuminphi
            
    @property
    def hnuminphi(self):
        r"""Returns the photon energy minus the work function in eV if it has
        been set, either during instantiation, with the setter, or by fitting
        the Fermi-Dirac distribution to the integrated weight.

        Returns
        -------
        hnuminphi : float, None
            Kinetic energy minus the work function [eV]
        """
        return self._hnuminphi

    @hnuminphi.setter
    def hnuminphi(self, x):
        r"""TBD
        """
        self._hnuminphi = x
        # Re-derive the non-explicit axis if possible
        if not getattr(self, "_ekin_explicit", False) \
                and self._enel is not None and x is not None:
            self._ekin = self._enel + x
        if not getattr(self, "_enel_explicit", False) \
                and self._ekin is not None and x is not None:
            self._enel = self._ekin - x

    @property
    def hnuminphi_std(self):
        r"""Returns standard deviation of the photon energy minus the work
        function in eV.

        Returns
        -------
        hnuminphi_std : float
            Standard deviation of energy minus the work function [eV]
        """
        return self._hnuminphi_std

    @hnuminphi_std.setter
    def hnuminphi_std(self, x):
        r"""Manually sets the standard deviation of photon energy minus the
        work function in eV.

        Parameters
        ----------
        hnuminphi_std : float
            Standard deviation of energy minus the work function [eV]
        """
        self._hnuminphi_std = x

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
        mdcs :
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
        enel_range_out, self.hnuminphi


    @add_fig_kwargs
    def plot(self, abscissa='momentum', ordinate='electron_energy', ax=None,
            **kwargs):
        
        # Validate options early
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

        ax, fig, plt = get_ax_fig_plt(ax=ax)

        Angl, Ekin = np.meshgrid(self.angles, self.ekin)
        mesh = None  # sentinel to detect missing branch

        if abscissa == 'angle':
            ax.set_xlabel('Angle ($\\degree$)')
            if ordinate == 'kinetic_energy':
                mesh = ax.pcolormesh(
                    Angl, Ekin, self.intensities, shading='auto',
                    cmap=plt.get_cmap('bone').reversed(), **kwargs
                )
                ax.set_ylabel('$E_{\\mathrm{kin}}$ (eV)')
            elif ordinate == 'electron_energy':
                Enel = Ekin - self.hnuminphi
                mesh = ax.pcolormesh(
                    Angl, Enel, self.intensities, shading='auto',
                    cmap=plt.get_cmap('bone').reversed(), **kwargs
                )
                ax.set_ylabel('$E-\\mu$ (eV)')

        elif abscissa == 'momentum':
            Mome = np.sqrt(Ekin / pref) * np.sin(Angl * dtor)
            ax.set_xlabel(r'$k_{//}$ ($\mathrm{\AA}^{-1}$)')
            if ordinate == 'kinetic_energy':
                mesh = ax.pcolormesh(
                    Mome, Ekin, self.intensities, shading='auto',
                    cmap=plt.get_cmap('bone').reversed(), **kwargs
                )
                ax.set_ylabel('$E_{\\mathrm{kin}}$ (eV)')
            elif ordinate == 'electron_energy':
                Enel = Ekin - self.hnuminphi
                mesh = ax.pcolormesh(
                    Mome, Enel, self.intensities, shading='auto',
                    cmap=plt.get_cmap('bone').reversed(), **kwargs
                )
                ax.set_ylabel('$E-\\mu$ (eV)')

        # If no branch set 'mesh', fail with a clear message
        if mesh is None:
            raise RuntimeError(
                "No plot produced for the combination: "
                f"abscissa='{abscissa}', ordinate='{ordinate}'. "
                f"Valid abscissa: {valid_abscissa}; "
                f"valid ordinate: {valid_ordinate}."
            )

        plt.colorbar(mesh, ax=ax, label='counts (-)')
        return fig


    @add_fig_kwargs
    def fit_fermi_edge(self, hnuminphi_guess, background_guess=0.0,
                       integrated_weight_guess=1.0, angle_min=-np.inf,
                       angle_max=np.inf, ekin_min=-np.inf,
                       ekin_max=np.inf, ax=None, **kwargs):
        r"""Fits the Fermi edge of the band map and plots the result.
        Also sets hnuminphi, the kinetic energy minus the work function in eV.
        The fitting includes an energy convolution with an abscissa range
        expanded by 5 times the energy resolution standard deviation.

        Parameters
        ----------
        hnuminphi_guess : float
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
                                  hnuminphi=hnuminphi_guess,
                                  background=background_guess,
                                  integrated_weight=integrated_weight_guess,
                                  name='Initial guess')

        parameters = np.array(
            [hnuminphi_guess, background_guess, integrated_weight_guess])

        extra_args = (self.temperature,)

        popt, pcov = fit_leastsq(
        parameters, energy_range, integrated_intensity, fdir_initial,
        self.energy_resolution, None, *extra_args)

        # Update hnuminphi; automatically sets self.enel
        self.hnuminphi = popt[0]
        self.hnuminphi_std = np.sqrt(np.diag(pcov)[0])

        fdir_final = FermiDirac(temperature=self.temperature,
                                hnuminphi=self.hnuminphi, background=popt[1],
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
    def correct_fermi_edge(self, hnuminphi_guess=None, background_guess=0.0,
                       integrated_weight_guess=1.0, angle_min=-np.inf,
                       angle_max=np.inf, ekin_min=-np.inf, ekin_max=np.inf,
                       slope_guess=0, offset_guess=None,
                           true_angle=0, ax=None, **kwargs):
        r"""TBD
        hnuminphi_guess should be estimate at true angle

        Parameters
        ----------

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
        
        if hnuminphi_guess is None:
            raise ValueError('Please provide an initial guess for ' +
                             'hnuminphi.')
 
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
        
        hnuminphi_left = hnuminphi_guess - (true_angle - angle_min) \
        * slope_guess
  
        fdir_initial = FermiDirac(temperature=self.temperature,
                      hnuminphi=hnuminphi_left,
                      background=background_guess,
                      integrated_weight=integrated_weight_guess,
                      name='Initial guess')
        
        parameters = np.array(
                [hnuminphi_left, background_guess, integrated_weight_guess])
        
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
            offset_guess = hnuminphi_guess - slope_guess * true_angle 
            
        parameters = np.array([offset_guess, slope_guess])
        
        lin_fun = Linear(offset_guess, slope_guess, 'Linear')
                    
        popt, pcov = fit_leastsq(parameters, angle_range, nmps, lin_fun, None,
                                 stds)

        linsp = lin_fun(angle_range, popt[0], popt[1])
            
        # Update hnuminphi; automatically sets self.enel
        self.hnuminphi = lin_fun(true_angle, popt[0], popt[1])
        self.hnuminphi_std = np.sqrt(true_angle**2 * pcov[1, 1] + pcov[0, 0] 
                                     + 2 * true_angle * pcov[0, 1])
                    
        Angl, Ekin = np.meshgrid(self.angles, self.ekin)

        ax, fig, plt = get_ax_fig_plt(ax=ax)
        
        ax.set_xlabel('Angle ($\degree$)')
        ax.set_ylabel('$E_{\mathrm{kin}}$ (eV)')
        mesh = ax.pcolormesh(Angl, Ekin, self.intensities,
                       shading='auto', cmap=plt.get_cmap('bone').reversed(),
                             zorder=1)

        ax.errorbar(angle_range, nmps, yerr=uncr * stds, zorder=1)
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
    Momentum Distribution Curves (MDC) container for fitted ARPES data.

    Holds the raw intensity maps, angular and energy grids, and the
    post-fit aggregated parameters and uncertainties produced by
    `.fit_selection()`.
    """

    def __init__(self, intensities, angles, angle_resolution, enel, hnuminphi):
        # Core input data (read-only)
        self._intensities = intensities
        self._angles = angles
        self._angle_resolution = angle_resolution
        self._enel = enel
        self._hnuminphi = hnuminphi

        # Derived attributes
        self._ekin_range = None

        # Aggregated results (populated by fit_selection)
        self._individual_properties = None
        self._individual_uncertainties = None

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
    def hnuminphi(self):
        """Work-function/photon-energy offset. Read-only."""
        return self._hnuminphi

    @hnuminphi.setter
    def hnuminphi(self, _):
        raise AttributeError("`hnuminphi` is read-only; set it via the constructor.")

    @property
    def ekin(self):
        """Kinetic energy array: enel + hnuminphi (computed on the fly)."""
        return self._enel + self._hnuminphi

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
        Aggregated fitted parameter values per distribution component.

        Returns
        -------
        dict
            Nested mapping:
            {
                label: {
                    class_name: {
                        'label': label,
                        '_class': class_name,
                        <param_name>: [values per slice],
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

    @property
    def individual_uncertainties(self):
        """
        Aggregated 1σ uncertainties aligned with `individual_properties`.

        Returns
        -------
        dict
            Same nested structure as `individual_properties`, but with
            lists of uncertainty values (or None for fixed parameters).
        """
        if self._individual_uncertainties is None:
            raise AttributeError(
                "`individual_uncertainties` not yet set. Run `.fit_selection()` first."
            )
        return self._individual_uncertainties


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
            ax.set_title(f"Energy slice: {energies * kilo:.3f} meV")

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
                ax.set_title(f"Energy slice: {energies[idx] * kilo:.3f} meV")

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
                             f"{energies[indices[idx]] * kilo:.3f} meV")

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
                                 f"{energies[indices[i]] * kilo:.3f} meV")
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
                     f"{(kinergy - self.hnuminphi) * kilo:.3f} meV")
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
        import re
        from lmfit import Minimizer
        from scipy.ndimage import gaussian_filter
        from .functions import construct_parameters, build_distributions, \
            residual
        
        def _resolve_param_name(params, label, pname):
            """
            Try to find the lmfit param key corresponding to this component `label`
            and bare parameter name `pname` (e.g., 'amplitude', 'peak', 'broadening').
            Works with common token separators.
            """
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
        aggregated_uncertainties = {}

        # map class_name -> parameter names to extract
        param_spec = {
            'Constant': ('offset',),
            'Linear': ('offset', 'slope'),
            'SpectralLinear': ('amplitude', 'peak', 'broadening'),
            'SpectralQuadratic': ('amplitude', 'peak', 'broadening'),
        }

        for kinergy, intensity in zip(kinergies, intensities):
            if matrix_element is not None:
                parameters, element_names = construct_parameters(
                    new_distributions, matrix_args)
                new_distributions = build_distributions(new_distributions, parameters)
                mini = Minimizer(
                    residual, parameters,
                    fcn_args=(self.angles, intensity, self.angle_resolution,
                              new_distributions, kinergy, self.hnuminphi,
                              matrix_element, element_names)
                )
            else:
                parameters = construct_parameters(new_distributions)
                new_distributions = build_distributions(new_distributions, parameters)
                mini = Minimizer(
                    residual, parameters,
                    fcn_args=(self.angles, intensity, self.angle_resolution,
                              new_distributions, kinergy, self.hnuminphi)
                )

            outcome = mini.minimize('least_squares')

            pcov = outcome.covar

            # lmfit gives var_names for the covariance order (varying params only)
            var_names = getattr(outcome, 'var_names', None)
            if not var_names:
                var_names = [n for n, p in outcome.params.items() if p.vary]

            var_idx = {n: i for i, n in enumerate(var_names)}

            # Build a FULL map: every param name -> sigma (sqrt diag if available, 
            # else stderr, else None)
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
                        kinergy is None or self.hnuminphi is None
                    ):
                        raise ValueError(
                            'Spectral quadratic function is defined in terms '
                            'of a center angle. Please provide a kinetic energy '
                            'and hnuminphi.'
                        )
                    extended_result = dist.evaluate(extend, kinergy, self.hnuminphi)
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
                class_bucket = label_bucket.setdefault(cls, {'label': label, '_class': cls})
                unc_label_bucket = aggregated_uncertainties.setdefault(label, {})
                unc_class_bucket = unc_label_bucket.setdefault(cls, {'label': label, '_class': cls})
                # ensure keys exist

                for pname in wanted:
                    class_bucket.setdefault(pname, [])
                    unc_class_bucket.setdefault(pname, [])

                for pname in wanted:
                    param_key = _resolve_param_name(outcome.params, label, pname)

                    if param_key is not None and param_key in outcome.params:
                        class_bucket[pname].append(outcome.params[param_key].value)
                        unc_class_bucket[pname].append(param_sigma_full.get(param_key, None))
                    else:
                        # Fallback to attribute on the distribution (not fitted this slice)
                        class_bucket[pname].append(getattr(dist, pname, None))
                        unc_class_bucket[pname].append(None)

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

        self._ekin_range = kinergies
        self._individual_properties = aggregated_properties
        self._individual_uncertainties = aggregated_uncertainties

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

            ax.set_title(f"Energy slice: {energies * kilo:.3f} meV")
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
            ax.set_title(f"Energy slice: {energies_sel[idx] * kilo:.3f} meV")
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
                             f"{energies_sel[i] * kilo:.3f} meV")
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
                     f"{(kinergy - self.hnuminphi) * kilo:.3f} meV")
        
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
                          new_distributions, kinergy, self.hnuminphi,
                          matrix_element, element_names))
        else:
            parameters = construct_parameters(distributions)
            new_distributions = build_distributions(new_distributions,
                                                    parameters)
            mini = Minimizer(residual, parameters,
                fcn_args=(self.angles, counts, self.angle_resolution,
                          new_distributions, kinergy, self.hnuminphi))

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
                    kinetic_energy is None or self.hnuminphi is None
                ):
                    raise ValueError(
                        'Spectral quadratic function is defined in terms '
                        'of a center angle. Please provide a kinetic energy '
                        'and hnuminphi.'
                    )
                extended_result = dist.evaluate(extend, kinetic_energy, \
                                                self.hnuminphi)
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
                          fermi_velocity=None):
        r"""
        Selects and returns the fitted parameters for a given label, together
        with optional user-specified physical parameters (as a dictionary).

        Parameters
        ----------
        select_label : str
            Label to look for among the fitted distributions.
        fermi_wavevector : float, optional
            Optional Fermi wave vector to include.
        fermi_velocity : float, optional
            Optional Fermi velocity to include.
        **kwargs :
            Any other user-defined quantities to export.

        Returns
        -------
        tuple
            (ekin_range, hnuminphi, label, properties, exported_parameters)

            where `exported_parameters` is a dict containing all optional
            quantities passed by the user.
        """
        if self._ekin_range is None:
            raise AttributeError(
                "ekin_range not yet set. Run `.fit_selection()` first."
            )

        store = getattr(self, "_individual_properties", None)
        if not store or select_label not in store:
            all_labels = (sorted(store.keys()) if isinstance(store, dict)
                        else [])
            raise ValueError(
                f"Label '{select_label}' not found in available labels: "
                f"{all_labels}"
            )

        # Convert lists → numpy arrays
        per_class_dicts = []
        for cls, bucket in store[select_label].items():
            dct = {}
            for k, v in bucket.items():
                dct[k] = v if k in ("label", "_class") else np.asarray(v)
            per_class_dicts.append(dct)

        selected_properties = (
            per_class_dicts[0] if len(per_class_dicts) == 1 else per_class_dicts
        )

        # Flexible dictionary of optional parameters
        exported_parameters = {
            "fermi_wavevector": fermi_wavevector,
            "fermi_velocity": fermi_velocity,
        }

        return self._ekin_range, self.hnuminphi, select_label, \
            selected_properties, exported_parameters


class SelfEnergy:
    r"""Self-energy (ekin-leading; hnuminphi/ekin are read-only)."""

    def __init__(self, ekin_range, hnuminphi, label, properties, parameters):
        # core read-only state
        self._ekin_range = ekin_range
        self._hnuminphi = hnuminphi
        self._label = label

        # optional, user-supplied extras (can be set later)
        self._parameters = dict(parameters or {})
        self._fermi_wavevector = self._parameters.get("fermi_wavevector")
        self._fermi_velocity = self._parameters.get("fermi_velocity")

        # optional parameter arrays from fit
        self._amplitude = properties.get("amplitude")
        self._peak = properties.get("peak")
        self._broadening = properties.get("broadening")
        self._class = properties.get("_class", None)

        # lazy caches
        self._peak_positions = None
        self._real = None  # real part cahce
        self._imag = None  # imaginary part cache

    # ---------------- core read-only axes ----------------

    @property
    def ekin_range(self):
        return self._ekin_range

    @property
    def enel_range(self):
        if self._ekin_range is None:
            return None
        hnp = 0.0 if self._hnuminphi is None else self._hnuminphi
        return np.asarray(self._ekin_range) - hnp

    @property
    def hnuminphi(self):
        return self._hnuminphi

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
    def fermi_wavevector(self):
        """Optional k_F; can be set later."""
        return self._fermi_wavevector

    @fermi_wavevector.setter
    def fermi_wavevector(self, x):
        self._fermi_wavevector = x
        self._parameters["fermi_wavevector"] = x
        self._real = None # invalidate dependent cache

    @property
    def fermi_velocity(self):
        """Optional v_F; can be set later."""
        return self._fermi_velocity

    @fermi_velocity.setter
    def fermi_velocity(self, x):
        self._fermi_velocity = x
        self._parameters["fermi_velocity"] = x
        self._imag = None # invalidate dependent cache
        self._real = None # invalidate dependent cache

    # ---------------- optional fit parameters ----------------
    @property
    def amplitude(self):
        return self._amplitude

    @amplitude.setter
    def amplitude(self, x):
        self._amplitude = x

    @property
    def peak(self):
        return self._peak

    @peak.setter
    def peak(self, x):
        self._peak = x
        self._peak_positions = None # invalidate dependent cache
        self._real = None # invalidate dependent cache

    @property
    def broadening(self):
        return self._broadening

    @broadening.setter
    def broadening(self, x):
        self._broadening = x
        self._imag = None # invalidate dependent cache

    # ---------------- derived outputs ----------------
    @property
    def peak_positions(self):
        r"""k_parallel = peak * dtor * sqrt(ekin_range / pref) (lazy)."""
        if getattr(self, "_peak_positions", None) is None:
            if self._peak is None or self._ekin_range is None:
                return None
            self._peak_positions = (
                np.asarray(self._peak) * dtor *
                np.sqrt(np.asarray(self._ekin_range) / pref)
            )
        return self._peak_positions

    @property
    def imag(self):
        r"""-Σ'': v_F * sqrt(E_kin / pref) * broadening (lazy)."""
        if getattr(self, "_imag", None) is None:
            if self._fermi_velocity is None:
                raise AttributeError(
                    "Cannot compute `imag`: fermi_velocity not set. "
                    "Provide it via `self.fermi_velocity = ...`."
                )
            if self._broadening is None or self._ekin_range is None:
                return None
            self._imag = self._fermi_velocity * np.sqrt(self._ekin_range / pref) * \
                self._broadening
            
        return self._imag
    
    @property
    def real(self):
        r"""Real part of Σ (lazy, cached).
        Depends on: fermi_velocity, fermi_wavevector, enel_range, peak.
        """
        if getattr(self, "_real", None) is None:
            if self._fermi_velocity is None or self._fermi_wavevector is None:
                raise AttributeError(
                    "Cannot compute `real`: set both fermi_velocity and "
                    "fermi_wavevector first."
                )
            if self._peak is None or self._ekin_range is None:
                return None
            self._real = self.enel_range - self._fermi_velocity * \
            (self.peak_positions - self._fermi_wavevector)

        return self._real