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

class BandMap():
    r"""Class for the band map from the ARPES experiment.

    Parameters
    ----------
    datafile : str
        Name of data file. Currently, only IGOR Binary Wave files are
        supported. If absent, `intensities`, `angles`, and `ekin` are
        mandatory. Otherwise, those arguments can be used to overwrite the 
        contents of `datafile`.
    intensities : ndarray
        2D array of counts for given (E,k) or (E,angle) pairs [counts]
    angles : ndarray
        1D array of angle values for the abscissa [degrees]
    ekin : ndarray
        1D array of kinetic energy values for the ordinate [eV]
    enel : ndarray
        1D array of electron energy (E-\mu) values for the ordinate [eV]
    angle_resolution : float, None
        Angle resolution of the detector [degrees]
    energy_resolution : float, None
        Energy resolution of the detector [eV]
    temperature : float, None
        Temperature of the sample [K]
    hnuminphi : float, None
        Kinetic energy minus the work function [eV]
    hnuminphi_std : float, None
        Standard deviation of kinetic energy minus work function [eV]
    transpose : bool, False
        Are the energy and angle axes swapped (angle first) in the input file?
    flip_ekin : bool, False
        Reverse energy axis of ``wData`` in `datafile`?
    flip_angles : bool, False
        Reverse angle axis of ``wData`` in `datafile`?
    """
    def __init__(self, datafile=None, intensities=None, angles=None, ekin=None,
                 enel=None, energy_resolution=None, angle_resolution=None,
                 temperature=None, hnuminphi=None, hnuminphi_std=None,
                 transpose=False, flip_ekin=False, flip_angles=False):
        
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
            self.ekin = np.linspace(fmin, fmin + (fnum - 1) * fstp, fnum)

        if intensities is not None:
            self.intensities = intensities
        elif datafile is None:
            raise ValueError('Please provide datafile or intensities.')

        if angles is not None:
            self.angles = angles
        elif datafile is None:
            raise ValueError('Please provide datafile or angles.')

        if ekin is not None:
            self.ekin = ekin
        elif datafile is None:
            raise ValueError('Please provide datafile or ekin.')

        self.energy_resolution = energy_resolution
        self.angle_resolution = angle_resolution
        self.temperature = temperature
        self.hnuminphi = hnuminphi
        self.hnuminphi_std = hnuminphi_std

    # Band map is still missing some properties and setters

    @property
    def ekin(self):
        r"""
        """
        return self._ekin
    
    @ekin.setter
    def ekin(self, x):
        r"""
        """
        self._ekin = x
    
    @property
    def enel(self):
        r"""
        """
        return self._enel

    @enel.setter
    def enel(self, x):
        r"""
        """
        self._enel = x

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
        r"""Manually sets the photon energy minus the work function in eV if 
        it has been set; otherwise returns None.

        Parameters
        ----------
        hnuminphi : float, None
            Kinetic energy minus the work function [eV]
        """
        self._hnuminphi = x

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
        r"""Plots the band map.

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

        ax, fig, plt = get_ax_fig_plt(ax=ax)

        Angl, Ekin = np.meshgrid(self.angles, self.ekin)

        if abscissa == 'angle':
            ax.set_xlabel('Angle ($\degree$)')
            if ordinate == 'kinetic_energy':
                mesh = ax.pcolormesh(Angl, Ekin, self.intensities,
                       shading='auto', cmap=plt.get_cmap('bone').reversed())
                ax.set_ylabel('$E_{\mathrm{kin}}$ (eV)')
            elif ordinate == 'electron_energy':
                Enel = Ekin - self.hnuminphi
                mesh = ax.pcolormesh(Angl, Enel, self.intensities,
                       shading='auto', cmap=plt.get_cmap('bone').reversed())
                ax.set_ylabel('$E-\mu$ (eV)')
        elif abscissa == 'momentum':
            Mome = np.sqrt(Ekin / pref) * np.sin(Angl * dtor)
            ax.set_xlabel(r'$k_{//}$ ($\mathrm{\AA}^{\endash1}$)')
            if ordinate == 'kinetic_energy':
                mesh = ax.pcolormesh(Mome, Ekin, self.intensities,
                       shading='auto', cmap=plt.get_cmap('bone').reversed())
                ax.set_ylabel('$E_{\mathrm{kin}}$ (eV)')
            elif ordinate == 'electron_energy':
                Enel = Ekin - self.hnuminphi
                mesh = ax.pcolormesh(Mome, Enel, self.intensities,
                       shading='auto', cmap=plt.get_cmap('bone').reversed())
                ax.set_ylabel('$E-\mu$ (eV)')

        cbar = plt.colorbar(mesh, ax=ax, label='counts (-)')

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

        self.hnuminphi = popt[0]
        self.hnuminphi_std = np.sqrt(np.diag(pcov)[0])
        self.enel = self.ekin - self.hnuminphi

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
            
        self.hnuminphi = lin_fun(true_angle, popt[0], popt[1])
        self.hnuminphi_std = np.sqrt(true_angle**2 * pcov[1, 1] + pcov[0, 0] 
                                     + 2 * true_angle * pcov[0, 1])
        self.enel = self.ekin - self.hnuminphi
                    
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

    
class MDCs():
    r"""
    Enel can currently be a list for plotting, but not yet for fitting.
    """
    def __init__(self, intensities, angles, angle_resolution, enel, 
                 hnuminphi):
        self.intensities = intensities
        self.angles = angles
        self.enel = enel
        self.hnuminphi = hnuminphi
        self.angle_resolution = angle_resolution

        self._enel_range= None
        self._individual_properties = None
        
    @property
    def enel(self):
        r"""
        """
        return self._enel

    @enel.setter
    def enel(self, x):
        r"""
        """
        self._enel = x

    @property
    def ekin(self):
        r"""
        """
        return self._enel + self.hnuminphi

    @ekin.setter
    def ekin(self, x):
        r"""
        """
        self._enel = x - self.hnuminphi

    @property
    def intensities(self):
        r"""
        """
        return self._intensities

    @intensities.setter
    def intensities(self, x):
        r"""
        """
        self._intensities = x

    @property
    def enel_range(self):
        if self._enel_range is None:
            raise AttributeError(
                "enel_range not yet set. Run `.fit_selection()` first."
            )
        return self._enel_range
    
    @property
    def individual_labels(self):
        if self._individual_labels is None:
            raise AttributeError(
                "individual_labels not yet set. Run `.fit_selection()` first."
            )
        return self._individual_labels
    
    @property
    def individual_properties(self):
        """
        Returns the list of per-slice, per-distribution labels and parameters.

        Returns
        -------
        list of list of dict
            Each element corresponds to one kinetic-energy slice and contains
            a list of dictionaries (one per distribution) with keys such as
            'amplitude', 'peak', 'broadening', etc., depending on the
            distribution type.
        """
        if not hasattr(self, "_individual_properties") or \
        self._individual_properties is None:
            raise AttributeError(
                "individual_properties not yet set. Run `.fit_selection()` or" \
                " the fitting routine first."
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
                ax.set_title(f"Energy slice: {energies[indices[idx]] * kilo:.3f} meV")

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
                    ax.set_title(f"Energy slice: {energies[indices[i]] * kilo:.3f} meV")
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
        ax.set_title(f"Energy slice: {(kinergy - self.hnuminphi) * kilo:.3f} meV")
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
        from .functions import construct_parameters, build_distributions, \
            residual

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
        all_individual_results = []    # List of (n_individuals, n_angles)
        all_individual_properties = [] # List of per-slice [ [label, {params}], ... ]
        individual_labels = None       # List of (n_individuals) labels

        # map class_name -> parameter names to extract
        param_spec = {
            'Constant': ('offset',),
            'Linear': ('offset', 'slope'),
            'SpectralLinear': ('amplitude', 'peak', 'broadening'),
            'SpectralQuadratic': ('amplitude', 'peak', 'broadening'),
        }

        for kinergy, intensity in zip(kinergies, intensities):
            if matrix_element is not None:
                parameters, element_names = construct_parameters(new_distributions, matrix_args)
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

            # Rebuild the *fitted* distributions from optimized params
            fitted_distributions = build_distributions(new_distributions, outcome.params)

            # If using a matrix element, extract slice-specific args from the fit
            if matrix_element is not None:
                new_matrix_args = {key: outcome.params[key].value for key in matrix_args}
            else:
                new_matrix_args = None

            # ---- Compute individual curves (smoothed, cropped) and final sum (no plotting here)
            from scipy.ndimage import gaussian_filter
            extend, step, numb = extend_function(self.angles, self.angle_resolution)

            total_result_ext = np.zeros_like(extend)
            indiv_rows = []   # (n_individuals, n_angles)
            indiv_params = [] # list of [label, {params}] per distribution
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
                cls = getattr(dist, 'class_name', None)
                wanted = param_spec.get(cls, ())
                pvals = {}
                for pname in wanted:
                    if pname in outcome.params:
                        pvals[pname] = outcome.params[pname].value
                    else:
                        pvals[pname] = getattr(dist, pname, None)
                pvals['_class'] = cls

                # combine label + parameter dict in same element
                indiv_params.append([label, pvals])

            # final (sum) curve, smoothed & cropped
            final_result_i = gaussian_filter(total_result_ext, sigma=step)[
                numb:-numb if numb else None
            ]
            final_result_i = np.asarray(final_result_i)

            # Residual for this slice
            residual_i = np.asarray(intensity) - final_result_i

            # Store per-slice results
            all_final_results.append(final_result_i)
            all_residuals.append(residual_i)
            all_individual_results.append(np.vstack(indiv_rows))
            all_individual_properties.append(indiv_params)

        # Set the enel_range variable
        self._enel_range = kinergies - self.hnuminphi
        self._individual_properties = all_individual_properties

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
            ax.relim()             # recompute data limits from all artists in the Axes
            ax.autoscale_view()    # apply autoscaling + axes.ymargin padding

        else:
            if energy_value is not None:
                _idx = int(np.abs(energies - energy_value).argmin())
                energies_sel = np.atleast_1d(energies[_idx])
            elif energy_range is not None:
                e_min, e_max = energy_range
                energies_sel = energies[(energies >= e_min) & (energies <= e_max)]
            else:
                energies_sel = energies

            # Number of slices must match
            n_slices = len(all_final_results)
            assert intensities.shape[0] == n_slices == len(all_residuals) == len(all_individual_results), (
                f"Mismatch: data {intensities.shape[0]}, fits {len(all_final_results)}, "
                f"residuals {len(all_residuals)}, individuals {len(all_individual_results)}."
            )
            n_individuals = all_individual_results[0].shape[0] if n_slices else 0

            fig.subplots_adjust(bottom=0.25)
            idx = 0

            # Initial draw (MDC + Individuals + Fit + Residual) at slice 0
            scatter = ax.scatter(self.angles, intensities[idx], label="Data")

            individual_lines = []
            if n_individuals:
                for j in range(n_individuals):
                    lab = individual_labels[j] if individual_labels and j < len(individual_labels) else f"Comp {j}"
                    ln, = ax.plot(self.angles, all_individual_results[idx][j], label=str(lab))
                    individual_lines.append(ln)

            result_line, = ax.plot(self.angles, all_final_results[idx], label="Fit")
            resid_scatter = ax.scatter(self.angles, all_residuals[idx], label="Residual")

            # Title + limits (use only the currently shown slice, scaled by plot_margin)
            ax.set_title(f"Energy slice: {energies_sel[idx] * kilo:.3f} meV")
            ax.relim()             # recompute data limits from all artists in the Axes
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
                    Yi = all_individual_results[i]  # (n_individuals, n_angles)
                    for j, ln in enumerate(individual_lines):
                        ln.set_ydata(Yi[j])

                # Update fit and residual
                result_line.set_ydata(all_final_results[i])
                resid_scatter.set_offsets(np.c_[self.angles, all_residuals[i]])

                ax.relim()             # recompute data limits from all artists in the Axes
                ax.autoscale_view()    # apply autoscaling + axes.ymargin padding

                # Update title and redraw
                ax.set_title(f"Energy slice: {energies_sel[i] * kilo:.3f} meV")
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
        ax.set_title(f"Energy slice: {(kinergy - self.hnuminphi) * kilo:.3f} meV")
        
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
    

    def fit_parameters(self, select_label):
        r"""
        Selects and returns the energy range together with the label
        and its corresponding parameter subsets from
        `self._individual_properties`.

        Parameters
        ----------
        select_label : str
            Label to look for among the fitted distributions.

        Returns
        -------
        tuple
            (self._enel_range, selected_label, selected_properties)

        Raises
        ------
        AttributeError
            If `_enel_range` or `_individual_properties` have not yet been set.
        ValueError
            If the given label is not found in any of the fitted distributions.
        """
        if self._enel_range is None:
            raise AttributeError(
                "enel_range not yet set. Run `.fit_selection()` first."
            )

        if not hasattr(self, "_individual_properties") or self._individual_properties is None:
            raise AttributeError(
                "individual_properties not yet set. Run `.fit_selection()` first."
            )

        # Collect all parameter dicts for the requested label
        selected_properties = []
        for slice_props in self._individual_properties:
            for label, params in slice_props:
                if label == select_label:
                    selected_properties.append(params)
                    break  # move to next slice once found

        if not selected_properties:
            # gather all labels for error reporting
            all_labels = [lbl for sl in self._individual_properties for lbl, _ in sl]
            raise ValueError(
                f"Label '{select_label}' not found in available labels: {sorted(set(all_labels))}"
            )

        return self._enel_range, select_label, selected_properties


class SelfEnergy():
    r"""Class for the self-energy.
    """
    def __init__(self, enel_range, label, properties):
        self.enel_range = enel_range
        self.label = label
        self.properties = properties

    @property
    def enel_range(self):
        r"""
        """
        return self._enel_range
    
    @enel_range.setter
    def enel_range(self, x):
        r"""
        """
        self._enel_range = x

    @property
    def label(self):
        r"""
        """
        return self._label
    
    @label.setter
    def label(self, x):
        r"""
        """
        self._label = x

    @property
    def properties(self):
        r"""
        """
        return self._properties
    
    @properties.setter
    def properties(self, x):
        r"""
        """
        self._properties = x