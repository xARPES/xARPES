# Copyright (C) 2024 xARPES Developers
# This program is free software under the terms of the GNU GPLv3 license.

# get_ax_fig_plt and add_fig_kwargs originate from pymatgen/util/plotting.py.
# Copyright (C) 2011-2024 Shyue Ping Ong and the pymatgen Development Team
# Pymatgen is released under the MIT License.

# See also abipy/tools/plotting.py.
# Copyright (C) 2021 Matteo Giantomassi and the AbiPy Group
# AbiPy is free software under the terms of the GNU GPLv2 license.

"""The band map class and allowed operations on it."""

import numpy as np
from igor2 import packed, binarywave
from .plotting import get_ax_fig_plt, add_fig_kwargs
from .functions import fit_leastsq, extend_function
from .distributions import FermiDirac, Linear

# Physical constants
# The first one can be generated with scipy.stats.norm.ppf(0.975)
uncr = 1.95996398 # Standard deviation to 95 % confidence [-]
dtor = np.pi / 180 # Degrees to radians [rad/deg]
pref = 3.80998211616 # hbar^2/(2m_e) [eV Angstrom^2]

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

    # Band map is still missing a whole lot of properties and setters

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

    def energy_check(self, energy_value):
        r"""
        """
        if energy_value is None and len(np.shape(self.intensities)) > 1:
            raise ValueError('Multiple MDCs. Please provide the nearest ' +
                         'energy value for which to plot an MDC.')

        if energy_value is not None and len(np.shape(self.intensities)) > 1:
            energy_index = np.abs(self.enel - energy_value).argmin()
            counts = self.intensities[energy_index, :]
        else:
            counts = self.intensities

        return counts


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
                "Provide either energy_value or energy_range, not both.")

        ax, fig, plt = get_ax_fig_plt(ax=ax)

        angles = self.angles
        energies = self.enel
        intensities = self.intensities
        existing_ymin, existing_ymax = ax.get_ylim()

        if np.isscalar(energies):
            if energy_value is not None or energy_range is not None:
                raise ValueError(
                    "This dataset contains only one energy slice; do not "
                    "provide energy_value or energy_range."
                )
            ydata = intensities
            ax.scatter(angles, ydata, label="Data")
            ax.set_title(f"Energy slice: {energies:.3f} eV")
            combined_ymin = min(existing_ymin, ydata.min())
            combined_ymax = max(existing_ymax, ydata.max())
            ax.set_ylim(combined_ymin, combined_ymax)

        else:
            if energy_value is not None:
                if (energy_value < energies.min() or
                        energy_value > energies.max()):
                    raise ValueError(
                        f"Requested energy_value {energy_value:.3f} eV is "
                        f"outside the available energy range "
                        f"[{energies.min():.3f}, {energies.max():.3f}] eV."
                    )
                idx = np.abs(energies - energy_value).argmin()
                ydata = intensities[idx]
                ax.scatter(angles, ydata, label="Data")
                ax.set_title(f"Energy slice: {energies[idx]:.3f} eV")
                combined_ymin = min(existing_ymin, ydata.min())
                combined_ymax = max(existing_ymax, ydata.max())
                ax.set_ylim(combined_ymin, combined_ymax)

            elif energy_range is not None:
                e_min, e_max = energy_range
                mask = (energies >= e_min) & (energies <= e_max)
                indices = np.where(mask)[0]
                if len(indices) == 0:
                    raise ValueError(
                        "No energies found in the specified energy_range."
                    )

                ydata = intensities[indices]

                fig.subplots_adjust(bottom=0.25)
                idx = 0
                scatter = ax.scatter(angles, ydata[idx], label="Data")
                ax.set_title(
                    f"Energy slice: {energies[indices[idx]]:.3f} eV"
                )
                combined_ymin = min(existing_ymin, ydata.min())
                combined_ymax = max(existing_ymax, ydata.max())
                ax.set_ylim(combined_ymin, combined_ymax)

                slider_ax = fig.add_axes([0.2, 0.08, 0.6, 0.04])
                slider = Slider(
                    slider_ax, "Energy index", 0, len(indices) - 1,
                    valinit=idx, valstep=1
                )

                def update(val):
                    i = int(slider.val)
                    scatter.set_offsets(np.c_[angles, ydata[i]])
                    ax.set_title(
                        f"Energy slice: {energies[indices[i]]:.3f} eV"
                    )
                    fig.canvas.draw_idle()

                slider.on_changed(update)
                self._slider = slider
                self._line = scatter

            else:
                e_min, e_max = energies[0], energies[-1]
                mask = (energies >= e_min) & (energies <= e_max)
                indices = np.where(mask)[0]
                if len(indices) == 0:
                    raise ValueError("No valid energy slices in the dataset.")

                ydata = intensities[indices]

                fig.subplots_adjust(bottom=0.25)
                idx = 0
                scatter = ax.scatter(angles, ydata[idx], label="Data")
                ax.set_title(
                    f"Energy slice: {energies[indices[idx]]:.3f} eV"
                )
                combined_ymin = min(existing_ymin, ydata.min())
                combined_ymax = max(existing_ymax, ydata.max())
                ax.set_ylim(combined_ymin, combined_ymax)

                slider_ax = fig.add_axes([0.2, 0.08, 0.6, 0.04])
                slider = Slider(
                    slider_ax, "Energy index", 0, len(indices) - 1,
                    valinit=idx, valstep=1
                )

                def update(val):
                    i = int(slider.val)
                    scatter.set_offsets(np.c_[angles, ydata[i]])
                    ax.set_title(
                        f"Energy slice: {energies[indices[i]]:.3f} eV"
                    )
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
                axis.annotate(
                    f"({tags[i]})", xy=(0.05, 0.95),
                    xycoords="axes fraction"
                )

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
        counts = self.energy_check(energy_value)
        ax, fig, plt = get_ax_fig_plt(ax=ax)

        ax.set_xlabel('Angle ($\\degree$)')
        ax.set_ylabel('Counts (-)')
        ax.scatter(self.angles, counts, label='Data')

        kinetic_energy = self.ekin

        final_result = self._merge_and_plot(
            ax=ax,
            distributions=distributions,
            kinetic_energy=kinetic_energy,
            matrix_element=matrix_element,
            matrix_args=dict(matrix_args) if matrix_args else None,
            plot_individual=True,
        )

        residual = counts - final_result
        ax.scatter(self.angles, residual, label='Residual')
        ax.legend()

        return fig
    

    @add_fig_kwargs
    def fit(self, distributions, energy_value=None, matrix_element=None,
            matrix_args=None, ax=None, **kwargs):
        r"""
        """
        import copy
        from lmfit import Minimizer
        from .functions import construct_parameters, build_distributions, \
            residual

        counts = self.energy_check(energy_value)
        ax, fig, plt = get_ax_fig_plt(ax=ax)

        ax.set_xlabel('Angle ($\\degree$)')
        ax.set_ylabel('Counts (-)')
        ax.scatter(self.angles, counts, label='Data')

        kinetic_energy = self.ekin
        new_distributions = copy.deepcopy(distributions)

        if matrix_element is not None:
            parameters, element_names = construct_parameters(distributions, \
                                                             matrix_args)
            new_distributions = build_distributions(new_distributions, \
                                                    parameters)
            mini = Minimizer(
                residual, parameters,
                fcn_args=(self.angles, self.intensities, self.angle_resolution,
                          new_distributions, kinetic_energy, self.hnuminphi,
                          matrix_element, element_names)
            )
        else:
            parameters = construct_parameters(distributions)
            new_distributions = build_distributions(new_distributions, \
                                                    parameters)
            mini = Minimizer(
                residual, parameters,
                fcn_args=(self.angles, self.intensities, self.angle_resolution,
                          new_distributions, kinetic_energy, self.hnuminphi)
            )

        outcome = mini.minimize('least_squares')
        pcov = outcome.covar

        # If matrix params were fitted, pass the fitted values to plotting
        if matrix_element is not None:
            new_matrix_args = {key: outcome.params[key].value for key in \
                               matrix_args}
        else:
            new_matrix_args = None

        final_result = self._merge_and_plot(
            ax=ax,
            distributions=new_distributions,
            kinetic_energy=kinetic_energy,
            matrix_element=matrix_element,
            matrix_args=new_matrix_args,
            plot_individual=True,
        )

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

            if plot_individual:
                individual = gaussian_filter(extended_result, sigma=step)\
                    [numb:-numb if numb else None]
                ax.plot(self.angles, individual, label=getattr(dist, \
                                                        'label', str(dist)))

        # Smoothed, cropped total curve aligned to self.angles
        final_result = gaussian_filter(total_result, sigma=step)[numb:-numb \
                                                            if numb else None]
        ax.plot(self.angles, final_result, label='Distribution sum')

        return final_result