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

from .plotting import get_ax_fig_plt, add_fig_kwargs
from .functions import fit_leastsq
from .distributions import fermi_dirac


class MDCs():
    r"""
    """
    def __init__(self, intensities, angles, ekin, angular_resolution):
        self.intensities = intensities
        self.angles = angles
        self.ekin = ekin
        self.angular_resolution = angular_resolution

    # @add_fig_kwargs
    # def fit(self):
    #     r"""
    #     """
    #     return 0

 
    # @add_fig_kwargs
    # def fit_fermi_edge(self, hnuminphi_guess, background_guess=0.0,
    #                    integrated_weight_guess=1.0, angle_min=-np.infty,
    #                    angle_max=np.infty, ekin_min=-np.infty,
    #                    ekin_max=np.infty, ax=None, **kwargs):
    
    
    @add_fig_kwargs
    def plot(self, ax=None, **kwargs):
        r"""
        """
        ax, fig, plt = get_ax_fig_plt(ax=ax)

        ax.scatter(self.angles, self.intensities)
        
        ax.set_xlabel("Angle ($\degree$)")
        ax.set_ylabel('Counts (-)')
        
        return fig
        

class band_map():
    r"""Class for the band map from the ARPES experiment.

    Parameters
    ----------
    intensities : ndarray
        2D array of counts for given (E,k) or (E,angle) pairs [counts]
    angles : ndarray
        1D array of angular values for the abscissa [degrees]
    ekin : ndarray
        1D array of kinetic energy values for the ordinate [eV]
    angular_resolution : float, None
        Angular resolution of the detector [degrees]
    energy_resolution : float, None
        Energy resolution of the detector [eV]
    temperature : float, None
        Temperature of the sample [K]
    hnuminphi : float, None
        Kinetic energy minus the work function [eV]
    hnuminphi_std : float, None
        Standard deviation of kinetic energy minus work function [eV]
    """
    def __init__(self, intensities, angles, ekin, energy_resolution=None,
                 angular_resolution=None, temperature=None, hnuminphi=None,
                 hnuminphi_std=None):
        self.intensities = intensities
        self.angles = angles
        self.ekin = ekin
        self.energy_resolution = energy_resolution
        self.angular_resolution = angular_resolution
        self.temperature = temperature
        self.hnuminphi = hnuminphi
        self.hnuminphi_std = hnuminphi_std

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
    def hnuminphi(self, hnuminphi):
        r"""Manually sets the photon energy minus the work function in eV if it
        has been set; otherwise returns None.

        Parameters
        ----------
        hnuminphi : float, None
            Kinetic energy minus the work function [eV]
        """
        self._hnuminphi = hnuminphi

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
    def hnuminphi_std(self, hnuminphi_std):
        r"""Manually sets the standard deviation of photon energy minus the
        work function in eV.

        Parameters
        ----------
        hnuminphi_std : float
            Standard deviation of energy minus the work function [eV]
        """
        self._hnuminphi_std = hnuminphi_std

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

    def slice(self, angle_min, angle_max, energy_value):
        r"""
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
            Array of size nxm containing the MDC intensities
        """
        
        energy_index = np.abs(self.ekin-energy_value).argmin()
        angle_min_index = np.abs(self.angles-angle_min).argmin() 
        angle_max_index = np.abs(self.angles-angle_max).argmin()
        
        angle_range = self.angles[angle_min_index:angle_max_index+1]
        energy_range = self.ekin[energy_index]
        mdcs = self.intensities[energy_index, angle_min_index:angle_max_index+1]

        return mdcs, angle_range, energy_range, self.angular_resolution

    
    @add_fig_kwargs
    def plot_band_map(self, ax=None, **kwargs):
        r"""Plots the band map.
        
        Parameters
        ----------

        Other parameters
        ----------------
        **kwargs : dict, optional
            Additional arguments passed on to add_fig_kwargs. See the keyword
            table below.      

        Returns
        -------
        fig : Matplotlib-Figure
            Figure containing the Fermi edge fit
        """        
        ax, fig, plt = get_ax_fig_plt(ax=ax)

        Angl, Ekin = np.meshgrid(self.angles, self.ekin)
        mesh = ax.pcolormesh(Angl, Ekin, self.intensities, shading="auto", cmap=plt.get_cmap("bone").reversed())
        cbar = plt.colorbar(mesh, ax=ax)
        
        ax.set_xlabel("Angle ($\degree$)")
        ax.set_ylabel("$E_{\mathrm{kin}}$ (eV)")
        
        return fig
    
    @add_fig_kwargs
    def fit_fermi_edge(self, hnuminphi_guess, background_guess=0.0,
                       integrated_weight_guess=1.0, angle_min=-np.infty,
                       angle_max=np.infty, ekin_min=-np.infty,
                       ekin_max=np.infty, ax=None, **kwargs):
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
            Additional arguments passed on to add_fig_kwargs. See the keyword
            table below.

        Returns
        -------
        fig : Matplotlib-Figure
            Figure containing the Fermi edge fit
        """
        ax, fig, plt = get_ax_fig_plt(ax=ax)

        min_angle_index = np.argmin(np.abs(self.angles - angle_min))
        max_angle_index = np.argmin(np.abs(self.angles - angle_max))

        min_ekin_index = np.argmin(np.abs(self.ekin - ekin_min))
        max_ekin_index = np.argmin(np.abs(self.ekin - ekin_max))

        energy_range = self.ekin[min_ekin_index:max_ekin_index]

        integrated_intensity = np.trapz(
            self.intensities[min_ekin_index:max_ekin_index,
                min_angle_index:max_angle_index], axis=1)

        fdir_initial = fermi_dirac(temperature=self.temperature,
                                   hnuminphi=hnuminphi_guess,
                                   background=background_guess,
                                   integrated_weight=integrated_weight_guess,
                                   name='Initial guess')

        parameters = np.array(
            [hnuminphi_guess, background_guess, integrated_weight_guess])

        extra_args = (self.energy_resolution)

        popt, pcov = fit_leastsq(parameters, energy_range, integrated_intensity,
                                 fdir_initial, extra_args)

        fdir_final = fermi_dirac(temperature=self.temperature,
                                 hnuminphi=popt[0], background=popt[1],
                                 integrated_weight=popt[2],
                                 name='Fitted result')

        self.hnuminphi = popt[0]
        self.hnuminphi_std = np.sqrt(np.diag(pcov))[0][0]

        ax.set_xlabel(r'$E_{\mathrm{kin}}$ (-)')
        ax.set_ylabel('Counts (-)')
        ax.set_xlim([ekin_min, ekin_max])

        ax.plot(energy_range, integrated_intensity, label='Data')

        ax.plot(energy_range, fdir_initial.convolve(energy_range,
                energy_resolution=self.energy_resolution),
                label=fdir_initial.name)

        ax.plot(energy_range, fdir_final.convolve(energy_range,
                energy_resolution=self.energy_resolution),
                label=fdir_final.name)

        ax.legend()

        return fig