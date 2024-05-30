# Copyright (C) 2024 xARPES Developers
# This program is free software under the terms of the GNU GPLv2 license.

"""The band map class and allowed operations on it."""

import numpy as np
from .plotting import get_ax_fig_plt, add_fig_kwargs
from .distributions import fermi_dirac

class band_map():
    r"""Class for the band map from the ARPES experiment.

    Parameters
    ----------
    intensities : ndarray
        2D array of counts for given (E,k) or (E,angle) pairs.
    angles : ndarray
        1D array of angular values for the abscissa.
    ekin : ndarray
        1D array of kinetic energy values for the ordinate.
    energy_resolution : float
        Energy resolution of the detector [eV]
    temperature : float
        Temperature of the sample [K]
    """
    def __init__(self, intensities, angles, ekin, energy_resolution=None,
                 temperature=None, hnuminphi=None):

        self.intensities = intensities
        self.angles = angles
        self.ekin = ekin
        self.energy_resolution = energy_resolution
        self.temperature = temperature
        self.hnuminphi = hnuminphi
        print(type(self.energy_resolution))
    
    @property
    def hnuminphi(self):
        r"""
        Returns the chemical potential corresponding to a particular kinetic
        energy.
        """
        return self._hnuminphi

    @hnuminphi.setter
    def hnuminphi(self, hnuminphi):
        r"""
        Sets the chemical potential corresponding to a particular kinetic
        energy.
        """
        self._hnuminphi = hnuminphi

    def shift_angles(self, shift):
        r"""
        Shifts the angles by the specified amount. Used to align with respect to
        experimentally observed angles.
        """
        self.angles = self.angles + shift

    @add_fig_kwargs
    def fit_fermi_edge(self, hnuminphi_guess, background_guess=0,
                       integrated_weight_guess=1, angle_min=-np.infty,
                       angle_max=np.infty, ekin_min=-np.infty,
                       ekin_max=np.infty, ax=None, **kwargs):
        r"""
        Plots the band map.

        Returns
        ----------
        Matplotlib-Figure
        """
        from xarpes.functions import fit_leastsq

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