# Copyright (C) 2024 xARPES Developers
# This program is free software under the terms of the GNU GPLv2 license.

# get_ax_fig_plt and add_fig_kwargs originate from pymatgen/util/plotting.py.
# Copyright (C) 2011-2024 Shyue Ping Ong and the pymatgen Development Team
# Pymatgen is released under the MIT License.

# See also abipy/tools/plotting.py.
# Copyright (C) 2021 Matteo Giantomassi and the AbiPy Group
# AbiPy is free software under the terms of the GNU GPLv2 license.

"""The band map class and allowed operations on it."""

import numpy as np
from .plotting import get_ax_fig_plt, add_fig_kwargs
from .distributions import fermi_dirac



class ExampleError(Exception):
    """Exceptions are documented in the same way as classes.

    The __init__ method may be documented in either the class level
    docstring, or as a docstring on the __init__ method itself.

    Either form is acceptable, but the two should not be mixed. Choose one
    convention to document the __init__ method and be consistent with it.

    Note
    ----
    Do not include the `self` parameter in the ``Parameters`` section.

    Parameters
    ----------
    msg : str
        Human readable string describing the exception.
    code : :obj:`int`, optional
        Numeric error code.

    Attributes
    ----------
    msg : str
        Human readable string describing the exception.
    code : int
        Numeric error code.

    """



class band_map():
    r"""Class for the band map from the ARPES experiment.

    Parameters
    ----------
    intensities : ndarray
        2D array of counts for given (E,k) or (E,angle) pairs [counts]
    angles : ndarray
        1D array of angular values for the abscissa [degrees]
    ekin : ndarray
        1D array of kinetic energy values for the ordinate. [eV]
    energy_resolution : float
        Energy resolution of the detector [eV]
    temperature : float
        Temperature of the sample [K]
    hnuminphi : float
        Kinetic energy minus the work function [eV]


    Attributes
    ----------
    module_level_variable1 : int
        Module level variables may be documented in either the ``Attributes``

       
    """
    def __init__(self, intensities, angles, ekin, energy_resolution=None,
                 temperature=None, hnuminphi=None):

        self.intensities = intensities
        self.angles = angles
        self.ekin = ekin
        self.energy_resolution = energy_resolution
        self.temperature = temperature
        self.hnuminphi = hnuminphi
    
    @property
    def hnuminphi(self):
        r"""Returns the the photon energy minus the work function in eV.

        Returns
        -------
        hnuminphi : float 
            Kinetic energy minus the work function [eV]
        """
        return self._hnuminphi

    @hnuminphi.setter
    def hnuminphi(self, hnuminphi):
        r"""Manually sets the photon energy minus the work function in eV.

        Parameters
        ----------
        hnuminphi : float
            Kinetic energy minus the work function [eV]
        """
        self._hnuminphi = hnuminphi

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

    @add_fig_kwargs
    def fit_fermi_edge(self, hnuminphi_guess, background_guess=0.0,
                       integrated_weight_guess=1.0, angle_min=-np.infty,
                       angle_max=np.infty, ekin_min=-np.infty,
                       ekin_max=np.infty, ax=None, **kwargs):
        r"""
        Fits the fermi edge of the band map and plots the result.

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
        **kwargs : dict, optional
            Additional arguments passed on to add_fig_kwargs. See the keyword
            table below.
            
        Returns
        -------
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