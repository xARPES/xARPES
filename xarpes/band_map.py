# Copyright (C) 2024 xARPES Developers
# This program is free software under the terms of the GNU GPLv3 license.

# get_ax_fig_plt and add_fig_kwargs originate from pymatgen/util/plotting.py.
# Copyright (C) 2011-2024 Shyue Ping Ong and the pymatgen Development Team
# Pymatgen is released under the MIT License.

# See also abipy/tools/plotting.py.
# Copyright (C) 2021 Matteo Giantomassi and the AbiPy Group
# AbiPy is free software under the terms of the GNU GPLv2 license.

"""The band map class and allowed operations on it."""

import igor2
import numpy as np
from .plotting import get_ax_fig_plt, add_fig_kwargs
from .functions import fit_leastsq, extend_function
from .distributions import fermi_dirac

# Physical constants
dtor = np.pi / 180 # Degrees to radians [rad/deg]
pref = 3.80998211616 # hbar^2/(2m_e) [eV Angstrom^2]

class MDCs():
    r"""
    Ebin is an attribute, not a parameter.
    """
    def __init__(self, intensities, angles, angle_resolution, ebin, hnuminphi):
        self.intensities = intensities
        self.angles = angles
        self.ebin = ebin
        self.ekin = ebin + hnuminphi
        self.angle_resolution = angle_resolution
        self.hnuminphi = hnuminphi

    @property
    def ebin(self):
        r"""
        """
        return self._ebin

    @ebin.setter
    def ebin(self, x):
        r"""
        """
        self._ebin = x

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
            energy_index = np.abs(self.ebin - energy_value).argmin()
            counts = self.intensities[energy_index, :]
        else:
            counts = self.intensities

        if energy_value is not None and len(np.shape(self.intensities)) == 1:
            print('There is only a single MDC. The provided energy value ' +
                  'will be ignored.')
        return counts

    @add_fig_kwargs
    def plot(self, energy_value=None, ax=None, **kwargs):
        r"""
        TBD
        """
        counts = self.energy_check(energy_value)

        ax, fig, plt = get_ax_fig_plt(ax=ax)

        ax.scatter(self.angles, counts, label='Data')

        ax.set_xlabel('Angle ($\degree$)')
        ax.set_ylabel('Counts (-)')

        ax.legend()

        return fig

    @add_fig_kwargs
    def visualize_guess(self, distributions, energy_value=None,
                        matrix_element=None, matrix_args=None, ax=None,
                        **kwargs):
        r"""
        """
        from scipy.ndimage import gaussian_filter

        counts = self.energy_check(energy_value)

        ax, fig, plt = get_ax_fig_plt(ax=ax)

        ax.set_xlabel('Angle ($\degree$)')
        ax.set_ylabel('Counts (-)')

        ax.scatter(self.angles, counts, label='Data')

        # Modify this when mdcs is a larger collection
        kinetic_energy = self.ekin

        extend, step, numb = extend_function(self.angles,
                                             self.angle_resolution)
        total_result = np.zeros(np.shape(extend))

        for dist in distributions:
            if dist.class_name == 'spectral_quadratic':
                if (dist.center_angle is not None) and (kinetic_energy is
                    None or hnuminphi is None):
                    raise ValueError('Spectral quadratic function is ' +
                    'defined in terms of a center angle. Please provide ' +
                    'a kinetic energy and hnuminphi.')
                extended_result = dist.evaluate(extend,
                                            kinetic_energy, self.hnuminphi)
            else:
                extended_result = dist.evaluate(extend)

            if matrix_element is not None and hasattr(dist, 'index'):
                extended_result *= matrix_element(extend, **matrix_args)

            total_result += extended_result

            individual_result = gaussian_filter(extended_result,
                                    sigma=step)[numb:-numb if numb else None]

            ax.plot(self.angles, individual_result, label=dist.label)

        final_result = gaussian_filter(total_result,
                                sigma=step)[numb:-numb if numb else None]

        ax.plot(self.angles, final_result, label='Distribution sum')

        residual = counts - final_result

        ax.scatter(self.angles, residual, label='Residual')
        ax.legend()

        return fig

    @add_fig_kwargs
    def fit(self, distributions, energy_value=None, matrix_element=None,
            matrix_args=None, ax=None, **kwargs):
        r"""
        """
        from .functions import construct_parameters, build_distributions, \
        residual
        from scipy.ndimage import gaussian_filter
        from lmfit import Minimizer
        import copy

        counts = self.energy_check(energy_value)

        ax, fig, plt = get_ax_fig_plt(ax=ax)

        ax.set_xlabel('Angle ($\degree$)')
        ax.set_ylabel('Counts (-)')

        ax.scatter(self.angles, counts, label='Data')

        # Modify this when mdcs is a larger collection
        kinetic_energy = self.ekin

        new_distributions = copy.deepcopy(distributions)

        if matrix_element is not None:
            parameters, element_names = construct_parameters(distributions,
                                                             matrix_args)
            new_distributions = build_distributions(new_distributions,
                                                    parameters)
            mini = Minimizer(residual, parameters, fcn_args=(self.angles,
            self.intensities, self.angle_resolution, new_distributions,
                    kinetic_energy, self.hnuminphi, matrix_element,
                                                             element_names))
        else:
            parameters = construct_parameters(distributions)
            new_distributions = build_distributions(new_distributions,
                                                    parameters)
            mini = Minimizer(residual, parameters, fcn_args=(self.angles,
            self.intensities, self.angle_resolution, new_distributions,
                                kinetic_energy, self.hnuminphi))

        outcome = mini.minimize('least_squares')
        pcov = outcome.covar

        if matrix_element is not None:
            new_matrix_args = {}
            for key in matrix_args:
                new_matrix_args[key] = outcome.params[key].value

        # The following until the residual statement could probably be combined
        # into a single method, such that visualize_guess() and fit() can both
        # call it.
        extend, step, numb = extend_function(self.angles,
                                             self.angle_resolution)

        total_result = np.zeros(np.shape(extend))

        for dist in new_distributions:
            if dist.class_name == 'spectral_quadratic':
                if (dist.center_angle is not None) and (kinetic_energy is
                    None or hnuminphi is None):
                    raise ValueError('Spectral quadratic function is ' +
                    'defined in terms of a center angle. Please provide ' +
                    'a kinetic energy and hnuminphi.')
                extended_result = dist.evaluate(extend,
                                            kinetic_energy, self.hnuminphi)
            else:
                extended_result = dist.evaluate(extend)

            if matrix_element is not None and hasattr(dist, 'index'):
                extended_result *= matrix_element(extend, **matrix_args)

            total_result += extended_result

            individual_result = gaussian_filter(extended_result,
                                    sigma=step)[numb:-numb if numb else None]

            ax.plot(self.angles, individual_result, label=dist.label)

        final_result = gaussian_filter(total_result,
                                sigma=step)[numb:-numb if numb else None]

        ax.plot(self.angles, final_result, label='Distribution sum')

        residual = counts - final_result

        ax.scatter(self.angles, residual, label='Residual')
        ax.legend()

        if matrix_element is not None:
            return fig, new_distributions, pcov, new_matrix_args
        else:
            return fig, new_distributions, pcov

class band_map():
    r"""Class for the band map from the ARPES experiment.

    Parameters
    ----------
    datafile : str
        Name of data file. Currently, only IGOR Binary Wave files are supported.
        If absent, `intensities`, `angles`, and `ekin` are mandatory. Otherwise,
        those arguments can be used to overwrite the contents of `datafile`.
    intensities : ndarray
        2D array of counts for given (E,k) or (E,angle) pairs [counts]
    angles : ndarray
        1D array of angle values for the abscissa [degrees]
    ekin : ndarray
        1D array of kinetic energy values for the ordinate [eV]
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
    """
    def __init__(self, datafile=None, intensities=None, angles=None, ekin=None,
                 ebin=None, energy_resolution=None, angle_resolution=None,
                 temperature=None, hnuminphi=None, hnuminphi_std=None,
                 transpose=False):

        if datafile is not None:
            data = igor2.binarywave.load(datafile)

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

        self.ebin = ebin
        self.energy_resolution = energy_resolution
        self.angle_resolution = angle_resolution
        self.temperature = temperature
        self.hnuminphi = hnuminphi
        self.hnuminphi_std = hnuminphi_std

    # Band map is still missing a whole lot of properties and setters

    @property
    def ebin(self):
        r"""
        """
        return self._ebin

    @ebin.setter
    def ebin(self, x):
        r"""
        """
        self._ebin = x

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
        r"""Manually sets the photon energy minus the work function in eV if it
        has been set; otherwise returns None.

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

    def slicing(self, angle_min, angle_max, energy_value=None,
                energy_range=None):
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

        if (energy_value is None and energy_range is None) or \
        (energy_value is not None and energy_range is not None):
            raise ValueError('Please provide either energy_value or ' +
            'energy_range.')

        angle_min_index = np.abs(self.angles - angle_min).argmin()
        angle_max_index = np.abs(self.angles - angle_max).argmin()
        angle_range_out = self.angles[angle_min_index:angle_max_index + 1]

        if energy_value is not None:
            energy_index = np.abs(self.ebin - energy_value).argmin()
            binding_range_out = self.ebin[energy_index]
            mdcs = self.intensities[energy_index,
                   angle_min_index:angle_max_index + 1]

        if energy_range:
            energy_indices = np.where((self.ebin >= np.min(energy_range))
                                      & (self.ebin <= np.max(energy_range)))[0]
            binding_range_out = self.ebin[energy_indices]
            mdcs = self.intensities[energy_indices,
                                    angle_min_index:angle_max_index + 1]

        return mdcs, angle_range_out, self.angle_resolution, \
        binding_range_out, self.hnuminphi

    @add_fig_kwargs
    def plot(self, abscissa='momentum', ordinate='binding_energy', ax=None,
             **kwargs):
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

        if abscissa == 'angle':
            ax.set_xlabel('Angle ($\degree$)')
            if ordinate == 'kinetic_energy':
                mesh = ax.pcolormesh(Angl, Ekin, self.intensities,
                       shading='auto', cmap=plt.get_cmap('bone').reversed())
                ax.set_ylabel('$E_{\mathrm{kin}}$ (eV)')
            elif ordinate == 'binding_energy':
                Ebin = Ekin - self.hnuminphi
                mesh = ax.pcolormesh(Angl, Ebin, self.intensities,
                       shading='auto', cmap=plt.get_cmap('bone').reversed())
                ax.set_ylabel('$E_{\mathrm{bin}}$ (eV)')
        elif abscissa == 'momentum':
            Mome = np.sqrt(Ekin / pref) * np.sin(Angl * dtor)
            ax.set_xlabel(r'$k_{//}$ ($\mathrm{\AA}^{\endash1}$)')
            if ordinate == 'kinetic_energy':
                mesh = ax.pcolormesh(Mome, Ekin, self.intensities,
                       shading='auto', cmap=plt.get_cmap('bone').reversed())
                ax.set_ylabel('$E_{\mathrm{kin}}$ (eV)')
            elif ordinate == 'binding_energy':
                Ebin = Ekin - self.hnuminphi
                mesh = ax.pcolormesh(Mome, Ebin, self.intensities,
                       shading='auto', cmap=plt.get_cmap('bone').reversed())
                ax.set_ylabel('$E_{\mathrm{bin}}$ (eV)')

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
            Additional arguments passed on to add_fig_kwargs. See the keyword
            table below.

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

        fdir_initial = fermi_dirac(temperature=self.temperature,
                                   hnuminphi=hnuminphi_guess,
                                   background=background_guess,
                                   integrated_weight=integrated_weight_guess,
                                   name='Initial guess')

        parameters = np.array(
            [hnuminphi_guess, background_guess, integrated_weight_guess])

        extra_args = (self.temperature)

        popt, pcov = fit_leastsq(parameters, energy_range,
                     integrated_intensity, fdir_initial,
                     self.energy_resolution, extra_args)

        fdir_final = fermi_dirac(temperature=self.temperature,
                                 hnuminphi=popt[0], background=popt[1],
                                 integrated_weight=popt[2],
                                 name='Fitted result')

        self.hnuminphi = popt[0]
        self.hnuminphi_std = np.sqrt(np.diag(pcov))[0][0]
        self.ebin = self.ekin - self.hnuminphi

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
