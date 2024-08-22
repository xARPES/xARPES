# Copyright (C) 2024 xARPES Developers
# This program is free software under the terms of the GNU GPLv3 license.

"""The distributions used throughout the code."""

import numpy as np
from .plotting import get_ax_fig_plt, add_fig_kwargs

# Physical constants
k_B = 8.617e-5 # Boltzmann constant [eV/K]
dtor = np.pi / 180 # Degrees to radians [rad/deg]
pref = 3.80998211616 # hbar^2/(2m_e) [eV Angstrom^2]
fwhm_to_std = np.sqrt(8 * np.log(2)) # Convert FWHM to std [-]
sigma_extend = 5 # Extend data range by "5 sigma"

class CreateDistributions:
    r"""
    """
    def __init__(self, distributions):
        # Adds a call method to the distribution list
        self.distributions = distributions

    def __call__(self):
        return self.distributions

    @property
    def distributions(self):
        r"""
        """
        return self._distributions

    @distributions.setter
    def distributions(self, x):
        r"""
        """
        self._distributions = x

    def __iter__(self):
        r"""
        """
        return iter(self.distributions)

    def __deepcopy__(self, memo):
        r"""
        """
        import copy
        return type(self)(copy.deepcopy(self.distributions, memo))

    def __getitem__(self, index):
        r"""
        """
        return self.distributions[index]

    @add_fig_kwargs
    def plot(self, angle_range, angle_resolution, kinetic_energy=None,
             hnuminphi=None, matrix_element=None, matrix_args=None, ax=None,
             **kwargs):
        r"""
        """
        if angle_resolution < 0:
            raise ValueError('Distributions cannot be plotted with negative '
                            + 'resolution.')

        from scipy.ndimage import gaussian_filter

        ax, fig, plt = get_ax_fig_plt(ax=ax)

        ax.set_xlabel('Angle ($\degree$)')
        ax.set_ylabel('Counts (-)')

        total_result = np.zeros(np.shape(extend))

        for dist in self.distributions:
            if dist.class_name == 'SpectralQuadratic':
                if (dist.center_angle is not None) and (kinetic_energy is
                    None or hnuminphi is None):
                    raise ValueError('Spectral quadratic function is ' +
                    'defined in terms of a center angle. Please provide ' +
                    'a kinetic energy and hnuminphi.')
                extended_result = dist.evaluate(extend,
                                            kinetic_energy, hnuminphi)
            else:
                extended_result = dist.evaluate(extend)

            if matrix_element is not None:
                extended_result *= matrix_element(extend, **matrix_args)

            total_result += extended_result

            individual_result = gaussian_filter(extended_result, sigma=step
                                               )[numb:-numb if numb else None]
            ax.plot(angle_range, individual_result, label=dist.label)

        final_result = gaussian_filter(total_result, sigma=step
                                               )[numb:-numb if numb else None]

        ax.plot(angle_range, final_result, label=str('Distribution sum'))

        ax.legend()

        return fig

class Distribution:
    r"""Parent class for distributions. The class cannot be used on its own,
    but is used to instantiate unique and non-unique distributions.

    Parameters
    ----------
    name : str
        Non-unique name for instances, not to be modified after instantiation.
    """
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        r"""
        """
        return self._name

    def extend_range(self, abscissa_range, abscissa_resolution):
        r"""
        """
        step_size = np.abs(abscissa_range[1] - abscissa_range[0])
        step = abscissa_resolution / (step_size * fwhm_to_std)
        numb = int(sigma_extend * step)
        extend = np.linspace(abscissa_range[0] - numb * step_size,
                             abscissa_range[-1] + numb * step_size,
                             len(abscissa_range) + 2 * numb)
        return extend, step, numb

    @property
    def class_name(self):
        r"""TBD
        """
        return self.__class__.__name__

    @add_fig_kwargs
    def plot(self, angle_range, angle_resolution, kinetic_energy=None,
             hnuminphi=None, matrix_element=None, matrix_args=None,
             ax=None, **kwargs):
        r"""Overwritten for FermiDirac distribution.
        """
        if angle_resolution < 0:
            raise ValueError('Distribution cannot be plotted with negative '
                            + 'resolution.')
        from scipy.ndimage import gaussian_filter

        ax, fig, plt = get_ax_fig_plt(ax=ax)

        ax.set_xlabel('Angle ($\degree$)')
        ax.set_ylabel('Counts (-)')

        extend, step, numb = self.extend_range(angle_range, angle_resolution)

        if self.class_name == 'SpectralQuadratic':
            extended_result = self.evaluate(extend, kinetic_energy, hnuminphi)
        else:
            extended_result = self.evaluate(extend)

        if matrix_element is not None:
            extended_result *= matrix_element(extend, **matrix_args)

        final_result = gaussian_filter(extended_result, sigma=step)\
        [numb:-numb if numb else None]

        ax.plot(angle_range, final_result, label=self.label)

        ax.legend()

        return fig

class UniqueDistribution(Distribution):
    r"""Parent class for unique distributions, to be used one at a time, e.g.,
    during the background of an MDC fit or the Fermi-Dirac distribution.

    Parameters
    ----------
    label : str
        Unique label for instances, identical to the name for unique
        distributions. Not to be modified after instantiation.
    """
    def __init__(self, name):
        super().__init__(name)
        self._label = name

    @property
    def label(self):
        r"""Returns the unique class label.

        Returns
        -------
        label : str
            Unique label for instances, identical to the name for unique
            distributions. Not to be modified after instantiation.
        """
        return self._label

class FermiDirac(UniqueDistribution):
    r"""Child class for Fermi-Dirac (FD) distributions, used e.g., during Fermi
    edge fitting. The FD class is unique, only one instance should be used
    per task.

    The Fermi-Dirac distribution is described by the following formula:

    .. math::

            \frac{A}{\rm{e}^{\beta(E_{\rm{kin}}-(h\nu-\Phi))}+1} + B

    with :math:`A` as :attr:`integrated_weight`, :math:`B` as
    :attr:`background`, :math:`h\nu-\Phi` as :attr:`hnuminphi`, and
    :math:`\beta=1/(k_{\rm{B}}T)` with :math:`T` as :attr:`temperature`.

    Parameters
    ----------
    temperature : float
        Temperature of the sample [K]
    hnuminphi : float
        Kinetic energy minus the work function [eV]
    background : float
        Background spectral weight [counts]
    integrated_weight : float
        Integrated weight on top of the background [counts]
    """
    def __init__(self, temperature, hnuminphi, background=0,
                 integrated_weight=1, name='FermiDirac'):
        super().__init__(name)
        self.temperature = temperature
        self.hnuminphi = hnuminphi
        self.background = background
        self.integrated_weight = integrated_weight

        @property
        def temperature(self):
            r"""Returns the temperature of the sample.

            Returns
            -------
            temperature : float
                Temperature of the sample [K]
            """
            return self._temperature

        @temperature.setter
        def temperature(self, x):
            r"""Sets the temperature of the FD distribution.

            Parameters
            ----------
            temperature : float
                Temperature of the sample [K]
            """
            self._temperature = x

        @property
        def hnuminphi(self):
            r"""Returns the photon energy minus the work function of the FD
            distribution.

            Returns
            -------
            hnuminphi: float
                Kinetic energy minus the work function [eV]
            """
            return self._hnuminphi

        @hnuminphi.setter
        def hnuminphi(self, x):
            r"""Sets the photon energy minus the work function of the FD
            distribution.

            Parameters
            ----------
            hnuminphi : float
                Kinetic energy minus the work function [eV]
            """
            self._hnuminphi = x

        @property
        def background(self):
            r"""Returns the background intensity of the FD distribution.

            Returns
            -------
            background : float
                Background spectral weight [counts]
            """
            return self._background

        @background.setter
        def background(self, x):
            r"""Sets the background intensity of the FD distribution.

            Parameters
            ----------
            background : float
                Background spectral weight [counts]
            """
            self._background = x

        @property
        def integrated_weight(self):
            r"""Returns the integrated weight of the FD distribution.

            Returns
            -------
            integrated_weight: float
                Integrated weight on top of the background [counts]
            """
            return self._integrated_weight

        @integrated_weight.setter
        def integrated_weight(self, x):
            r"""Sets the integrated weight of the FD distribution.

            Parameters
            ----------
            integrated_weight : float
                Integrated weight on top of the background [counts]
            """
            self._integrated_weight = x

    def __call__(self, energy_range, hnuminphi, background, integrated_weight,
                 temperature):
        """Call method to directly evaluate a FD distribution without having to
        instantiate a class instance.

        Parameters
        ----------
        energy_range : ndarray
            1D array on which to evaluate the FD distribution [eV]
        hnuminphi : float
            Kinetic energy minus the work function [eV]
        background : float
            Background spectral weight [counts]
        integrated_weight : float
            Integrated weight on top of the background [counts]
        energy_resolution : float
            Energy resolution of the detector for the convolution [eV]

        Returns
        -------
        evalf : ndarray
            1D array of the energy-convolved FD distribution [counts]
        """
        k_BT = temperature * k_B

        return (integrated_weight / (1 + np.exp((energy_range - hnuminphi)
               / k_BT)) + background)

    def evaluate(self, energy_range):
        r"""Evaluates the FD distribution for a given class instance.
        No energy convolution is performed with evaluate.

        Parameters
        ----------
        energy_range : ndarray
            1D array on which to evaluate the FD distribution [eV]

        Returns
        -------
        evalf : ndarray
            1D array of the evaluated FD distribution [counts]
        """
        k_BT = self.temperature * k_B

        return (self.integrated_weight
            / (1 + np.exp((energy_range - self.hnuminphi) / k_BT))
            + self.background)

    @add_fig_kwargs
    def plot(self, energy_range, energy_resolution, ax=None, **kwargs):
        r"""TBD
        """
        if energy_resolution < 0:
            raise ValueError('Distribution cannot be plotted with negative '
                            + 'resolution.')
        from scipy.ndimage import gaussian_filter

        ax, fig, plt = get_ax_fig_plt(ax=ax)

        ax.set_xlabel(r'$E_{\mathrm{kin}}$ (-)')
        ax.set_ylabel('Counts (-)')

        extend, step, numb = self.extend_range(energy_range, energy_resolution)

        extended_result = self.evaluate(extend)

        final_result = gaussian_filter(extended_result, sigma=step)\
        [numb:-numb if numb else None]

        ax.plot(energy_range, final_result, label=self.label)

        ax.legend()

        return fig

class Constant(UniqueDistribution):
    r"""Child class for constant distributions, used e.g., during MDC fitting.
    The constant class is unique, only one instance should be used per task.

    Parameters
    ----------
    offset : float
        The value of the distribution for the abscissa equal to 0.
    """
    def __init__(self, offset, name='Constant'):
        super().__init__(name)
        self.offset = offset

    @property
    def offset(self):
        r"""Returns the offset of the constant distribution.

        Returns
        -------
        offset : float
            The value of the distribution for the abscissa equal to 0.
        """
        return self._offset

    @offset.setter
    def offset(self, x):
        r"""Sets the offset of the constant distribution.

        Parameters
        ----------
        offset : float
            The value of the distribution for the abscissa equal to 0.
        """
        self._offset = x

    def __call__(self, angle_range, angle_resolution, offset):
        r"""For a constant, convolution changes nothing.
        """
        return np.full(np.shape(angle_range), offset)

    def evaluate(self, angle_range):
        r"""For a constant, convolution changes nothing.
        """
        return np.full(np.shape(angle_range), self.offset)

class Linear(UniqueDistribution):
    r"""Child cass for for linear distributions, used e.g., during MDC fitting.
    The linear class is unique, only one instance should be used per task.

    Parameters
    ----------
    offset : float
        The value of the distribution for the abscissa equal to 0.
    slope : float
        The linear slope of the distribution w.r.t. the abscissa.
    """
    def __init__(self, slope, offset, name='Linear'):
        super().__init__(name)
        self.offset = offset
        self.slope = slope

    @property
    def offset(self):
        r"""Returns the offset of the linear distribution.

        Returns
        -------
        offset : float
            The value of the distribution for the abscissa equal to 0.
        """
        return self._offset

    @offset.setter
    def offset(self, x):
        r"""Sets the offset of the linear distribution.

        Parameters
        ----------
        offset : float
            The value of the distribution for the abscissa equal to 0.
        """
        self._offset = x

    @property
    def slope(self):
        r"""Returns the slope of the linear distribution.

        Returns
        -------
        slope : float
            The linear slope of the distribution w.r.t. the abscissa.
        """
        return self._slope

    @slope.setter
    def slope(self, x):
        r"""Sets the slope of the linear distribution.

        Parameters
        ----------
        slope : float
            The linear slope of the distribution w.r.t. the abscissa.
        """
        self._slope = x

    def __call__(self, angle_range, angle_resolution, offset, slope):
        r"""For a constant, convolution changes nothing.
        """
        return offset + slope * angle_range

    def evaluate(self, angle_range):
        r"""For a straight line, convolution changes nothing.
        """
        return self.offset + self.slope * angle_range

class NonUniqueDistribution(Distribution):
    r"""Parent class for unique distributions, to be used one at a time, e.g.,
    during the background of an MDC fit or the Fermi-Dirac distribution.

    Parameters
    ----------
    label : str
        Unique label for instances, identical to the name for unique
        distributions. Not to be modified after instantiation.
    """
    def __init__(self, name, index):
        super().__init__(name)
        self._label = name + '_' + index
        self._index = index

    @property
    def label(self):
        r"""Returns the unique class label.

        Returns
        -------
        label : str
            Unique label for instances, consisting of the label and the index
            for non-unique distributions. Not to be modified after
            instantiation.
        """
        return self._label

    @property
    def index(self):
        r"""Returns the unique class index.

        Returns
        -------
        index : str
            Unique index for instances. Not to be modified after instantiation.
        """
        return self._index

class Dispersion(NonUniqueDistribution):
    r"""Dispersions are assumed to be unique, so they need an index.
    """
    def __init__(self, amplitude, peak, broadening, name, index):
        super().__init__(name, index)
        self.amplitude = amplitude
        self.peak = peak
        self.broadening = broadening

    @property
    def amplitude(self):
        r"""
        """
        return self._amplitude

    @amplitude.setter
    def amplitude(self, x):
        r"""
        """
        self._amplitude = x

    @property
    def peak(self):
        r"""
        """
        return self._peak

    @peak.setter
    def peak(self, x):
        r"""
        """
        self._peak = x

    @property
    def broadening(self):
        r"""
        """
        return self._broadening

    @broadening.setter
    def broadening(self, x):
        r"""
        """
        self._broadening = x

class SpectralLinear(Dispersion):
    r"""Class for the linear dispersion spectral function"""
    def __init__(self, amplitude, peak, broadening, name, index):
        super().__init__(amplitude=amplitude, peak=peak,
                         broadening=broadening, name=name, index=index)

    def __call__(self, angle_range, angle_resolution, amplitude, broadening,
                peak):
        r"""
        """
        result = amplitude / np.pi * broadening / ((np.sin(angle_range * dtor) -
              np.sin(peak * dtor)) ** 2 + broadening ** 2)
        return result

    def evaluate(self, angle_range):
        r"""
        """
        dtor = np.pi / 180

        return self.amplitude / np.pi * self.broadening / ((np.sin(
        angle_range * dtor) - np.sin(self.peak * dtor)) ** 2 +
        self.broadening ** 2)

class SpectralQuadratic(Dispersion):
    r"""Class for the quadratic dispersion spectral function"""
    def __init__(self, amplitude, peak, broadening, side, name, index,
                 center_wavevector=None, center_angle=None):
        self.check_center_coordinates(center_wavevector, center_angle)
        super().__init__(amplitude=amplitude, peak=peak,
                         broadening=broadening, name=name, index=index)
        self.side = side
        self.center_wavevector = center_wavevector
        self.center_angle = center_angle

    @property
    def center_angle(self):
        r"""TBD
        """
        return self._center_angle

    @center_angle.setter
    def center_angle(self, x):
        r"""TBD
        """
        self._center_angle = x

    @property
    def center_wavevector(self):
        r"""TBD
        """
        return self._center_wavevector

    @center_wavevector.setter
    def center_wavevector(self, x):
        r"""TBD
        """
        self._center_wavevector = x

    @property
    def side(self):
        r"""TBD
        """
        return self._side

    @side.setter
    def side(self, x):
        r"""TBD
        """
        self._side = x

    def check_center_coordinates(self, center_wavevector, center_angle):
        r"""TBD
        """
        if (center_wavevector is None and center_angle is None) \
        or (center_wavevector is not None and center_angle is not None):
            raise ValueError('Please specify exactly one of ' +
                             'center_wavevector and center_angle.')

    def check_binding_angle(self, binding_angle):
        r"""TBD
        """
        if np.isnan(binding_angle):
            raise ValueError('The provided wavevector cannot be reached ' +
                             'with the available range of kinetic ' +
                             'energies. Please check again.')

    def __call__(self, angle_range, amplitude, broadening,
                peak, kinetic_energy, hnuminphi, center_wavevector=None,
                 center_angle=None):
        r"""TBD
        """
        self.check_center_coordinates(center_wavevector, center_angle)

        if center_wavevector is not None:
            binding_angle = np.arcsin(np.sqrt(pref / kinetic_energy)
                                      * center_wavevector) / dtor
            self.check_binding_angle(binding_angle)
        elif center_angle is not None:
            binding_angle = self.center_angle * np.sqrt(hnuminphi /
                                                      kinetic_energy)

        return amplitude / np.pi * broadening / (((np.sin(angle_range * dtor)
             - np.sin(binding_angle * dtor)) ** 2 - np.sin(peak * dtor) ** 2)
             ** 2 + broadening ** 2)

    def evaluate(self, angle_range, kinetic_energy, hnuminphi):
        r"""TBD
        """
        if self.center_wavevector is not None:
            binding_angle = np.arcsin(np.sqrt(pref / kinetic_energy)
                                      * self.center_wavevector) / dtor
            self.check_binding_angle(binding_angle)
        elif self.center_angle is not None:
            binding_angle = self.center_angle * np.sqrt(hnuminphi /
                                                      kinetic_energy)

        return self.amplitude / np.pi * self.broadening / (((np.sin(
            angle_range * dtor) - np.sin(binding_angle * dtor)) ** 2 - np.sin(
            self.peak * dtor) ** 2) ** 2 + self.broadening ** 2)

    @add_fig_kwargs
    def plot(self, angle_range, angle_resolution, kinetic_energy, hnuminphi,
             matrix_element=None, matrix_args=None, ax=None, **kwargs):
        r"""Overwrites generic class plotting method.
        """
        from scipy.ndimage import gaussian_filter

        ax, fig, plt = get_ax_fig_plt(ax=ax)

        ax.set_xlabel('Angle ($\degree$)')
        ax.set_ylabel('Counts (-)')

        extend, step, numb = self.extend_range(angle_range, angle_resolution)

        extended_result = self.evaluate(extend, kinetic_energy, hnuminphi)

        if matrix_element is not None:
            extended_result *= matrix_element(extend, **matrix_args)

        final_result = gaussian_filter(extended_result, sigma=step)[
        numb:-numb if numb else None]

        ax.plot(angle_range, final_result, label=self.label)

        ax.legend()

        return fig
