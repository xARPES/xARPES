# Copyright (C) 2024 xARPES Developers
# This program is free software under the terms of the GNU GPLv3 license.

"""The distributions used throughout the code."""

class distribution:
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
        r"""Returns the name of the class instance.

        Returns
        -------
        name : str
            Non-unique name for instances, not to be modified after
            instantiation.
        """
        return self._name

class unique_distribution(distribution):
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

class constant(unique_distribution):
    r"""Child class for constant distributions, used e.g., during MDC fitting.
    The constant class is unique, only one instance should be used per task.

    Parameters
    ----------
    offset : float
        The value of the distribution for the abscissa equal to 0.
    """
    def __init__(self, offset):
        super().__init__(name='constant')
        self._offset = offset

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
    def set_offset(self, x):
        r"""Sets the offset of the constant distribution.

        Parameters
        ----------
        offset : float
            The value of the distribution for the abscissa equal to 0.
        """
        self._offset = x

class linear(unique_distribution):
    r"""Child cass for for linear distributions, used e.g., during MDC fitting.
    The constant class is unique, only one instance should be used per task.

    Parameters
    ----------
    offset : float
        The value of the distribution for the abscissa equal to 0.
    slope : float
        The linear slope of the distribution w.r.t. the abscissa.
    """
    def __init__(self, slope, offset):
        super().__init__(name='linear')
        self._offset = offset
        self._slope = slope

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
    def set_offset(self, x):
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
    def set_slope(self, x):
        r"""Sets the slope of the linear distribution.

        Parameters
        ----------
        slope : float
            The linear slope of the distribution w.r.t. the abscissa.
        """
        self._slope = x

class linear(unique_distribution):
    r"""Child cass for for linear distributions, used e.g., during MDC fitting.
    The constant class is unique, only one instance should be used per task.

    Parameters
    ----------
    offset : float
        The value of the distribution for the abscissa equal to 0.
    slope : float
        The linear slope of the distribution w.r.t. the abscissa.
    """
    def __init__(self, slope, offset):

        super().__init__(name='linear')
        self._offset = offset
        self._slope = slope

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
    def set_offset(self, x):
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
    def set_slope(self, x):
        r"""Sets the slope of the linear distribution.

        Parameters
        ----------
        slope : float
            The linear slope of the distribution w.r.t. the abscissa.
        """
        self._slope = x

class fermi_dirac(unique_distribution):
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
                 integrated_weight=1, name='fermi_dirac'):
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
        def set_temperature(self, x):
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
        def set_hnuminphi(self, x):
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
        def set_background(self, x):
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
        def set_integrated_weight(self, x):
            r"""Sets the integrated weight of the FD distribution.

            Parameters
            ----------
            integrated_weight : float
                Integrated weight on top of the background [counts]
            """
            self._integrated_weight = x

    def __call__(self, energy_range, hnuminphi, background, integrated_weight,
                 energy_resolution):
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
        from scipy.ndimage import gaussian_filter
        import numpy as np
        sigma_extend = 5 # Extend data range by "5 sigma"
        # Conversion from FWHM to standard deviation [-]
        fwhm_to_std = np.sqrt(8 * np.log(2))
        k_B = 8.617e-5 # Boltzmann constant [eV/K]
        k_BT = self.temperature * k_B
        step_size = np.abs(energy_range[1] - energy_range[0])
        estep = energy_resolution / (step_size * fwhm_to_std)
        enumb = int(sigma_extend * estep)
        extend = np.linspace(energy_range[0] - enumb * step_size,
                             energy_range[-1] + enumb * step_size,
                             len(energy_range) + 2 * enumb)
        result = (integrated_weight / (1 + np.exp((extend - hnuminphi) / k_BT))
            + background)
        evalf = gaussian_filter(result, sigma=estep)[enumb:-enumb]
        return evalf

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
        import numpy as np
        k_B = 8.617e-5 # Boltzmann constant [eV/K]
        k_BT = self.temperature * k_B
        evalf = (self.integrated_weight
            / (1 + np.exp((energy_range - self.hnuminphi) / k_BT))
            + self.background)
        return evalf

    def convolve(self, energy_range, energy_resolution):
        r"""Evaluates the FD distribution for a given class instance and
        performs the energy convolution with the given resolution. The
        convolution is performed with an expanded abscissa range of 5
        times the standard deviation.

        Parameters
        ----------
        energy_range : ndarray
            1D array on which to evaluate and convolve FD distribution [eV]
        energy_resolution : float
            Energy resolution of the detector for the convolution [eV]

        Returns
        -------
        evalf : ndarray
            1D array of the energy-convolved FD distribution [counts]
        """
        import numpy as np
        from scipy.ndimage import gaussian_filter
        sigma_extend = 5 # Extend data range by "5 sigma"
        # Conversion from FWHM to standard deviation [-]
        fwhm_to_std = np.sqrt(8 * np.log(2))
        step_size = np.abs(energy_range[1] - energy_range[0])
        estep = energy_resolution / (step_size * fwhm_to_std)
        enumb = int(sigma_extend * estep)
        extend = np.linspace(energy_range[0] - enumb * step_size,
                             energy_range[-1] + enumb * step_size,
                             len(energy_range) + 2 * enumb)
        evalf = gaussian_filter(self.evaluate(extend),
                                sigma=estep)[enumb:-enumb]
        return evalf