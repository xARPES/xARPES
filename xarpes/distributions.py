# To do: Convert the Fermi-Dirac distribution to a proper distribution
# Convolve will be a method for the distributions, whether it be an energy or momentum convolution.


class distribution:
    """
    TBD
    """
    def __init__(self, name):
        self.name = name

    def dist(self):
        return self._dist


class unique_distribution(distribution):
    """
    TBD
    """
    def __init__(self, name):
        super().__init__(name)
        self._label = name

    def label(self):
        return self._label


class fermi_dirac(unique_distribution):
    """
    Returns the Fermi-Dirac distribution.

    Parameters
    ----------
    background

    integrated_weight

    Methods
    ----------

    """

    def __init__(self, temperature, hnuminphi, background=0, integrated_weight=1, name='fermi_dirac'):
        super().__init__(name)
        self.temperature = temperature
        self.hnuminphi = hnuminphi
        self.background = background
        self.integrated_weight = integrated_weight
        self.name = name

    def __call__(self, energy_range, hnuminphi, background, integrated_weight, energy_resolution):
        """
        TBD
        """
        from scipy.ndimage import gaussian_filter
        import numpy as np
        sigma_extend = 5 # Extend data range by "5 sigma"
        fwhm_to_std = np.sqrt(8 * np.log(2)) # Conversion from FWHM to standard deviation [-]
        k_B = 8.617e-5 # Boltzmann constant [eV/K]
        k_BT = self.temperature * k_B
        step_size = np.abs(energy_range[1] - energy_range[0])
        estep = energy_resolution / (step_size * fwhm_to_std)
        enumb = int(sigma_extend * estep)
        extend = np.linspace(energy_range[0] - enumb * step_size, energy_range[-1] + enumb * step_size, \
                             len(energy_range) + 2 * enumb)
        result = integrated_weight / (1 + np.exp((extend - hnuminphi) / k_BT)) + background
        return gaussian_filter(result, sigma=estep)[enumb:-enumb]

    def evaluate(self, energy_range):
        """
        Evaluates the Fermi-Dirac distribution on a certain energy range.
        """
        import numpy as np
        k_B = 8.617e-5 # Boltzmann constant [eV/K]
        k_BT = self.temperature * k_B
        return self.integrated_weight / (1 + np.exp((energy_range - self.hnuminphi) / k_BT)) + self.background

    def convolve(self, energy_range, energy_resolution):
        """
        TBD
        """
        import numpy as np
        from scipy.ndimage import gaussian_filter
        sigma_extend = 5 # Extend data range by "5 sigma"
        fwhm_to_std = np.sqrt(8 * np.log(2)) # Conversion from FWHM to standard deviation [-]
        step_size = np.abs(energy_range[1] - energy_range[0])
        estep = energy_resolution / (step_size * fwhm_to_std)
        enumb = int(sigma_extend * estep)
        extend = np.linspace(energy_range[0] - enumb * step_size, energy_range[-1] + enumb * step_size, \
                             len(energy_range) + 2 * enumb)
        return gaussian_filter(self.evaluate(extend), sigma=estep)[enumb:-enumb]


class constant(unique_distribution):
    """
    TBD
    """
    def __init__(self, offset):
        super().__init__(name='constant')
        self._offset = offset

    def offset(self):
        return self._offset

    def set_offset(self, x):
        self._offset = x


class linear(unique_distribution):
    """
    TBD
    """
    def __init__(self, slope, offset):
        super().__init__(name='linear')
        self._offset = offset
        self._slope = slope

    def offset(self):
        return self._offset

    def set_offset(self, x):
        self._offset = x

    def slope(self):
        return self._slope

    def set_slope(self, x):
        self._slope = x
