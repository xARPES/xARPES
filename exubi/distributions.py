# To do: Convert the Fermi-Dirac distribution to a proper distribution
# Convolve will be a method for the distributions, whether it be an energy or momentum convolution.

import numpy as np

class distribution:
    """
    TBD
    """
    def __init__(self, name):
        self.type = type
        self.name = name


class fermi_dirac(distribution):
    """
    Returns the Fermi-Dirac distribution. 

    Parameters
    ----------
    background

    integrated_weight

    
    Methods
    ----------
    
    """
    
    def __init__(self, temperature, mu, background=0, integrated_weight=1, energy_resolution=None, name="fermi_dirac"):
        super().__init__(type)
        self.temperature=temperature
        self.mu=mu
        self.background=background
        self.integrated_weight=integrated_weight
        self.name=name
        self.energy_resolution=energy_resolution


    def __call__(self, energy_range, mu, background, integrated_weight):
        """
        TBD
        """
        from scipy.ndimage import gaussian_filter
        fwhm_to_std = np.sqrt(8*np.log(2)) # Conversion from FWHM to standard deviation [-]
        k_B = 8.617e-5                     # Boltzmann constant [eV/K]
        k_BT = self.temperature*k_B
        step_size=np.abs(energy_range[1]-energy_range[0])
        result = integrated_weight/(1+np.exp((energy_range-mu)/k_BT))+background
        return gaussian_filter(result, sigma=self.energy_resolution/(fwhm_to_std*step_size))

    
    def calculate(self, energy_range):
        """
        Calculates the Fermi-Dirac distribution on a certain energy range
        """
        k_B = 8.617e-5 # Boltzmann constant [eV/K]
        k_BT = self.temperature*k_B
        return self.integrated_weight/(1+np.exp((energy_range-self.mu)/k_BT))+self.background


    def convolve(self, energy_range):
        """
        TBD
        """
        # If energy_resolution is not defined, this function shouldn't work
        from scipy.ndimage import gaussian_filter
        step_size=np.abs(energy_range[1]-energy_range[0])
        fwhm_to_std = np.sqrt(8*np.log(2)) # Conversion from FWHM to standard deviation [-]
        return gaussian_filter(self.calculate(energy_range), sigma=self.energy_resolution/(fwhm_to_std*step_size))
        












