"""The band map class and its operations."""

import numpy as np
from exubi.plotting import get_ax_fig_plt, add_fig_kwargs
from exubi.distributions import fermi_dirac


class band_map():
    """Raw data from the ARPES experiment.

    Parameters
    ----------
    intensities: ndarray
        Intensities of the band map.
    angles: ndarray
        Theta angle values.
    ekin: ndarray
        Kinetic energies detected during experiment.
    temperature: float
        Temperature of the band map in Kelvin.
    """
    
    def __init__(self, intensities, angles, ekin, energy_resolution=None, temperature=None, mu=None):
        self.intensities=intensities
        self.angles=angles
        self.ekin=ekin
        self.energy_resolution=energy_resolution
        self.temperature=temperature
        self.mu=mu


    @property
    def mu(self):
        """
        Returns the chemical potential corresponding to a particular kinetic energy.
        """
        return self._mu

    
    @mu.setter
    def mu(self, mu):
        """
        Sets the chemical potential corresponding to a particular kinetic energy.
        """
        self._mu=mu

    
    @add_fig_kwargs
    def fit_fermi_edge(self, mu, temperature, energy_resolution, angle_min=-np.infty, angle_max=np.infty, ekin_min=-np.infty, ekin_max=np.infty, \
                    background=0, integrated_weight=1, ax=None, fontsize=12, **kwargs):
        """
        Plots the band map.

        Returns
        ----------
        Matplotlib-Figure
        """
        from scipy.optimize import curve_fit
        
        ax, fig, plt = get_ax_fig_plt(ax=ax)
        
        min_angle_index = np.argmin(np.abs(self.angles-angle_min))
        max_angle_index = np.argmin(np.abs(self.angles-angle_max))
        
        min_ekin_index = np.argmin(np.abs(self.ekin-ekin_min))
        max_ekin_index = np.argmin(np.abs(self.ekin-ekin_max))

        energy_range = self.ekin[min_ekin_index:max_ekin_index]
        
        integrated_intensity = np.trapz(self.intensities[min_ekin_index:max_ekin_index, min_angle_index:max_angle_index], axis=1)

    
        fdir_initial = fermi_dirac(temperature=temperature, mu=mu, background=background, integrated_weight=integrated_weight, \
                                    energy_resolution=self.energy_resolution, name="Initial guess")

        parameters = np.array([mu, background, integrated_weight])
        
        popt, pcov = curve_fit(fdir_initial, energy_range, integrated_intensity, p0=parameters)

        fdir_final = fermi_dirac(temperature=temperature, mu=popt[0], background=popt[1], integrated_weight=popt[2], \
                                 energy_resolution=self.energy_resolution, name="Fitted result")

        self.mu = popt[0]

        ax.set_xlabel("$E_{\mathrm{kin}}$ (-)", fontsize=fontsize)
        ax.set_ylabel("Counts (-)", fontsize=fontsize)
        ax.set_xlim([ekin_min, ekin_max])
        
        
        ax.plot(energy_range, integrated_intensity, label="Data")
        ax.plot(energy_range, fdir_initial.convolve(energy_range), label=fdir_initial.name)
        ax.plot(energy_range, fdir_final.convolve(energy_range), label=fdir_final.name)

        ax.legend(fontsize=fontsize)

        return fig










            



                
