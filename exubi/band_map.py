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


    
    def shift_angles(self, shift):
        """
        Shifts the angles by the specified amount. Used to align with respect to experimentally
        observed angles.
        """
        self.angles=self.angles + shift
           
    
    @add_fig_kwargs
    def fit_fermi_edge(self, mu_guess, background_guess=0, integrated_weight_guess=1, \
                       angle_min=-np.infty, angle_max=np.infty, ekin_min=-np.infty, ekin_max=np.infty, \
                        ax=None, **kwargs):
        """
        Plots the band map.

        Returns
        ----------
        Matplotlib-Figure
        """
        from exubi.functions import fit_leastsq        
        
        ax, fig, plt = get_ax_fig_plt(ax=ax)
        
        min_angle_index = np.argmin(np.abs(self.angles-angle_min))
        max_angle_index = np.argmin(np.abs(self.angles-angle_max))
        
        min_ekin_index = np.argmin(np.abs(self.ekin-ekin_min))
        max_ekin_index = np.argmin(np.abs(self.ekin-ekin_max))

        energy_range = self.ekin[min_ekin_index:max_ekin_index]
        
        integrated_intensity = np.trapz(self.intensities[min_ekin_index:max_ekin_index, min_angle_index:max_angle_index], axis=1)

    
        fdir_initial = fermi_dirac(temperature=self.temperature, mu=mu_guess, background=background_guess, \
                                   integrated_weight=integrated_weight_guess, name="Initial guess")

        parameters = np.array([mu_guess, background_guess, integrated_weight_guess])
        
        extra_args = (self.energy_resolution)
        
        popt, pcov = fit_leastsq(parameters, energy_range, integrated_intensity, fdir_initial, extra_args)
        
        fdir_final = fermi_dirac(temperature=self.temperature, mu=popt[0], background=popt[1], integrated_weight=popt[2], \
                                  name="Fitted result")

        self.mu = popt[0]

        ax.set_xlabel(r"$E_{\mathrm{kin}}$ (-)")
        ax.set_ylabel("Counts (-)")
        ax.set_xlim([ekin_min, ekin_max])
        
        ax.plot(energy_range, integrated_intensity, label="Data")
        ax.plot(energy_range, fdir_initial.convolve(energy_range, energy_resolution=self.energy_resolution), label=fdir_initial.name)
        ax.plot(energy_range, fdir_final.convolve(energy_range, energy_resolution=self.energy_resolution), label=fdir_final.name)

        ax.legend()

        return fig










            



                
