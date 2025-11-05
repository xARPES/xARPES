#!/usr/bin/env python3

# $\rm{SrTiO}_3$ example
### In this example, we extract the self-energies and Eliashberg function from  
### a 2DEL in the $d_{xy}$ bands on the $\rm{TiO}_{2}$-terminated surface of $\rm{SrTiO}_3$.

import matplotlib as mpl
mpl.use('Qt5Agg')

import xarpes
import matplotlib.pyplot as plt
import os

xarpes.plot_settings('default')

script_dir = xarpes.set_script_dir()

dfld = 'data_sets' # Folder containing the data
flnm = 'STO_2_0010STO_2_' # Name of the file
extn = '.ibw' # Extension of the file

data_file_path = os.path.join(script_dir, dfld, flnm + extn)


fig = plt.figure(figsize=(8, 5))
ax = fig.gca()

bmap = xarpes.BandMap(data_file_path, energy_resolution=0.01,
                      angle_resolution=0.2, temperature=20)

bmap.shift_angles(shift=-0.57)

fig = bmap.plot(abscissa='angle', ordinate='kinetic_energy', ax=ax)


fig = bmap.fit_fermi_edge(hnuminphi_guess=42.24, background_guess=1e4,
                          integrated_weight_guess=1e6, angle_min=-5,
                          angle_max=5, ekin_min=42.22, ekin_max=42.3,
                          show=True, title='Fermi edge fit')

print('The optimised h nu - Phi = ' + f'{bmap.hnuminphi:.4f}' + ' +/- '
      + f'{bmap.hnuminphi_std:.4f}' + ' eV.')


from xarpes.constants import dtor

k_0 = -0.0014
theta_0 = 0

guess_dists = xarpes.CreateDistributions([
xarpes.Constant(offset=600),
xarpes.SpectralQuadratic(amplitude=3800, peak=-2.45, broadening=0.00024,
            center_wavevector=k_0, name='Inner_band', index='1'),
xarpes.SpectralQuadratic(amplitude=1800, peak=-3.6, broadening=0.0004,
            center_wavevector=k_0, name='Outer_band', index='2')
])

import numpy as np

mat_el = lambda x: np.sin((x - theta_0) * dtor) ** 2

mat_args = {}

energy_range = [-0.1, 0.003]
angle_min = 0.0
angle_max = 4.8

mdcs = xarpes.MDCs(*bmap.mdc_set(angle_min, angle_max, energy_range=energy_range))

fig = plt.figure(figsize=(7, 5))
ax = fig.gca()

fig = mdcs.visualize_guess(distributions=guess_dists, matrix_element=mat_el,
                           matrix_args=mat_args, energy_value=-0.000, ax=ax)


fig = plt.figure(figsize=(7, 5))
ax = fig.gca()

fig = mdcs.fit_selection(distributions=guess_dists, matrix_element=mat_el, 
                         matrix_args=mat_args, ax=ax)

# fig = mdcs.fit_selection(distributions=guess_dists, matrix_element=mat_el, 
#                          matrix_args=mat_args, ax=ax)

self_energy = xarpes.SelfEnergy(*mdcs.expose_parameters(select_label='Inner_band_1', 
                                bare_mass=0.6, fermi_wavevector=0.14, side='right'))

self_two = xarpes.SelfEnergy(*mdcs.expose_parameters(select_label='Outer_band_2',
                                bare_mass=0.6, fermi_wavevector=0.207))

self_two.side='right'

fig = plt.figure(figsize=(7, 5))
ax = fig.gca()

from xarpes.constants import stdv

ax.errorbar(self_energy.enel_range, self_energy.imag, 
            yerr=stdv * self_energy.imag_sigma, label =r"$-\Sigma''_{\rm{IR}}(E)$")
ax.errorbar(self_energy.enel_range, self_two.imag, 
            yerr=stdv * self_two.imag_sigma, label =r"$-\Sigma''_{\rm{OR}}(E)$")
ax.errorbar(self_energy.enel_range, self_energy.real, 
            yerr=stdv * self_energy.real_sigma, label =r"$\Sigma'_{\rm{IR}}(E)$")
ax.errorbar(self_energy.enel_range, self_two.real,
            yerr=stdv * self_two.real_sigma, label =r"$\Sigma'_{\rm{OR}}(E)$")
ax.set_xlabel(r'$E-\mu$ (eV)'); ax.set_ylabel(r"$\Sigma'(E), -\Sigma''(E)$ (eV)")

ax.set_ylim([0, 0.06])

plt.legend(); plt.show()




fig = plt.figure(figsize=(10, 7))
ax = fig.gca()

from xarpes.constants import stdv

ax.errorbar(self_energy.peak_positions, self_energy.enel_range, 
            xerr=stdv * self_energy.peak_positions_sigma,
           markersize=2, color='tab:blue', label=self_energy.label)
ax.errorbar(self_two.peak_positions, self_two.enel_range, 
            xerr=stdv * self_two.peak_positions_sigma,
           markersize=2, color='tab:purple', label=self_two.label)

ax.set_xlim([-0.25, 0.25]); ax.set_ylim([-0.3, 0.1])

plt.legend()

fig = bmap.plot(abscissa='momentum', ordinate='electron_energy', ax=ax)

plt.show()


guess_dists = xarpes.CreateDistributions([
xarpes.Constant(offset=600),

xarpes.SpectralQuadratic(amplitude=8, peak=2.45, broadening=0.00024,
            center_wavevector=k_0, name='Inner_nm', index='1'),

xarpes.SpectralQuadratic(amplitude=8, peak=3.6, broadening=0.0004,
            center_wavevector=k_0, name='Outer_nm', index='2')
])

energy_range = [-0.1, 0.003]
angle_min=0.0
angle_max=5.0

mdcs = xarpes.MDCs(*bmap.mdc_set(angle_min, angle_max, energy_range=energy_range))

fig = plt.figure(figsize=(7, 5))
ax = fig.gca()

fig = mdcs.visualize_guess(distributions=guess_dists, ax=ax, energy_value=0)


fig = plt.figure(figsize=(7, 5))
ax = fig.gca()

fig = mdcs.fit_selection(distributions=guess_dists, ax=ax)

self_three = xarpes.SelfEnergy(*mdcs.expose_parameters(select_label='Inner_nm_1', side='right',
                                fermi_wavevector=10))

self_four = xarpes.SelfEnergy(*mdcs.expose_parameters(select_label='Outer_nm_2', side='right',
                                fermi_wavevector=10))


fig = plt.figure(figsize=(10, 7))
ax = fig.gca()

from xarpes.constants import stdv

ax.errorbar(self_energy.peak_positions, self_energy.enel_range, 
            xerr=stdv * self_energy.peak_positions_sigma,
           markersize=2, color='tab:blue', label=self_energy.label)
ax.errorbar(self_two.peak_positions, self_two.enel_range, 
            xerr=stdv * self_two.peak_positions_sigma,
           markersize=2, color='tab:purple', label=self_two.label)
ax.errorbar(self_three.peak_positions, self_three.enel_range, 
            xerr=stdv * self_three.peak_positions_sigma,
            markersize=2, color='tab:brown', label=self_three.label)
ax.errorbar(self_four.peak_positions, self_four.enel_range, 
            xerr=stdv * self_four.peak_positions_sigma, 
            markersize=2, color='palevioletred', label=self_four.label)

ax.set_xlim([0, 0.25]); ax.set_ylim([-0.15, 0.05])

plt.legend()

# Put <1 zorder such that scatters aren't overrridden
bmap.plot(abscissa='momentum', ordinate='electron_energy', ax=ax, zorder=0.5)

plt.show()




# fig = plt.figure(figsize=(7, 5))
# ax = fig.gca()

# from xarpes.constants import dtor

# k_0 = -0.0014
# theta_0 = 0

# guess_dists = xarpes.CreateDistributions([
# xarpes.Constant(offset=600),

# xarpes.SpectralQuadratic(amplitude=3800, peak=2.45, broadening=0.00024,
#             center_wavevector=k_0, name='Inner_band', index='2'),

# xarpes.SpectralQuadratic(amplitude=1800, peak=3.6, broadening=0.0004,
#             center_wavevector=k_0, name='Outer_band', index='3')
# ])

# import numpy as np
# mat_el = lambda x: np.sin((x - theta_0) * dtor) ** 2

# # mat_el = lambda x, theta_0: np.sin((x - theta_0) * dtor) ** 2

# mat_args = {
# #  'theta_0' : 0.0
# }

# fig = mdc.visualize_guess(distributions=guess_dists, matrix_element=mat_el,
#                            ax=ax, matrix_args=mat_args, show=True)


# fig = plt.figure(figsize=(7, 5))
# ax = fig.gca()

# fig, new_dists, covariance_matrix, new_mat_args = mdc.fit(
#     distributions=guess_dists, matrix_element=mat_el, matrix_args=mat_args,
#     ax=ax)


angle_min = 0.0
angle_max = 5.0
en_val = 0.0

mdc = xarpes.MDCs(*bmap.mdc_set(angle_min, angle_max, energy_value=en_val))

fig = plt.figure(figsize=(6, 5))
ax = fig.gca()

fig = mdc.plot(ax=ax)


fig = plt.figure(figsize=(7, 5))
ax = fig.gca()

from xarpes.constants import dtor

k_0 = -0.0014
theta_0 = 0

guess_dists = xarpes.CreateDistributions([
xarpes.Constant(offset=600),

xarpes.SpectralLinear(amplitude=3000, peak=2.4, broadening=0.0001,
                      name='Faint_band', index='1'),

xarpes.SpectralQuadratic(amplitude=3800, peak=2.45, broadening=0.00024,
            center_wavevector=k_0, name='Inner_band', index='2'),

xarpes.SpectralQuadratic(amplitude=1800, peak=3.6, broadening=0.0004,
            center_wavevector=k_0, name='Outer_band', index='3')
])

import numpy as np
mat_el = lambda x: np.sin((x - theta_0) * dtor) ** 2

# mat_el = lambda x, theta_0: np.sin((x - theta_0) * dtor) ** 2

mat_args = {
#  'theta_0' : 0.0
}

fig = mdc.visualize_guess(distributions=guess_dists, matrix_element=mat_el,
                           ax=ax, matrix_args=mat_args, show=True)



fig = plt.figure(figsize=(7, 5))
ax = fig.gca()

fig = mdcs.fit_selection(distributions=new_dists, matrix_element=mat_el, 
                         matrix_args=mat_args, ax=ax)

# fig = mdcs.fit_selection(distributions=guess_dists, matrix_element=mat_el, 
#                          matrix_args=mat_args, ax=ax)




fig = plt.figure(figsize=(7, 5))
ax = fig.gca()

fig, new_dists, covariance_matrix, new_mat_args = mdc.fit(
    distributions=guess_dists, matrix_element=mat_el, matrix_args=mat_args,
    ax=ax)


energy_range = [-0.1, 0.01]

angle_min = 0.0
angle_max = 4.8

mdcs = xarpes.MDCs(*bmap.mdc_set(angle_min, angle_max, energy_range=energy_range))

fig = plt.figure(figsize=(7, 5))
ax = fig.gca()

fig = mdcs.plot(angle_range=mdcs.angles, angle_resolution=0.2, ax=ax)


fig = plt.figure(figsize=(7, 5))
ax = fig.gca()

fig, new_dists, covariance_matrix, new_mat_args = mdcs.fit(
    distributions=guess_dists, matrix_element=mat_el, matrix_args=mat_args,
    energy_value=0, ax=ax)


guess_dists = xarpes.CreateDistributions([
xarpes.Constant(offset=600),
xarpes.SpectralQuadratic(amplitude=3800, peak=-2.45, broadening=0.00024,
            center_wavevector=k_0, name='Inner_band', index='1'),
xarpes.SpectralQuadratic(amplitude=1800, peak=-3.6, broadening=0.0004,
            center_wavevector=k_0, name='Outer_band', index='2')
])

import numpy as np

mat_el = lambda x: np.sin((x - theta_0) * dtor) ** 2

mat_args = {}

energy_range = [-0.1, 0.003]
angle_min = 0.0
angle_max = 4.8

mdcs = xarpes.MDCs(*bmap.mdc_set(angle_min, angle_max, energy_range=energy_range))

fig = plt.figure(figsize=(7, 5))
ax = fig.gca()

fig = mdcs.visualize_guess(distributions=guess_dists, matrix_element=mat_el,
                           matrix_args=mat_args, energy_value=-0.000, ax=ax)


fig = plt.figure(figsize=(7, 5))
ax = fig.gca()

fig, new_dists, cov_mat, new_mat_args = mdcs.fit(distributions=guess_dists, 
    matrix_element=mat_el, matrix_args=mat_args, energy_value=0.0, ax=ax)


fig = plt.figure(figsize=(7, 5))
ax = fig.gca()

fig = mdcs.fit_selection(distributions=new_dists, matrix_element=mat_el, 
                         matrix_args=mat_args, ax=ax)

# fig = mdcs.fit_selection(distributions=guess_dists, matrix_element=mat_el, 
#                          matrix_args=mat_args, ax=ax)

For quadratic bands, the user has to explicitly assign the peaks as left-hand or right-hand side.
In theory, one could incorporate such information in a minus sign of the peak position.
However, this would also require setting boundaries for the fitting range.
Instead, the user is advised to carefully check correspondence of peak maxima with MDC fitting results.

self_energy = xarpes.SelfEnergy(*mdcs.expose_parameters(select_label='Inner_band_1', side='right'))

self_two = xarpes.SelfEnergy(*mdcs.expose_parameters(select_label='Outer_band_2'))

self_two.side='right'


fig = plt.figure(figsize=(10, 7))
ax = fig.gca()

from xarpes.constants import stdv

ax.errorbar(self_energy.peak_positions, self_energy.enel_range, 
            xerr=stdv * self_energy.peak_positions_sigma,
           markersize=2, color='tab:blue', label=self_energy.label)
ax.errorbar(self_two.peak_positions, self_two.enel_range, 
            xerr=stdv * self_two.peak_positions_sigma,
           markersize=2, color='tab:purple', label=self_two.label)

ax.set_xlim([-0.25, 0.25])
ax.set_ylim([-0.3, 0.1])

plt.legend()
fig = bmap.plot(abscissa='momentum', ordinate='electron_energy', ax=ax)

plt.show()


guess_dists = xarpes.CreateDistributions([
xarpes.Constant(offset=600),

xarpes.SpectralQuadratic(amplitude=8, peak=2.45, broadening=0.00024,
            center_wavevector=k_0, name='Inner_nm', index='1'),

xarpes.SpectralQuadratic(amplitude=8, peak=3.6, broadening=0.0004,
            center_wavevector=k_0, name='Outer_nm', index='2')
])

energy_range = [-0.1, 0.003]
angle_min=0.0
angle_max=5.0

mdcs = xarpes.MDCs(*bmap.mdc_set(angle_min, angle_max, energy_range=energy_range))

fig = plt.figure(figsize=(7, 5))
ax = fig.gca()

fig = mdcs.visualize_guess(distributions=guess_dists, ax=ax, energy_value=0)


fig = plt.figure(figsize=(7, 5))
ax = fig.gca()

fig, new_dists, cov_mat = mdcs.fit(distributions=guess_dists, energy_value=0, ax=ax)


fig = plt.figure(figsize=(7, 5))
ax = fig.gca()

fig = mdcs.fit_selection(distributions=new_dists, ax=ax)

self_three = xarpes.SelfEnergy(*mdcs.expose_parameters(select_label='Inner_nm_1', side='right',
                                fermi_wavevector=10))

self_four = xarpes.SelfEnergy(*mdcs.expose_parameters(select_label='Outer_nm_2', side='right',
                                fermi_wavevector=10))


fig = plt.figure(figsize=(10, 7))
ax = fig.gca()

from xarpes.constants import stdv

ax.errorbar(self_energy.peak_positions, self_energy.enel_range, 
            xerr=stdv * self_energy.peak_positions_sigma,
           markersize=2, color='tab:blue', label=self_energy.label)
ax.errorbar(self_two.peak_positions, self_two.enel_range, 
            xerr=stdv * self_two.peak_positions_sigma,
           markersize=2, color='tab:purple', label=self_two.label)
ax.errorbar(self_three.peak_positions, self_three.enel_range, 
            xerr=stdv * self_three.peak_positions_sigma,
            markersize=2, color='tab:brown', label=self_three.label)
ax.errorbar(self_four.peak_positions, self_four.enel_range, 
            xerr=stdv * self_four.peak_positions_sigma, 
            markersize=2, color='palevioletred', label=self_four.label)

ax.set_xlim([0, 0.25]); ax.set_ylim([-0.15, 0.05])

plt.legend()

# Put <1 zorder such that scatters aren't overrridden
bmap.plot(abscissa='momentum', ordinate='electron_energy', ax=ax, zorder=0.5)

plt.show()


