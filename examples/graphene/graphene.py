#!/usr/bin/env python3

# Intercalated graphene
### In this example, we extract the self-energies and Eliashberg function of
### intercalated graphene.
### Data have been provided with permission from re-use, originating from
### https://journals.aps.org/prb/abstract/10.1103/PhysRevB.97.085132.

import matplotlib as mpl
mpl.use('Qt5Agg')

import xarpes
import matplotlib.pyplot as plt
import os

xarpes.plot_settings('default')

script_dir = xarpes.set_script_dir()

dfld = 'data_sets'    # Folder containing the data
flnm = 'graphene_152' # Name of the file
extn = '.ibw'         # Extension of the file

data_file_path = os.path.join(script_dir, dfld, flnm + extn)


fig = plt.figure(figsize=(8, 5))
ax = fig.gca()

bmap = xarpes.BandMap(data_file_path, energy_resolution=0.01,
                      angle_resolution=0.1, temperature=50)

bmap.shift_angles(shift=-2.28)

fig = bmap.plot(abscissa='momentum', ordinate='kinetic_energy', ax=ax)


fig, ax = plt.subplots(2, 1, figsize=(6, 8))

fig = bmap.correct_fermi_edge(
      hnuminphi_guess=32, background_guess=1e2,
      integrated_weight_guess=1e3, angle_min=-10, angle_max=10,
      ekin_min=31.96, ekin_max=32.1, true_angle=0,
      ax=ax[0], show=False, fig_close=False)

fig = bmap.plot(ordinate='electron_energy', abscissa='momentum',
      ax=ax[1], show=False, fig_close=False)

# Figure customization
ax[0].set_xlabel(''); ax[0].set_xticklabels([])
ax[0].set_title('Fermi correction fit')
fig.subplots_adjust(top=0.92, hspace=0.1)
plt.show()

print('The optimised hnu - Phi=' + f'{bmap.hnuminphi:.4f}' + ' +/- '
      + f'{1.96 * bmap.hnuminphi_std:.5f}' + ' eV.')

# fig = bmap.plot(ordinate='kinetic_energy', abscissa='angle')


fig = plt.figure(figsize=(6, 5))
ax = fig.gca()

fig = bmap.fit_fermi_edge(hnuminphi_guess=32, background_guess=1e5,
                          integrated_weight_guess=1.5e6, angle_min=-10,
                          angle_max=10, ekin_min=31.96, ekin_max=32.1,
                          ax=ax, show=True, fig_close=True, title='Fermi edge fit')

print('The optimised hnu - Phi=' + f'{bmap.hnuminphi:.4f}' + ' +/- '
      + f'{1.96 * bmap.hnuminphi_std:.5f}' + ' eV.')


angle_min = 0.1
angle_max = 1e6
en_val = 0
energy_range = [-0.3, 0.05]

mdcs = xarpes.MDCs(*bmap.mdc_set(angle_min, angle_max, energy_value=en_val))

fig = plt.figure(figsize=(6, 5))
ax = fig.gca()

fig = mdcs.plot(ax=ax)

# Subsequent plot customization
# ax.set_xlim([2, 10])
# ax.set_ylim([0, 10000])

# change = xarpes.SpectralLinear(amplitude=500, peak=7.5, broadening=0.01,
#                                name='Linear_test', index='1')

# change.broadening = 0.02

# print(change.broadening)


angle_min = 0
angle_max = 1e6
en_val = 0

mdc = xarpes.MDCs(*bmap.mdc_set(angle_min, angle_max, energy_value=en_val))
new_range = mdc.angles

line1 = xarpes.SpectralLinear(amplitude=500, peak=7.5, broadening=0.01,
                              name='Linear test', index='1')
line2 = xarpes.SpectralLinear(amplitude=600, peak=5.5, broadening=0.02,
                              name='Linear test', index='2')

fig = plt.figure(figsize=(7, 5))
ax = fig.gca()

line1.plot(angle_range=new_range, angle_resolution=0.1, ax=ax, show=False,
           fig_close=False)
line2.plot(angle_range=new_range, angle_resolution=0.1, ax=ax, show=False,
           fig_close=False)
fig = mdc.plot(ax=ax)


fig = plt.figure(figsize=(8, 6))
ax = fig.gca()

guess_dists = xarpes.CreateDistributions([
xarpes.Linear(offset=3.0e3, slope=-100),
xarpes.SpectralLinear(amplitude=450, peak=7.4, broadening=0.012,
                      name='Linear_test', index='1'),
xarpes.SpectralQuadratic(amplitude=20, peak=4.5, center_wavevector=0,
                            broadening=0.005, name='Quadratic_test', index='1')
])

fig = mdc.visualize_guess(distributions=guess_dists, ax=ax, show=True)


fig = plt.figure(figsize=(8, 6))
ax = fig.gca()

fig, new_distributions, covariance_matrix = mdc.fit(
     distributions=guess_dists, ax=ax, show=True)

angle_min = 0
angle_max = 1e6

energy_range = [-0.1, 0.01]

mdcs = xarpes.MDCs(*bmap.mdc_set(angle_min, angle_max, energy_range=energy_range))

guess_dists = xarpes.CreateDistributions([
xarpes.Linear(offset=3.0e3, slope=-100),
xarpes.SpectralLinear(amplitude=450, peak=7.4, broadening=0.012,
                      name='Linear_test', index='1'),
# xarpes.SpectralQuadratic(amplitude=20, peak=5.5, center_wavevector=0,
#    broadening=0.005, name='Quadratic_test', index='1')
])


fig = plt.figure(figsize=(8, 6))
ax = fig.gca()

fig = mdcs.visualize_guess(distributions=guess_dists, energy_value=0, ax=ax)


fig = plt.figure(figsize=(8, 6))
ax = fig.gca()

fig, new_distributions, covariance_matrix = mdcs.fit(
     distributions=guess_dists, energy_value=0.001, ax=ax)


fig = plt.figure(figsize=(8, 6))
ax = fig.gca()

energy_range = [-0.25, 0.01]
energy_value = 0.01

mdcs = xarpes.MDCs(*bmap.mdc_set(angle_min, angle_max, energy_range=energy_range))

fig = mdcs.fit_selection(distributions=guess_dists, ax=ax)

# from importlib import reload
# import xarpes
# reload(xarpes)       

self_energy = xarpes.SelfEnergy(*mdcs.expose_parameters(select_label='Linear_test_1', fermi_velocity=0.1))

# print(self_energy.ekin_range)
# print(self_energy.peak_positions)










angle_min2 = -1e6
angle_max2 = 0

mdc2 = xarpes.MDCs(*bmap.mdc_set(angle_min2, angle_max2, energy_range=energy_range))

guess_dists2 = xarpes.CreateDistributions([
xarpes.Linear(offset=2.0e3, slope=100),
xarpes.SpectralLinear(amplitude=450, peak=-7.25, broadening=0.01,
                      name='Linear_left', index='1'),
])

fig = plt.figure(figsize=(8, 6))
ax = fig.gca()

fig = mdc2.visualize_guess(distributions=guess_dists2, energy_value=0, ax=ax)


# fig = plt.figure(figsize=(8, 6))
# ax = fig.gca()

fig = mdc2.fit_selection(distributions=guess_dists2, show=False, fig_close=True)

self_left = xarpes.SelfEnergy(*mdc2.expose_parameters(select_label='Linear_left_1', fermi_wavevector=0.1))


fig = plt.figure(figsize=(8, 5))
ax = fig.gca()

ax.plot(self_left.peak_positions, self_left.enel_range, color='tab:red', linewidth=2)

ax.plot(self_energy.peak_positions, self_energy.enel_range, color='tab:blue', linewidth=2)

fig = bmap.plot(abscissa='momentum', ordinate='electron_energy', ax=ax)

# change = xarpes.SpectralLinear(amplitude=500, peak=7.5, broadening=0.01,
#                                name='Linear_test', index='1')

# change.broadening = 0.02

# print(change.broadening)

# angle_min = 0
# angle_max = 1e6

# energy_range = [-0.05, 0]

# mdcs = xarpes.MDCs(*bmap.mdc_set(angle_min, angle_max, energy_range=energy_range))


# line1 = xarpes.SpectralLinear(amplitude=500, peak=7.5, broadening=0.01,
#                               name='Linear test', index='1')
# line2 = xarpes.SpectralLinear(amplitude=600, peak=5.5, broadening=0.02,
#                               name='Linear test', index='2')

# fig = plt.figure(figsize=(7, 5))
# ax = fig.gca()

# # In case the lines are plotted on top, they may exceed the auto scaling
# line1.plot(angle_range=new_range, angle_resolution=0.1, ax=ax, show=False,
#            fig_close=False)
# line2.plot(angle_range=new_range, angle_resolution=0.1, ax=ax, show=False,
#            fig_close=False)
# fig = mdcs.plot(ax=ax)


# fig = plt.figure(figsize=(7, 5))
# ax = fig.gca()

# fig = mdcs.plot(distributions=guess_dists, ax=ax, energy_value=-0.003)
