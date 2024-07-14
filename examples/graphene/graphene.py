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

xarpes.plot_settings('default')

script_dir = xarpes.set_script_dir()

dfld = 'data_sets'  # Folder containing the data
flnm = 'graphene_raw_101'  # Name of the file
extn = '.ibw'  # Extension of the file

# These three packages no longer have to be loaded once data is generated
# inside the band map class
import numpy as np 
from igor2 import binarywave
import os

data_file_path = os.path.join(script_dir, dfld, flnm + extn)
data = binarywave.load(data_file_path)

intn = data['wave']['wData']

fnum, anum = data['wave']['wave_header']['nDim'][0:2]
fstp, astp = data['wave']['wave_header']['sfA'][0:2]
fmin, amin = data['wave']['wave_header']['sfB'][0:2]

angl = np.linspace(amin, amin + (anum - 1) * astp, anum)
ekin = np.linspace(fmin, fmin + (fnum - 1) * fstp, fnum)

fig = plt.figure(figsize=(6, 5))
ax = fig.gca()

bmap = xarpes.band_map(intensities=intn, angles=angl, ekin=ekin,
                       energy_resolution=0.01, angle_resolution=0.1,
                       temperature=80)

fig = bmap.fit_fermi_edge(hnuminphi_guess=32, background_guess=1e5,
                          integrated_weight_guess=1.5e6, angle_min=-10,
                          angle_max=10, ekin_min=31.9, ekin_max=32.1,
                          ax=ax, show=True, title='Fermi edge fit')

print('The optimised h nu - phi=' + f'{bmap.hnuminphi:.4f}' + ' +/- '
      + f'{bmap.hnuminphi_std:.4f}' + ' eV.')

fig = bmap.plot()

angle_min = 0.1
angle_max = 1e6
en_val = 0
energy_range = [-0.3, 0.05]

mdcs = xarpes.MDCs(*bmap.slicing(angle_min, angle_max, energy_value=en_val))

fig = plt.figure(figsize=(5, 4))
ax = fig.gca()

fig = mdcs.plot(ax=ax, show=True)

angle_min = 0
angle_max = 1e6
en_val = 0

mdcs = xarpes.MDCs(*bmap.slicing(angle_min, angle_max, energy_value=en_val))
new_range = mdcs.angles

fig = plt.figure(figsize=(7, 5))
ax = fig.gca()

line1 = xarpes.spectral_linear(amplitude=500, peak=7.5, broadening=0.01, name="Linear test", index="1")
line2 = xarpes.spectral_linear(amplitude=600, peak=5.5, broadening=0.02, name="Linear test", index="2")

line1.plot(angle_range=new_range, angle_resolution=0.1, ax=ax, show=False, fig_close=False)
line2.plot(angle_range=new_range, angle_resolution=0.1, ax=ax, show=False, fig_close=False)
fig = mdcs.plot(ax=ax)

fig = plt.figure(figsize=(8, 6))
ax = fig.gca()

guess_dists = xarpes.create_distributions([
xarpes.linear(offset=3.0e3, slope=-100),
xarpes.spectral_linear(amplitude=450, peak=7.4, broadening=0.012, name='Linear_test', index='1'),
xarpes.spectral_linear(amplitude=60, peak=4.2, broadening=0.018, name='Linear_test', index='2'),
# xarpes.spectral_quadratic(amplitude=20, peak=4.5, center_wavevector=0, broadening=0.005, 
#                        side='right', name='Quadratic_test', index='1')
])

fig = mdcs.visualize_guess(distributions=guess_dists, ax=ax, show=True)

fig = plt.figure(figsize=(8, 6))
ax = fig.gca()

fig, new_distributions, covariance_matrix = mdcs.fit(distributions=guess_dists, ax=ax, show=True)

change = xarpes.spectral_linear(amplitude=500, peak=7.5, broadening=0.01, name="Linear_test", index="1")

change.broadening = 0.02

print(change.broadening)
