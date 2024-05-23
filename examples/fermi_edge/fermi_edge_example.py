#!/usr/bin/env python3

# Fermi edge fitting example
### In this example, we fit the Fermi edge of the Si-intercalated graphene data set, demonstrating at which kinetic energy to set the chemical potential when it has not been determined from an external reference (such as gold).

import xarpes
import numpy as np
import matplotlib.pyplot as plt
from igor2 import binarywave

xarpes.my_plot_settings('default')

dfld = 'data_sets' # Folder containing the data
flnm = 'graphene_raw_101' # Name of the file
extn = '.ibw' # Extension of the file
tmpr = 80 # Data temperature [K]

data = binarywave.load(dfld + '/' + flnm + extn)

intn = data['wave']['wData']

fnum, anum = data['wave']['wave_header']['nDim'][0:2]

fstp, astp = data['wave']['wave_header']['sfA'][0:2]
fmin, amin = data['wave']['wave_header']['sfB'][0:2]

angl = np.linspace(amin, amin + (anum - 1) * astp, anum)
ekin = np.linspace(fmin, fmin + (fnum - 1) * fstp, fnum)

fdir = xarpes.fermi_dirac(hnuminphi=31.7, temperature=20, background=100, integrated_weight=1000, \
        name='test_distribution')

fig = plt.figure(figsize=(6, 5))
ax = fig.gca()

ax.plot(ekin, fdir.convolve(ekin, energy_resolution=0.05))
plt.savefig('fermi_dirac.png')
plt.close()

fig = plt.figure(figsize=(6, 5))
ax = fig.gca()

bmap = xarpes.band_map(intensities=intn, angles=angl, ekin=ekin, energy_resolution=0.01, temperature=80)

fig = bmap.fit_fermi_edge(hnuminphi_guess=32, background_guess=1e5, integrated_weight_guess=1.5e6, \
                          angle_min=-10, angle_max=10, ekin_min=31.9, ekin_max=32.1, \
                          ax=ax, savefig='edge_fit.png', show=True, title='Fermi edge fit')

print('The optimised h nu - phi=' + f'{bmap.hnuminphi:.4f}' + ' eV.')
