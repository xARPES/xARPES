#!/usr/bin/env python3

# Fermi edge fitting example
### In this example, we fit a linear dispersion to one of the graphene bands.

import matplotlib as mpl                                        
mpl.use('Qt5Agg')

import os
import xarpes
import numpy as np
import matplotlib.pyplot as plt
from igor2 import binarywave

xarpes.plot_settings('default')

try:
    # This block checks if the script is running in an IPython environment
    cfg = get_ipython().config
    script_dir = os.getcwd()
except:
    try:
        # This should work if we're running as a standalone script 
        # and __file__ is defined
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        # If __file__ isn't defined, fall back to current working directory
        script_dir = os.getcwd()

dfld = 'data_sets'  # Folder containing the data
flnm = 'graphene_raw_101'  # Name of the file
extn = '.ibw'  # Extension of the file
tmpr = 80  # Data temperature [K]

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
                       energy_resolution=0.01, angular_resolution=0.1,
                       temperature=80)

fig = bmap.fit_fermi_edge(hnuminphi_guess=32, background_guess=1e5,
                          integrated_weight_guess=1.5e6, angle_min=-10,
                          angle_max=10, ekin_min=31.9, ekin_max=32.1,
                          ax=ax, show=True,
                          title='Fermi edge fit')


print('The optimised h nu - phi=' + f'{bmap.hnuminphi:.4f}' + ' +/- '
      + f'{bmap.hnuminphi_std:.4f}' + ' eV.')

fig = bmap.plot_band_map()

angle_min = 1
angle_max = 10
energy_value = 32.0107

mdc = xarpes.MDCs(*bmap.slice(angle_min, angle_max, energy_value))
fig = mdc.plot()

distributions = [xarpes.constant(offset=10),
                 xarpes.linear(offset=5, slope=-2, name='something_else')]

distributions = [
# xarpes.constant(offset=5e3),
# xarpes.spectral_quadratic(amp=450, cen=-7.0, gam=0.002, kmin=-0.306, ind="1"),
# xarpes.spectral_linear(amp=200, cen=-32, gam=0.005, ind="1"),
]


