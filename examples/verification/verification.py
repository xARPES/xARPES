#!/usr/bin/env python3

# # Verification example
# In this example, we investigate the verification example from the xARPES manuscript examples section.

# The notebook also contains an execution of the Bayesian loop with a set of parameters that is "far from" the optimal solution, similar to the supplemental section on the example.

# In the future, functionality will be added to xARPES for users to generate their own mock example, allowing for testing of desired hypotheses. 

import matplotlib as mpl
mpl.use('Qt5Agg')

# Necessary packages
import xarpes
import numpy as np
import matplotlib.pyplot as plt
import os

# Default plot configuration from xarpes.plotting.py
xarpes.plot_settings('default')

script_dir = xarpes.set_script_dir()

dfld = 'data_sets'           # Folder containing the data
flnm = 'artificial_einstein' # Name of the file

angl = np.load(os.path.join(script_dir, dfld, "verification_angles.npy"))
ekns = np.load(os.path.join(script_dir, dfld, "verification_kinergies.npy"))
intn = np.load(os.path.join(script_dir, dfld, "verification_intensities.npy"))

bmap = xarpes.BandMap.from_np_arrays(intensities=intn, angles=angl, ekin=ekns,
        energy_resolution=0.0025, angle_resolution=0.1, temperature=10)


fig = plt.figure(figsize=(6, 5)); ax = fig.gca()

fig = bmap.fit_fermi_edge(hnuminPhi_guess=30, background_guess=1e4,
                          integrated_weight_guess=3e4, angle_min=-6,
                          angle_max=10, ekin_min=29.99, ekin_max=30.02,
                          ax=ax, show=True, fig_close=True,
                          title='Fermi edge fit')

print('The optimised hnu - Phi=' + f'{bmap.hnuminPhi:.4f}' + ' +/- '
      + f'{1.96 * bmap.hnuminPhi_std:.5f}' + ' eV.')


angle_min = 2
angle_max = 10

energy_range = [-0.08, 0.0001]
energy_value = 0.0

k_0 = 0.1

mdcs = xarpes.MDCs(*bmap.mdc_set(angle_min, angle_max, energy_range=energy_range))

guess_dists = xarpes.CreateDistributions([
xarpes.Constant(offset=40),
xarpes.SpectralQuadratic(amplitude=0.25, peak=5.1, broadening=0.0005,
            center_wavevector=k_0, name='Right_branch', index='1')
])

fig = plt.figure(figsize=(8, 6)); ax = fig.gca()

fig = mdcs.visualize_guess(distributions=guess_dists, energy_value=energy_value, ax=ax)

# **Note on interactive figures**
# - The interactive figure might not work inside the Jupyter notebooks, despite our best efforts to ensure stability.
# - As a fallback, the user may switch from "%matplotlib widget" to "%matplotlib qt", after which the figure should pop up in an external window.
# - For some package versions, a static version of the interactive widget may spuriously show up inside other cells. In that case, uncomment the #get_ipython()... line in the first cell for your notebooks.


fig = plt.figure(figsize=(8, 6)); ax = fig.gca()

fig = mdcs.fit_selection(distributions=guess_dists, ax=ax)


fig = plt.figure(figsize=(6, 5)); ax = fig.gca()

self_energy = xarpes.SelfEnergy(*mdcs.expose_parameters(select_label='Right_branch_1', 
                                bare_mass=1.59675179, fermi_wavevector=0.2499758715, side='right'))

fig = self_energy.plot_both(ax=ax, scale='meV')

plt.show()


self_energies = xarpes.CreateSelfEnergies([self_energy])

fig = plt.figure(figsize=(8, 5)); ax = fig.gca()

fig = bmap.plot(abscissa='momentum', ordinate='kinetic_energy', 
                plot_dispersions='domain', 
                self_energies=self_energies, ax=ax)

# In the following cell, we extract the Eliashberg function from the self-energy. The result of the chi2kink fit is plotted during the extraction. Setting "show=False" and "fig_close=True" will prevent the figure from being displayed. Afterwards, the Eliashberg function and model function with the appropriate self-energy methods.


fig, spectrum, model, omega_range, alpha_select = self_energy.extract_a2f(
    omega_min=1.0, omega_max=80, omega_num=250, omega_I=20, omega_M=60, 
    alpha_min=1.5, alpha_num=10, alpha_max=9.5, lambda_el=0.1132858, 
    impurity_magnitude=10.041243, h_n=0.0803366, show=True, fig_close=False)

plt.show()


fig = plt.figure(figsize=(7, 5)); ax = fig.gca()

fig = self_energy.plot_spectra(ax=ax)

plt.show()

# The following plots all of the extracted quantities in a single figure. The default plotting range is taken from the second plotting statement.
# By default, The Eliashberg function is extracted while removing the self-energies for binding energies smaller than the energy resolution. In that case, it is transparent to also eliminate these self-energies from the displayed result.


fig = plt.figure(figsize=(10, 8)); ax1 = fig.add_subplot(111); ax2 = ax1.twinx()

# ax1.set_ylim([0, 0.5]); ax2.set_ylim([0, 40])

self_energy.plot_spectra( ax=ax1, abscissa="reversed", show=False, fig_close=False)
self_energy.plot_both(ax=ax2, scale="meV", resolution_range='applied', show=False, fig_close=False)

# --- Change colours for spectra
a2f_line, model_line = ax1.get_lines()[-2:]
a2f_line.set_color("mediumvioletred")
model_line.set_color("darkgoldenrod"); model_line.set_linestyle("--")

# --- Change colours for self-energy lines
real_line, imag_line = ax2.get_lines()[-2:]
real_line.set_color("tab:blue"); imag_line.set_color("tab:orange")

# Change colours for error bars
real_err, imag_err = ax2.collections[-2:]
real_err.set_color(real_line.get_color()); imag_err.set_color(imag_line.get_color())

# --- Overwrite the legend with a custom legend
for ax in (ax1, ax2): ax.get_legend() and ax.get_legend().remove()
h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, ncol=2)

plt.show()

# In the following cell, we start from the optimal solution. Unsurprisingly, the optimal solution is obtained after just a couple of iterations.

with xarpes.trim_notebook_output(print_lines=10):
    spectrum, model, omega_range, alpha_select, cost, params = self_energy.bayesian_loop(
            omega_min=1.0, omega_max=80, omega_num=250, omega_I=20, omega_M=60,
            alpha_min=3.0, alpha_max=11.0, bare_mass=1.597636665,
            fermi_wavevector=0.2499774217, h_n=0.081811739,
            impurity_magnitude=10.0379498, lambda_el=0.1054932517,
            vary=("impurity_magnitude", "lambda_el", "fermi_wavevector",
                  "bare_mass", "h_n"),
            scale_mb=0.01, scale_imp=0.1, scale_kF=0.001,
            scale_lambda_el=0.1, scale_hn=0.1,
        )

# In the following cell, we start from a much less probable solution, showing that the code can occasionally heavily improve on the solution.

# The optimisation has been tested against Python v3.10.12, NumPy v2.2.6, and SciPy v1.15.3. Other combinations of these packages still have to be tested.

with xarpes.trim_notebook_output(print_lines=10):
    spectrum, model, omega_range, alpha_select, cost, params = self_energy.bayesian_loop(omega_min=1.0,
                omega_max=80, omega_num=250, omega_I=20, omega_M=60,
                alpha_min=3.0, alpha_max=11.0, bare_mass=1.74625, fermi_wavevector=0.250125,
                h_n=0.09, impurity_magnitude=9.1, lambda_el=0.22, sigma_svd=0.1,
                vary=("impurity_magnitude", "lambda_el", "fermi_wavevector", "bare_mass", "h_n"), 
                converge_iters=100, tole=1e-8, scale_mb=0.1, scale_imp=1.0, scale_kF=0.001,
                scale_lambda_el=0.1, scale_hn=0.01, print_lines=10)

# Following the recommended procedure, we perform a final optimisation with very tight criteria, for the purpose of further narrowing down the solution.

# With the tested combination of packages, the result is a tiny bit closer to the true solution for bare_mass, impurity_magnitude, and lambda_el.

with xarpes.trim_notebook_output(print_lines=10):
    spectrum, model, omega_range, alpha_select, cost, params = self_energy.bayesian_loop(omega_min=1.0,
                omega_max=80, omega_num=250, omega_I=20, omega_M=60,
                alpha_min=1.0, alpha_max=9.0, sigma_svd=1e-4,
                bare_mass=1.597636093, fermi_wavevector=0.2499774208, h_n=0.08181151626, 
                impurity_magnitude=10.03795642, lambda_el=0.1054945571,
                vary=("impurity_magnitude", "lambda_el", "fermi_wavevector", "bare_mass", "h_n"), 
                converge_iters=100, tole=1e-8, scale_mb=0.1, scale_imp=0.1, scale_kF=0.01,
                scale_lambda_el=0.1, scale_hn=0.1, print_lines=10)


