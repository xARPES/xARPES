# Copyright (C) 2025 xARPES Developers
# This program is free software under the terms of the GNU GPLv3 license.

# get_ax_fig_plt and add_fig_kwargs originate from pymatgen/util/plotting.py.
# Copyright (C) 2011-2024 Shyue Ping Ong and the pymatgen Development Team
# Pymatgen is released under the MIT License.

# See also abipy/tools/plotting.py.
# Copyright (C) 2021 Matteo Giantomassi and the AbiPy Group
# AbiPy is free software under the terms of the GNU GPLv2 license.

"""File containing the MDCs class."""

import numpy as np
from .plotting import get_ax_fig_plt, add_fig_kwargs
from .functions import extend_function
from .constants import KILO

class MDCs:
    r"""
    Container for momentum distribution curves (MDCs) and their fits.

    This class stores the MDC intensity maps, angular and energy grids, and
    the aggregated fit results produced by :meth:`fit_selection`.

    Parameters
    ----------
    intensities : ndarray
        MDC intensity data. Typically a 2D array with shape
        ``(n_energy, n_angle)`` or a 1D array for a single curve.
    angles : ndarray
        Angular grid corresponding to the MDCs [degrees].
    angle_resolution : float
        Angular step size or effective angular resolution [degrees].
    enel : ndarray or float
        Electron binding energies of the MDC slices [eV].
        Can be a scalar for a single MDC.
    hnuminPhi : float
        Photon energy minus work function, used to convert ``enel`` to
        kinetic energy [eV].

    Attributes
    ----------
    intensities : ndarray
        MDC intensity data (same object as passed to the constructor).
    angles : ndarray
        Angular grid [degrees].
    angle_resolution : float
        Angular step size or resolution [degrees].
    enel : ndarray or float
        Electron binding energies [eV], as given at construction.
    ekin : ndarray or float
        Kinetic energies [eV], computed as ``enel + hnuminPhi``.
    hnuminPhi : float
        Photon energy minus work function [eV].
    ekin_range : ndarray
        Kinetic-energy values of the slices that were actually fitted.
        Set by :meth:`fit_selection`.
    individual_properties : dict
        Nested mapping of fitted parameters and their uncertainties for each
        component and each energy slice. Populated by :meth:`fit_selection`.

    Notes
    -----
    After calling :meth:`fit_selection`, :attr:`individual_properties` has the
    structure::

        {
            label: {
                class_name: {
                    'label': label,
                    '_class': class_name,
                    param:       [values per energy slice],
                    param_sigma: [1σ per slice or None],
                    ...
                }
            }
        }

    where ``param`` is typically one of ``'offset'``, ``'slope'``,
    ``'amplitude'``, ``'peak'``, ``'broadening'``, and ``param_sigma`` stores
    the corresponding uncertainty for each slice.
    
    """

    def __init__(self, intensities, angles, angle_resolution, enel, hnuminPhi):
        # Core input data (read-only)
        self._intensities = intensities
        self._angles = angles
        self._angle_resolution = angle_resolution
        self._enel = enel
        self._hnuminPhi = hnuminPhi

        # Derived attributes (populated by fit_selection)
        self._ekin_range = None
        self._individual_properties = None  # combined values + sigmas

    # -------------------- Immutable physics inputs --------------------

    @property
    def angles(self):
        """Angular axis for the MDCs."""
        return self._angles

    @property
    def angle_resolution(self):
        """Angular step size (float)."""
        return self._angle_resolution

    @property
    def enel(self):
        """Photoelectron binding energies (array-like). Read-only."""
        return self._enel

    @enel.setter
    def enel(self, _):
        raise AttributeError("`enel` is read-only; set it via the constructor.")

    @property
    def hnuminPhi(self):
        """Work-function/photon-energy offset. Read-only."""
        return self._hnuminPhi

    @hnuminPhi.setter
    def hnuminPhi(self, _):
        raise AttributeError("`hnuminPhi` is read-only; set it via the constructor.")

    @property
    def ekin(self):
        """Kinetic energy array: enel + hnuminPhi (computed on the fly)."""
        return self._enel + self._hnuminPhi

    @ekin.setter
    def ekin(self, _):
        raise AttributeError("`ekin` is derived and read-only.")

    # -------------------- Data arrays --------------------

    @property
    def intensities(self):
        """2D or 3D intensity map (energy × angle)."""
        return self._intensities

    @intensities.setter
    def intensities(self, x):
        self._intensities = x

    # -------------------- Results populated by fit_selection --------------------

    @property
    def ekin_range(self):
        """Kinetic-energy slices that were fitted."""
        if self._ekin_range is None:
            raise AttributeError("`ekin_range` not yet set. Run `.fit_selection()` first.")
        return self._ekin_range

    @property
    def individual_properties(self):
        """
        Aggregated fitted parameter values and uncertainties per component.

        Returns
        -------
        dict
            Nested mapping::

                {
                    label: {
                        class_name: {
                            'label': label,
                            '_class': class_name,
                            <param>:        [values per slice],
                            <param>_sigma:  [1σ per slice or None],
                            ...
                        }
                    }
                }
        """
        if self._individual_properties is None:
            raise AttributeError(
                "`individual_properties` not yet set. Run `.fit_selection()` first."
            )
        return self._individual_properties

    def energy_check(self, energy_value):
        r"""
        """
        if np.isscalar(self.ekin):
            if  energy_value is not None:
                raise ValueError("This dataset contains only one " \
                "momentum-distribution curve; do not provide energy_value.")
            else: 
                kinergy = self.ekin
                counts = self.intensities
        else:
            if energy_value is None:
                raise ValueError("This dataset contains multiple " \
                "momentum-distribution curves. Please provide an energy_value "
                "for which to plot the MDCs.")
            else:
                energy_index = np.abs(self.enel - energy_value).argmin()
                kinergy = self.ekin[energy_index]
                counts = self.intensities[energy_index, :]

                if not (self.enel.min() <= energy_value <= self.enel.max()):
                    raise ValueError(
                        f"Selected energy_value={energy_value:.3f} "
                        f"is outside the available energy range "
                        f"({self.enel.min():.3f} – {self.enel.max():.3f}) "
                        "of the MDC collection."
                    )

        return counts, kinergy


    def plot(self, energy_value=None, energy_range=None, ax=None, **kwargs):
        """
        Interactive or static plot with optional slider and full wrapper 
        support. Behavior consistent with Jupyter and CLI based on show / 
        fig_close.
        """
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider
        import string
        import sys
        import warnings

        # Wrapper kwargs
        title = kwargs.pop("title", None)
        savefig = kwargs.pop("savefig", None)
        show = kwargs.pop("show", True)
        fig_close = kwargs.pop("fig_close", False)
        tight_layout = kwargs.pop("tight_layout", False)
        ax_grid = kwargs.pop("ax_grid", None)
        ax_annotate = kwargs.pop("ax_annotate", False)
        size_kwargs = kwargs.pop("size_kwargs", None)

        if energy_value is not None and energy_range is not None:
            raise ValueError(
                "Provide at most energy_value or energy_range, not both.")

        ax, fig, plt = get_ax_fig_plt(ax=ax)

        angles = self.angles
        energies = self.enel

        if np.isscalar(energies):
            if energy_value is not None or energy_range is not None:
                raise ValueError(
                    "This dataset contains only one momentum-distribution "
                    "curve; do not provide energy_value or energy_range."
                )

            intensities = self.intensities
            ax.scatter(angles, intensities, label="Data")
            ax.set_title(f"Energy slice: {energies * KILO:.3f} meV")

            # --- y-only autoscale, preserve x ---
            x0, x1 = ax.get_xlim()                 # keep current x-range
            ax.relim(visible_only=True)            # recompute data limits
            ax.autoscale_view(scalex=False, scaley=True)
            ax.set_xlim(x0, x1)                    # restore x (belt-and-suspenders)

        else:
            if (energy_value is not None) and (energy_range is not None):
                raise ValueError("Provide either energy_value or energy_range, not both.")

            emin, emax = energies.min(), energies.max()

            # ---- Single-slice path (no slider) ----
            if energy_value is not None:
                if energy_value < emin or energy_value > emax:
                    raise ValueError(
                        f"Requested energy_value {energy_value:.3f} eV is "
                        f"outside the available energy range "
                        f"[{emin:.3f}, {emax:.3f}] eV."
                    )
                idx = int(np.abs(energies - energy_value).argmin())
                intensities = self.intensities[idx]
                ax.scatter(angles, intensities, label="Data")
                ax.set_title(f"Energy slice: {energies[idx] * KILO:.3f} meV")

                # --- y-only autoscale, preserve x ---
                x0, x1 = ax.get_xlim()                 # keep current x-range
                ax.relim(visible_only=True)            # recompute data limits
                ax.autoscale_view(scalex=False, scaley=True)
                ax.set_xlim(x0, x1)                    # restore x (belt-and-suspenders)

            # ---- Multi-slice path (slider) ----
            else:
                if energy_range is not None:
                    e_min, e_max = energy_range
                    mask = (energies >= e_min) & (energies <= e_max)
                else:
                    mask = np.ones_like(energies, dtype=bool)

                indices = np.where(mask)[0]
                if len(indices) == 0:
                    raise ValueError("No energies found in the specified selection.")

                intensities = self.intensities[indices]

                fig.subplots_adjust(bottom=0.25)
                idx = 0
                scatter = ax.scatter(angles, intensities[idx], label="Data")
                ax.set_title(f"Energy slice: "
                             f"{energies[indices[idx]] * KILO:.3f} meV")

                # Suppress single-point slider warning (when len(indices) == 1)
                warnings.filterwarnings(
                    "ignore",
                    message="Attempting to set identical left == right",
                    category=UserWarning
                )

                slider_ax = fig.add_axes([0.2, 0.08, 0.6, 0.04])
                slider = Slider(
                    slider_ax, "Index", 0, len(indices) - 1,
                    valinit=idx, valstep=1
                )

                def update(val):
                    i = int(slider.val)
                    yi = intensities[i]

                    scatter.set_offsets(np.c_[angles, yi])

                    x0, x1 = ax.get_xlim()

                    yv = np.asarray(yi, dtype=float).ravel()
                    mask = np.isfinite(yv)
                    if mask.any():
                        y_min = float(yv[mask].min())
                        y_max = float(yv[mask].max())
                        span  = y_max - y_min
                        frac  = plt.rcParams['axes.ymargin']

                        if span <= 0 or not np.isfinite(span):
                            scale = max(abs(y_max), 1.0)
                            pad = frac * scale
                        else:
                            pad = frac * span

                        ax.set_ylim(y_min - pad, y_max + pad)

                    # Keep x unchanged
                    ax.set_xlim(x0, x1)

                    # Update title and redraw
                    ax.set_title(f"Energy slice: "
                                 f"{energies[indices[i]] * KILO:.3f} meV")
                    fig.canvas.draw_idle()

                slider.on_changed(update)
                self._slider = slider
                self._line = scatter

        ax.set_xlabel("Angle (°)")
        ax.set_ylabel("Counts (-)")
        ax.legend()
        self._fig = fig

        if size_kwargs:
            fig.set_size_inches(size_kwargs.pop("w"),
                size_kwargs.pop("h"), **size_kwargs)
        if title:
            fig.suptitle(title)
        if tight_layout:
            fig.tight_layout()
        if savefig:
            fig.savefig(savefig)
        if ax_grid is not None:
            for axis in fig.axes:
                axis.grid(bool(ax_grid))
        if ax_annotate:
            tags = string.ascii_lowercase
            for i, axis in enumerate(fig.axes):
                axis.annotate(f"({tags[i]})", xy=(0.05, 0.95),
                    xycoords="axes fraction")

        is_interactive = hasattr(sys, 'ps1') or 'ipykernel' in sys.modules
        is_cli = not is_interactive

        if show:
            if is_cli:
                plt.show()
        if fig_close:
            plt.close(fig)

        if not show and (fig_close or is_cli):
            return None
        return fig


    @add_fig_kwargs
    def visualize_guess(self, distributions, energy_value=None,
                        matrix_element=None, matrix_args=None,
                        ax=None, **kwargs):
        r"""
        """
        
        counts, kinergy = self.energy_check(energy_value)

        ax, fig, plt = get_ax_fig_plt(ax=ax)

        ax.set_xlabel('Angle ($\\degree$)')
        ax.set_ylabel('Counts (-)')
        ax.set_title(f"Energy slice: "
                     f"{(kinergy - self.hnuminPhi) * KILO:.3f} meV")
        ax.scatter(self.angles, counts, label='Data')

        final_result = self._merge_and_plot(ax=ax, 
            distributions=distributions, kinetic_energy=kinergy,
            matrix_element=matrix_element,
            matrix_args=dict(matrix_args) if matrix_args else None,
            plot_individual=True,
        )

        residual = counts - final_result
        ax.scatter(self.angles, residual, label='Residual')
        ax.legend()

        return fig
    

    def fit_selection(self, distributions, energy_value=None, energy_range=None,
            matrix_element=None, matrix_args=None, ax=None, **kwargs):
        r"""
        """
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider
        from copy import deepcopy
        import string
        import sys
        import warnings
        from lmfit import Minimizer
        from scipy.ndimage import gaussian_filter
        from .functions import construct_parameters, build_distributions, \
            residual, resolve_param_name
        
        # Wrapper kwargs
        title = kwargs.pop("title", None)
        savefig = kwargs.pop("savefig", None)
        show = kwargs.pop("show", True)
        fig_close = kwargs.pop("fig_close", False)
        tight_layout = kwargs.pop("tight_layout", False)
        ax_grid = kwargs.pop("ax_grid", None)
        ax_annotate = kwargs.pop("ax_annotate", False)
        size_kwargs = kwargs.pop("size_kwargs", None)

        ax, fig, plt = get_ax_fig_plt(ax=ax)

        energies = self.enel
        new_distributions = deepcopy(distributions)

        if energy_value is not None and energy_range is not None:
            raise ValueError(
                "Provide at most energy_value or energy_range, not both.")

        if np.isscalar(energies):
            if energy_value is not None or energy_range is not None:
                raise ValueError(
                    "This dataset contains only one momentum-distribution "
                    "curve; do not provide energy_value or energy_range."
                )
            kinergies = np.atleast_1d(self.ekin)
            intensities = np.atleast_2d(self.intensities)

        else:
            if energy_value is not None:
                if (energy_value < energies.min() or energy_value > energies.max()):
                    raise ValueError( f"Requested energy_value {energy_value:.3f} eV is " 
                                     f"outside the available energy range " 
                                     f"[{energies.min():.3f}, {energies.max():.3f}] eV." )
                idx = np.abs(energies - energy_value).argmin()
                indices = np.atleast_1d(idx)
                kinergies = self.ekin[indices]
                intensities = self.intensities[indices, :]

            elif energy_range is not None:
                e_min, e_max = energy_range
                indices = np.where((energies >= e_min) & (energies <= e_max))[0]
                if len(indices) == 0:
                    raise ValueError("No energies found in the specified energy_range.")
                kinergies = self.ekin[indices]
                intensities = self.intensities[indices, :]

            else: # Without specifying a range, all MDCs are plotted
                kinergies = self.ekin
                intensities = self.intensities

        # Final shape guard
        kinergies = np.atleast_1d(kinergies)
        intensities = np.atleast_2d(intensities)

        all_final_results = []
        all_residuals = []
        all_individual_results = [] # List of (n_individuals, n_angles)

        aggregated_properties = {}

        # map class_name -> parameter names to extract
        param_spec = {
            'Constant': ('offset',),
            'Linear': ('offset', 'slope'),
            'SpectralLinear': ('amplitude', 'peak', 'broadening'),
            'SpectralQuadratic': ('amplitude', 'peak', 'broadening'),
        }

        order = np.argsort(kinergies)[::-1]
        for idx in order:
            kinergy   = kinergies[idx]
            intensity = intensities[idx]
            if matrix_element is not None:
                parameters, element_names = construct_parameters(
                    new_distributions, matrix_args)
                new_distributions = build_distributions(new_distributions, parameters)
                mini = Minimizer(
                    residual, parameters,
                    fcn_args=(self.angles, intensity, self.angle_resolution,
                              new_distributions, kinergy, self.hnuminPhi,
                              matrix_element, element_names)
                )
            else:
                parameters = construct_parameters(new_distributions)
                new_distributions = build_distributions(new_distributions, parameters)
                mini = Minimizer(
                    residual, parameters,
                    fcn_args=(self.angles, intensity, self.angle_resolution,
                              new_distributions, kinergy, self.hnuminPhi)
                )

            outcome = mini.minimize('least_squares')

            pcov = outcome.covar

            var_names = getattr(outcome, 'var_names', None)
            if not var_names:
                var_names = [n for n, p in outcome.params.items() if p.vary]
            var_idx = {n: i for i, n in enumerate(var_names)}

            param_sigma_full = {}
            for name, par in outcome.params.items():
                sigma = None
                if pcov is not None and name in var_idx:
                    d = pcov[var_idx[name], var_idx[name]]
                    if np.isfinite(d) and d >= 0:
                        sigma = float(np.sqrt(d))
                if sigma is None:
                    s = getattr(par, 'stderr', None)
                    sigma = float(s) if s is not None else None
                param_sigma_full[name] = sigma

            # Rebuild the *fitted* distributions from optimized params
            fitted_distributions = build_distributions(new_distributions, outcome.params)

            # If using a matrix element, extract slice-specific args from the fit
            if matrix_element is not None:
                new_matrix_args = {key: outcome.params[key].value for key in matrix_args}
            else:
                new_matrix_args = None

            # individual curves (smoothed, cropped) and final sum (no plotting here)
            extend, step, numb = extend_function(self.angles, self.angle_resolution)

            total_result_ext = np.zeros_like(extend)
            indiv_rows = [] # (n_individuals, n_angles)
            individual_labels = []

            for dist in fitted_distributions:
                # evaluate each component on the extended grid
                if getattr(dist, 'class_name', None) == 'SpectralQuadratic':
                    if (getattr(dist, 'center_angle', None) is not None) and (
                        kinergy is None or self.hnuminPhi is None
                    ):
                        raise ValueError(
                            'Spectral quadratic function is defined in terms '
                            'of a center angle. Please provide a kinetic energy '
                            'and hnuminPhi.'
                        )
                    extended_result = dist.evaluate(extend, kinergy, self.hnuminPhi)
                else:
                    extended_result = dist.evaluate(extend)

                if matrix_element is not None and hasattr(dist, 'index'):
                    args = new_matrix_args or {}
                    extended_result *= matrix_element(extend, **args)

                total_result_ext += extended_result

                # smoothed & cropped individual
                individual_curve = gaussian_filter(extended_result, sigma=step)[
                    numb:-numb if numb else None
                ]
                indiv_rows.append(np.asarray(individual_curve))

                # label
                label = getattr(dist, 'label', str(dist))
                individual_labels.append(label)

                # ---- collect parameters for this distribution
                # (Aggregated over slices)
                cls = getattr(dist, 'class_name', None)
                wanted = param_spec.get(cls, ())

                # ensure dicts exist
                label_bucket = aggregated_properties.setdefault(label, {})
                class_bucket = label_bucket.setdefault(
                    cls, {'label': label, '_class': cls}
                )

                # store center_wavevector (scalar) for SpectralQuadratic
                if (
                    cls == 'SpectralQuadratic'
                    and hasattr(dist, 'center_wavevector')
                ):
                    class_bucket.setdefault(
                        'center_wavevector', dist.center_wavevector
                    )

                # ensure keys for both values and sigmas
                for pname in wanted:
                    class_bucket.setdefault(pname, [])
                    class_bucket.setdefault(f"{pname}_sigma", [])

                # append values and sigmas in the order of slices
                for pname in wanted:
                    param_key = resolve_param_name(outcome.params, label, pname)

                    if param_key is not None and param_key in outcome.params:
                        class_bucket[pname].append(outcome.params[param_key].value)
                        class_bucket[f"{pname}_sigma"].append(param_sigma_full.get(param_key, None))
                    else:
                        # Not fitted in this slice → keep the value if present on the dist, sigma=None
                        class_bucket[pname].append(getattr(dist, pname, None))
                        class_bucket[f"{pname}_sigma"].append(None)

            # final (sum) curve, smoothed & cropped
            final_result_i = gaussian_filter(total_result_ext, sigma=step)[
                numb:-numb if numb else None]
            final_result_i = np.asarray(final_result_i)

            # Residual for this slice
            residual_i = np.asarray(intensity) - final_result_i

            # Store per-slice results
            all_final_results.append(final_result_i)
            all_residuals.append(residual_i)
            all_individual_results.append(np.vstack(indiv_rows))

        # --- after the reversed-order loop, restore original (ascending) order ---
        inverse_order = np.argsort(np.argsort(kinergies)[::-1])

        # Reorder per-slice arrays/lists computed in the loop
        all_final_results[:]      = [all_final_results[i]      for i in inverse_order]
        all_residuals[:]          = [all_residuals[i]          for i in inverse_order]
        all_individual_results[:] = [all_individual_results[i] for i in inverse_order]

        # Reorder all per-slice lists in aggregated_properties
        for label_dict in aggregated_properties.values():
            for cls_dict in label_dict.values():
                for key, val in cls_dict.items():
                    if isinstance(val, list) and len(val) == len(kinergies):
                        cls_dict[key] = [val[i] for i in inverse_order]

        self._ekin_range = kinergies
        self._individual_properties = aggregated_properties

        if np.isscalar(energies):
            # One slice only: plot MDC, Fit, Residual, and Individuals
            ydata = np.asarray(intensities).squeeze()
            yfit  = np.asarray(all_final_results[0]).squeeze()
            yres  = np.asarray(all_residuals[0]).squeeze()
            yind  = np.asarray(all_individual_results[0])

            ax.scatter(self.angles, ydata, label="Data")
            # plot individuals with their labels
            for j, lab in enumerate(individual_labels or []):
                ax.plot(self.angles, yind[j], label=str(lab))
            ax.plot(self.angles, yfit, label="Fit")
            ax.scatter(self.angles, yres, label="Residual")

            ax.set_title(f"Energy slice: {energies * KILO:.3f} meV")
            ax.relim()          # recompute data limits from all artists
            ax.autoscale_view() # apply autoscaling + axes.ymargin padding

        else:
            if energy_value is not None:
                _idx = int(np.abs(energies - energy_value).argmin())
                energies_sel = np.atleast_1d(energies[_idx])
            elif energy_range is not None:
                e_min, e_max = energy_range
                energies_sel = energies[(energies >= e_min) 
                                        & (energies <= e_max)]
            else:
                energies_sel = energies

            # Number of slices must match
            n_slices = len(all_final_results)
            assert intensities.shape[0] == n_slices == len(all_residuals) \
                == len(all_individual_results), (f"Mismatch: data \
                {intensities.shape[0]}, fits {len(all_final_results)}, "
                f"residuals {len(all_residuals)}, \
                individuals {len(all_individual_results)}."
            )
            n_individuals = all_individual_results[0].shape[0] \
                if n_slices else 0

            fig.subplots_adjust(bottom=0.25)
            idx = 0

            # Initial draw (MDC + Individuals + Fit + Residual) at slice 0
            scatter = ax.scatter(self.angles, intensities[idx], label="Data")

            individual_lines = []
            if n_individuals:
                for j in range(n_individuals):
                    if individual_labels and j < len(individual_labels):
                        label = str(individual_labels[j])
                    else:
                        label = f"Comp {j}"

                    yvals = all_individual_results[idx][j]
                    line, = ax.plot(self.angles, yvals, label=label)
                    individual_lines.append(line)

            result_line, = ax.plot(self.angles, all_final_results[idx], 
                                   label="Fit")
            resid_scatter = ax.scatter(self.angles, all_residuals[idx], 
                                       label="Residual")

            # Title + limits (use only the currently shown slice)
            ax.set_title(f"Energy slice: {energies_sel[idx] * KILO:.3f} meV")
            ax.relim()             # recompute data limits from all artists
            ax.autoscale_view()    # apply autoscaling + axes.ymargin padding

            # Suppress warning when a single MDC is plotted
            warnings.filterwarnings(
                "ignore",
                message="Attempting to set identical left == right",
                category=UserWarning
            )

            # Slider over slice index (0..n_slices-1)
            slider_ax = fig.add_axes([0.2, 0.08, 0.6, 0.04])
            slider = Slider(
                slider_ax, "Index", 0, n_slices - 1,
                valinit=idx, valstep=1
            )

            def update(val):
                i = int(slider.val)
                # Update MDC points
                scatter.set_offsets(np.c_[self.angles, intensities[i]])

                # Update individuals
                if n_individuals:
                    Yi = all_individual_results[i] # (n_individuals, n_angles)
                    for j, ln in enumerate(individual_lines):
                        ln.set_ydata(Yi[j])

                # Update fit and residual
                result_line.set_ydata(all_final_results[i])
                resid_scatter.set_offsets(np.c_[self.angles, all_residuals[i]])

                ax.relim()
                ax.autoscale_view()

                # Update title and redraw
                ax.set_title(f"Energy slice: "
                             f"{energies_sel[i] * KILO:.3f} meV")
                fig.canvas.draw_idle()

            slider.on_changed(update)
            self._slider = slider
            self._line = scatter
            self._individual_lines = individual_lines
            self._result_line = result_line
            self._resid_scatter = resid_scatter

        ax.set_xlabel("Angle (°)")
        ax.set_ylabel("Counts (-)")
        ax.legend()
        self._fig = fig

        if size_kwargs:
            fig.set_size_inches(size_kwargs.pop("w"),
                size_kwargs.pop("h"), **size_kwargs)
        if title:
            fig.suptitle(title)
        if tight_layout:
            fig.tight_layout()
        if savefig:
            fig.savefig(savefig)
        if ax_grid is not None:
            for axis in fig.axes:
                axis.grid(bool(ax_grid))
        if ax_annotate:
            tags = string.ascii_lowercase
            for i, axis in enumerate(fig.axes):
                axis.annotate(f"({tags[i]})", xy=(0.05, 0.95),
                    xycoords="axes fraction")

        is_interactive = hasattr(sys, 'ps1') or 'ipykernel' in sys.modules
        is_cli = not is_interactive

        if show:
            if is_cli:
                plt.show()
        if fig_close:
            plt.close(fig)

        if not show and (fig_close or is_cli):
            return None
        return fig
    

    @add_fig_kwargs
    def fit(self, distributions, energy_value=None, matrix_element=None, 
            matrix_args=None, ax=None, **kwargs):
        r"""
        """      
        from copy import deepcopy
        from lmfit import Minimizer
        from .functions import construct_parameters, build_distributions, \
            residual
        
        counts, kinergy = self.energy_check(energy_value)

        ax, fig, plt = get_ax_fig_plt(ax=ax)

        ax.set_xlabel('Angle ($\\degree$)')
        ax.set_ylabel('Counts (-)')
        ax.set_title(f"Energy slice: "
                     f"{(kinergy - self.hnuminPhi) * KILO:.3f} meV")
        
        ax.scatter(self.angles, counts, label='Data')

        new_distributions = deepcopy(distributions)

        if matrix_element is not None:
            parameters, element_names = construct_parameters(distributions,
                                                             matrix_args)
            new_distributions = build_distributions(new_distributions, \
                                                    parameters)
            mini = Minimizer(
                residual, parameters,
                fcn_args=(self.angles, counts, self.angle_resolution,
                          new_distributions, kinergy, self.hnuminPhi,
                          matrix_element, element_names))
        else:
            parameters = construct_parameters(distributions)
            new_distributions = build_distributions(new_distributions,
                                                    parameters)
            mini = Minimizer(residual, parameters,
                fcn_args=(self.angles, counts, self.angle_resolution,
                          new_distributions, kinergy, self.hnuminPhi))

        outcome = mini.minimize('least_squares')
        pcov = outcome.covar
        
        # If matrix params were fitted, pass the fitted values to plotting
        if matrix_element is not None:
            new_matrix_args = {key: outcome.params[key].value for key in
                               matrix_args}
        else:
            new_matrix_args = None

        final_result = self._merge_and_plot(ax=ax, 
            distributions=new_distributions, kinetic_energy=kinergy,
            matrix_element=matrix_element, matrix_args=new_matrix_args,
            plot_individual=True)
        
        residual_vals = counts - final_result
        ax.scatter(self.angles, residual_vals, label='Residual')
        ax.legend()
        if matrix_element is not None:
            return fig, new_distributions, pcov, new_matrix_args
        else:
            return fig, new_distributions, pcov
        

    def _merge_and_plot(self, ax, distributions, kinetic_energy,
                        matrix_element=None, matrix_args=None,
                        plot_individual=True):
        r"""
        Evaluate distributions on the extended grid, apply optional matrix
        element, smooth, plot individuals and the summed curve.

        Returns
        -------
        final_result : np.ndarray
            Smoothed, cropped total distribution aligned with self.angles.
        """
        from scipy.ndimage import gaussian_filter

        # Build extended grid
        extend, step, numb = extend_function(self.angles, self.angle_resolution)
        total_result = np.zeros_like(extend)

        for dist in distributions:
            # Special handling for SpectralQuadratic
            if getattr(dist, 'class_name', None) == 'SpectralQuadratic':
                if (getattr(dist, 'center_angle', None) is not None) and (
                    kinetic_energy is None or self.hnuminPhi is None
                ):
                    raise ValueError(
                        'Spectral quadratic function is defined in terms '
                        'of a center angle. Please provide a kinetic energy '
                        'and hnuminPhi.'
                    )
                extended_result = dist.evaluate(extend, kinetic_energy, \
                                                self.hnuminPhi)
            else:
                extended_result = dist.evaluate(extend)

            # Optional matrix element (only for components that advertise an index)
            if matrix_element is not None and hasattr(dist, 'index'):
                args = matrix_args or {}
                extended_result *= matrix_element(extend, **args)

            total_result += extended_result

            if plot_individual and ax:
                individual = gaussian_filter(extended_result, sigma=step)\
                    [numb:-numb if numb else None]
                ax.plot(self.angles, individual, label=getattr(dist, \
                                                        'label', str(dist)))

        # Smoothed, cropped total curve aligned to self.angles
        final_result = gaussian_filter(total_result, sigma=step)[numb:-numb \
                                                            if numb else None]
        if ax:
            ax.plot(self.angles, final_result, label='Distribution sum')

        return final_result
    

    def expose_parameters(self, select_label, fermi_wavevector=None,
                          fermi_velocity=None, bare_mass=None, side=None):
        r"""
        Select and return fitted parameters for a given component label, plus a
        flat export dictionary containing values **and** 1σ uncertainties.

        Parameters
        ----------
        select_label : str
            Label to look for among the fitted distributions.
        fermi_wavevector : float, optional
            Optional Fermi wave vector to include.
        fermi_velocity : float, optional
            Optional Fermi velocity to include.
        bare_mass : float, optional
            Optional bare mass to include (used for SpectralQuadratic
            dispersions).
        side : {'left','right'}, optional
            Optional side selector for SpectralQuadratic dispersions.

        Returns
        -------
        ekin_range : np.ndarray
            Kinetic-energy grid corresponding to the selected label.
        hnuminPhi : float
            Photoelectron work-function offset.
        label : str
            Label of the selected distribution.
        selected_properties : dict or list of dict
            Nested dictionary (or list thereof) containing <param> and
            <param>_sigma arrays. For SpectralQuadratic components, a
            scalar `center_wavevector` is also present.
        exported_parameters : dict
            Flat dictionary of parameters and their uncertainties, plus
            optional Fermi quantities and `side`. For SpectralQuadratic
            components, `center_wavevector` is included and taken directly
            from the fitted distribution.
        """

        if self._ekin_range is None:
            raise AttributeError(
                "ekin_range not yet set. Run `.fit_selection()` first."
            )

        store = getattr(self, "_individual_properties", None)
        if not store or select_label not in store:
            all_labels = (sorted(store.keys())
                          if isinstance(store, dict) else [])
            raise ValueError(
                f"Label '{select_label}' not found in available labels: "
                f"{all_labels}"
            )

        # Convert lists → numpy arrays within the selected label’s classes.
        # Keep scalar center_wavevector as a scalar.
        per_class_dicts = []
        for cls, bucket in store[select_label].items():
            dct = {}
            for k, v in bucket.items():
                if k in ("label", "_class"):
                    dct[k] = v
                elif k == "center_wavevector":
                    # keep scalar as-is, do not wrap in np.asarray
                    dct[k] = v
                else:
                    dct[k] = np.asarray(v)
            per_class_dicts.append(dct)

        selected_properties = (
            per_class_dicts[0] if len(per_class_dicts) == 1 else per_class_dicts
        )

        # Flat export dict: simple keys, includes optional extras
        exported_parameters = {
            "fermi_wavevector": fermi_wavevector,
            "fermi_velocity": fermi_velocity,
            "bare_mass": bare_mass,
            "side": side,
        }

        # Collect parameters without prefixing by class. This will also include
        # center_wavevector from the fitted SpectralQuadratic class, and since
        # there is no function argument with that name, it cannot be overridden.
        if isinstance(selected_properties, dict):
            for key, val in selected_properties.items():
                if key not in ("label", "_class"):
                    exported_parameters[key] = val
        else:
            # If multiple classes, merge sequentially
            # (last overwrites same-name keys).
            for cls_bucket in selected_properties:
                for key, val in cls_bucket.items():
                    if key not in ("label", "_class"):
                        exported_parameters[key] = val

        return (self._ekin_range, self.hnuminPhi, select_label,
                selected_properties, exported_parameters)