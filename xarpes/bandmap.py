# Copyright (C) 2025 xARPES Developers
# This program is free software under the terms of the GNU GPLv3 license.

# get_ax_fig_plt and add_fig_kwargs originate from pymatgen/util/plotting.py.
# Copyright (C) 2011-2024 Shyue Ping Ong and the pymatgen Development Team
# Pymatgen is released under the MIT License.

# See also abipy/tools/plotting.py.
# Copyright (C) 2021 Matteo Giantomassi and the AbiPy Group
# AbiPy is free software under the terms of the GNU GPLv2 license.

"""File containing the band map class."""

import numpy as np
from igor2 import binarywave
from .plotting import get_ax_fig_plt, add_fig_kwargs
from .functions import fit_least_squares, extend_function
from .distributions import FermiDirac, Linear
from .constants import PREF

class MdcSetResult(dict):
    """Dictionary-like MDC payload with backward-compatible unpacking.

    Iteration yields values in a positional order compatible with
    ``xarpes.MDCs(*bmap.mdc_set(...))``.
    """

    def __iter__(self):
        abscissa_type = self.get('abscissa_type')
        if abscissa_type == 'momenta':
            values = (
                self['mdcs'],
                None,
                None,
                self['energy_resolution'],
                self['temperature'],
                self['enel_range'],
                self['hnuminPhi'],
                self['abscissa_range'],
                self['abscissa_resolution'],
                'momenta',
            )
        else:
            values = (
                self['mdcs'],
                self['abscissa_range'],
                self['abscissa_resolution'],
                self['energy_resolution'],
                self['temperature'],
                self['enel_range'],
                self['hnuminPhi'],
                None,
                None,
                'angle',
            )
        return iter(values)


class BandMap:
    """
    Band map container for ARPES intensity data.

    A `BandMap` stores a two-dimensional ARPES intensity map together with
    one abscissa axis (angles or momenta) and a single energy axis (either
    kinetic or binding energy). Conversion between kinetic and binding energy
    is handled internally via the work-function offset ``hnuminPhi`` when
    available.

    Notes
    -----
    Users are encouraged to construct instances via
    :meth:`from_ibw_file` or :meth:`from_np_arrays`. The ``__init__``
    method expects fully initialized, canonical NumPy arrays and performs
    no file I/O.

    The intensity array is assumed to have shape
    ``(n_energy, n_angle)``, consistent with all downstream operations
    (plotting, MDC extraction, Fermi-edge fitting).

    See Also
    --------
    BandMap.from_ibw_file
    BandMap.from_np_arrays
    """

    @classmethod
    def from_ibw_file(cls, datafile, transpose=False, flip_ekin=False,
                       flip_angles=False, **kwargs):
        """
        Construct a `BandMap` from an IGOR Binary Wave (``.ibw``) file.

        The IGOR wave header is used to reconstruct the angular and
        kinetic-energy axes. The resulting instance uses kinetic energy
        (`ekin`) as the explicit energy axis.

        Parameters
        ----------
        datafile : path-like
            Path to the ``.ibw`` file.
        transpose : bool, optional
            If True, transpose the loaded intensity array and swap the
            associated axis metadata accordingly.
        flip_ekin : bool, optional
            If True, reverse the kinetic-energy axis (first dimension).
        flip_angles : bool, optional
            If True, reverse the angle axis (second dimension).
        **kwargs
            Additional keyword arguments forwarded to
            :class:`BandMap.__init__`.

        Returns
        -------
        BandMap
            New instance constructed from the file contents.

        Raises
        ------
        ValueError
            If the dimensions reported in the file header do not match
            the shape of the stored intensity array.
        """
        data = binarywave.load(datafile)
        intensities = data['wave']['wData']

        fnum, anum = data['wave']['wave_header']['nDim'][0:2]
        fstp, astp = data['wave']['wave_header']['sfA'][0:2]
        fmin, amin = data['wave']['wave_header']['sfB'][0:2]

        if intensities.shape != (fnum, anum):
            raise ValueError('nDim and shape of wData do not match.')

        if transpose:
            intensities = intensities.T
            fnum, anum = anum, fnum
            fstp, astp = astp, fstp
            fmin, amin = amin, fmin

        if flip_ekin:
            intensities = intensities[::-1, :]

        if flip_angles:
            intensities = intensities[:, ::-1]

        angles = np.linspace(amin, amin + (anum - 1) * astp, anum)
        ekin = np.linspace(fmin, fmin + (fnum - 1) * fstp, fnum)

        return cls(intensities=intensities, angles=angles, ekin=ekin,
                   **kwargs)


    @classmethod
    def from_np_arrays(cls, intensities=None, angles=None, momenta=None,
                       ekin=None, enel=None, **kwargs):
        """
        Construct a `BandMap` directly from NumPy arrays.

        Exactly one of `ekin` or `enel` must be provided and will become the
        authoritative energy axis. The other energy axis may be derived
        later if the work-function offset ``hnuminPhi`` is known.

        Parameters
        ----------
        intensities : array-like
            ARPES intensity map with shape ``(n_energy, n_angle)`` [counts].
        angles : array-like, optional
            Angular axis values with shape ``(n_angle,)`` [degree].
        momenta : array-like, optional
            Momentum axis values with shape ``(n_angle,)`` [Å⁻¹].
        ekin : array-like, optional
            Kinetic-energy axis values with shape ``(n_energy,)`` [eV].
        enel : array-like, optional
            Binding-energy axis values with shape ``(n_energy,)`` [eV].
        **kwargs
            Additional keyword arguments forwarded to
            :class:`BandMap.__init__`.

        Returns
        -------
        BandMap
            New instance constructed from the provided arrays.

        Raises
        ------
        ValueError
            If `intensities` is missing, if both (or neither) of `angles`
            and `momenta` are provided, or if both (or neither) of `ekin`
            and `enel` are provided.
        """
        if intensities is None:
            raise ValueError('Please provide intensities.')
        if (angles is None) == (momenta is None):
            raise ValueError('Provide exactly one of angles or momenta.')
        if (ekin is None) == (enel is None):
            raise ValueError('Provide exactly one of ekin or enel.')
        kwargs = dict(kwargs)
        if enel is not None and kwargs.get('hnuminPhi', None) is None:
            kwargs['hnuminPhi'] = 0.0

        return cls(intensities=intensities, angles=angles, momenta=momenta,
                   ekin=ekin, enel=enel, **kwargs)


    def __init__(self, intensities=None, angles=None, momenta=None,
                 ekin=None, enel=None, energy_resolution=None,
                 angle_resolution=None, momentum_resolution=None,
                 temperature=None, hnuminPhi=None, hnuminPhi_std=None):
        """
        Initialize a `BandMap` from canonical arrays and metadata.

        A `BandMap` represents a two-dimensional ARPES intensity map defined on
        one abscissa axis (`angles` or `momenta`) and a single authoritative
        energy axis. Exactly one of `ekin` (kinetic energy) or `enel` (binding
        energy) must be provided.
        The non-authoritative energy axis may be derived automatically when the
        work-function offset ``hnuminPhi = h\\nu - \\Phi`` has been set with
        the Fermi-edge fitting.

        Parameters
        ----------
        intensities : array-like
            ARPES intensity map with shape ``(n_energy, n_angle)`` [counts].
        angles : array-like, optional
            Angular axis values with shape ``(n_angle,)`` [degree].
        momenta : array-like, optional
            Momentum axis values with shape ``(n_angle,)`` [Å⁻¹].
        ekin : array-like, optional
            Kinetic-energy axis values with shape ``(n_energy,)`` [eV]. If
            provided, `ekin` becomes the authoritative energy axis.
        enel : array-like, optional
            Binding-energy axis values with shape ``(n_energy,)`` [eV]. If
            provided, `enel` becomes the authoritative energy axis.
        energy_resolution : float, optional
            Energy resolution of the measurement, [eV].
        angle_resolution : float, optional
            Angular resolution of the measurement [degree].
        momentum_resolution : float, optional
            Momentum resolution of the measurement [Å⁻¹].
        temperature : float, optional
            Sample temperature [K].
        hnuminPhi : float, optional
            Photon energy minus the work function, ``h\\nu - \\Phi`` [eV]. When
            provided, this value is used to convert between kinetic and binding
            energy via ``enel = ekin - hnuminPhi``.
        hnuminPhi_std : float, optional
            One-sigma standard deviation of `hnuminPhi` [eV].

        Notes
        -----
        Exactly one of `ekin` or `enel` must be provided at initialization.
        Attempting to set both (or neither) raises a `ValueError`.

        The energy axis provided at initialization becomes *authoritative*.
        After initialization:
        - If `ekin` is authoritative, setting `enel` raises an
          `AttributeError`.
        - If `enel` is authoritative, setting `ekin` raises an
          `AttributeError`.
        - Updating `hnuminPhi` updates the non-authoritative energy axis when
          possible.

        No copying or validation of array shapes is performed beyond basic
        presence checks; consistency of dimensions is assumed.

        Raises
        ------
        ValueError
            If required arrays are missing, if both (or neither) of `angles`
            and `momenta` are provided, or if both (or neither) of `ekin`
            and `enel` are provided.
        """

        # --- Required arrays ------------------------------------------------
        if intensities is None:
            raise ValueError('Please provide intensities.')
        if (angles is None) == (momenta is None):
            raise ValueError('Provide exactly one of angles or momenta.')

        self.intensities = intensities
        self.angles = angles
        self.momenta = momenta

        # --- Initialize energy axes (raw slots) ----------------------------
        self._ekin = None
        self._enel = None

        # Apply user overrides or file ekin
        if ekin is not None and enel is not None:
            raise ValueError('Provide only one of ekin or enel, not both.')

        if ekin is not None:
            self._ekin = ekin
        elif enel is not None:
            self._enel = enel
        else:
            raise ValueError('Please provide datafile, ekin, or enel.')

        # Scalars / metadata
        self.energy_resolution = energy_resolution
        self.angle_resolution = angle_resolution
        self.momentum_resolution = momentum_resolution
        self.temperature = temperature

        # --- 1) Track which axis is authoritative --------------------------
        self._ekin_explicit = ekin is not None
        self._enel_explicit = enel is not None

        if self._enel_explicit and hnuminPhi is None:
            hnuminPhi = 0.0

        # Work-function combo and its std
        self._hnuminPhi = None
        self._hnuminPhi_std = None
        self.hnuminPhi = hnuminPhi
        self.hnuminPhi_std = hnuminPhi_std

        # --- 2) Derive missing axis if possible ----------------------------
        if self._ekin is None and self._enel is not None \
                and self._hnuminPhi is not None:
            self._ekin = self._enel + self._hnuminPhi
        if self._enel is None and self._ekin is not None \
                and self._hnuminPhi is not None:
            self._enel = self._ekin - self._hnuminPhi

    # -------------------- Properties: data arrays ---------------------------
    @property
    def intensities(self):
        return self._intensities

    @intensities.setter
    def intensities(self, x):
        self._intensities = x

    @property
    def angles(self):
        return self._angles

    @angles.setter
    def angles(self, x):
        self._angles = x

    @property
    def momenta(self):
        return self._momenta

    @momenta.setter
    def momenta(self, x):
        self._momenta = x

    # -------------------- 3) Resolution / temperature ----------------------
    @property
    def angle_resolution(self):
        return self._angle_resolution

    @angle_resolution.setter
    def angle_resolution(self, x):
        self._angle_resolution = x

    @property
    def energy_resolution(self):
        return self._energy_resolution

    @energy_resolution.setter
    def energy_resolution(self, x):
        self._energy_resolution = x

    @property
    def momentum_resolution(self):
        return self._momentum_resolution

    @momentum_resolution.setter
    def momentum_resolution(self, x):
        self._momentum_resolution = x

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, x):
        self._temperature = x

    # -------------------- 4) Sync ekin / enel / hnuminPhi ------------------
    @property
    def ekin(self):
        if self._ekin is None and self._enel is not None \
                and self._hnuminPhi is not None:
            return self._enel + self._hnuminPhi
        return self._ekin

    @ekin.setter
    def ekin(self, x):
        if getattr(self, "_enel_explicit", False):
            raise AttributeError('enel is explicit; set hnuminPhi instead.')
        self._ekin = x
        self._ekin_explicit = True
        if not getattr(self, "_enel_explicit", False) \
                and self._hnuminPhi is not None and x is not None:
            self._enel = x - self._hnuminPhi

    @property
    def enel(self):
        if self._enel is None and self._ekin is not None \
                and self._hnuminPhi is not None:
            return self._ekin - self._hnuminPhi
        return self._enel

    @enel.setter
    def enel(self, x):
        if getattr(self, "_ekin_explicit", False):
            raise AttributeError('ekin is explicit; set hnuminPhi instead.')
        self._enel = x
        self._enel_explicit = True
        if not getattr(self, "_ekin_explicit", False) \
                and self._hnuminPhi is not None and x is not None:
            self._ekin = x + self._hnuminPhi

    @property
    def hnuminPhi(self):
        r"""Returns the photon energy minus the work function in eV if it has
        been set, either during instantiation, with the setter, or by fitting
        the Fermi-Dirac distribution to the integrated weight.

        Returns
        -------
        hnuminPhi : float, None
            Kinetic energy minus the work function [eV]

        """
        return self._hnuminPhi

    @hnuminPhi.setter
    def hnuminPhi(self, x):
        r"""TBD
        """
        self._hnuminPhi = x
        # Re-derive the non-explicit axis if possible
        if not getattr(self, "_ekin_explicit", False) \
                and self._enel is not None and x is not None:
            self._ekin = self._enel + x
        if not getattr(self, "_enel_explicit", False) \
                and self._ekin is not None and x is not None:
            self._enel = self._ekin - x

    @property
    def hnuminPhi_std(self):
        r"""Returns standard deviation of the photon energy minus the work
        function in eV.

        Returns
        -------
        hnuminPhi_std : float
            Standard deviation of energy minus the work function [eV]

        """
        return self._hnuminPhi_std

    @hnuminPhi_std.setter
    def hnuminPhi_std(self, x):
        r"""Manually sets the standard deviation of photon energy minus the
        work function in eV.

        Parameters
        ----------
        hnuminPhi_std : float
            Standard deviation of energy minus the work function [eV]

        """
        self._hnuminPhi_std = x

    def shift_angles(self, shift):
        r"""
        Shifts the angles by the specified amount in degrees. Used to shift
        from the detector angle to the material angle.

        Parameters
        ----------
        shift : float
            Angular shift [degrees]

        """
        if self.angles is None:
            raise RuntimeError(
                'Angles are not available for this BandMap. '
                'Use a BandMap initialized with angles.'
            )
        self.angles = self.angles + shift

    def mdc_set(self, abscissa_min=None, abscissa_max=None, energy_value=None,
                energy_range=None, angle_min=None, angle_max=None):
        r"""Return a dictionary containing a set of MDC data.

        This method extracts MDCs from the stored ARPES intensity map from a
        specified abscissa interval and either selecting a single energy slice
        or an energy window.

        Parameters
        ----------
        abscissa_min : float, optional
            Minimum value of the selected abscissa interval.
            For angle-initialized maps this is interpreted in [degrees];
            for momentum-initialized maps in [Å⁻¹].
        abscissa_max : float, optional
            Maximum value of the selected abscissa interval.
            For angle-initialized maps this is interpreted in [degrees];
            for momentum-initialized maps in [Å⁻¹].
        angle_min : float, optional
            Deprecated alias for ``abscissa_min``.
        angle_max : float, optional
            Deprecated alias for ``abscissa_max``.
        energy_value : float, optional
            Energy value [same units as ``self.enel``] at which a single MDC
            is extracted. Exactly one of ``energy_value`` or ``energy_range``
            must be provided.
        energy_range : array-like, optional
            Energy interval [same units as ``self.enel``] over which MDCs are
            extracted. Exactly one of ``energy_value`` or ``energy_range``
            must be provided.

        Returns
        -------
        dict
            Dictionary with keys:

            - ``mdcs``: extracted MDC intensities.
            - ``abscissa_range``: selected angle or momentum values.
            - ``abscissa_resolution``: matching scalar resolution.
            - ``abscissa_type``: ``"angle"`` or ``"momenta"``.
            - ``energy_resolution``: energy resolution [eV].
            - ``temperature``: sample temperature [K].
            - ``enel_range``: selected energy value(s).
            - ``hnuminPhi``: photon-energy-related offset.

            The returned object is dict-like and supports backward-compatible
            positional unpacking into ``MDCs(*result)``.

        Raises
        ------
        ValueError
            If neither or both of ``energy_value`` and ``energy_range`` are
            provided.
        RuntimeError
            If no abscissa axis is available.
        """

        if (energy_value is None and energy_range is None) or \
        (energy_value is not None and energy_range is not None):
            raise ValueError('Please provide either energy_value or ' +
            'energy_range.')

        if abscissa_min is None and angle_min is not None:
            abscissa_min = angle_min
        if abscissa_max is None and angle_max is not None:
            abscissa_max = angle_max

        if abscissa_min is None or abscissa_max is None:
            raise ValueError(
                'Please provide abscissa_min and abscissa_max.'
            )

        if self.momenta is not None:
            abscissa = self.momenta
            abscissa_type = 'momenta'
            abscissa_resolution = self.momentum_resolution
        elif self.angles is not None:
            abscissa = self.angles
            abscissa_type = 'angle'
            abscissa_resolution = self.angle_resolution
        else:
            raise RuntimeError('BandMap has no abscissa axis for MDC extraction.')

        abscissa_min_index = np.abs(abscissa - abscissa_min).argmin()
        abscissa_max_index = np.abs(abscissa - abscissa_max).argmin()
        abscissa_range_out = abscissa[abscissa_min_index:abscissa_max_index + 1]

        if energy_value is not None:
            energy_index = np.abs(self.enel - energy_value).argmin()
            enel_range_out = self.enel[energy_index]
            mdcs = self.intensities[energy_index,
                   abscissa_min_index:abscissa_max_index + 1]

        if energy_range is not None:
            energy_indices = np.where((self.enel >= np.min(energy_range))
                                      & (self.enel <= np.max(energy_range))) \
                                        [0]
            enel_range_out = self.enel[energy_indices]
            mdcs = self.intensities[energy_indices,
                                    abscissa_min_index:abscissa_max_index + 1]

        return MdcSetResult({
            'mdcs': mdcs,
            'abscissa_range': abscissa_range_out,
            'abscissa_resolution': abscissa_resolution,
            'abscissa_type': abscissa_type,
            'energy_resolution': self.energy_resolution,
            'temperature': self.temperature,
            'enel_range': enel_range_out,
            'hnuminPhi': self.hnuminPhi,
        })

    @add_fig_kwargs
    def plot(self, abscissa='momentum', ordinate='electron_energy',
             self_energies=None, ax=None, markersize=None,
             plot_dispersions='none', **kwargs):
        r"""
        Plot the band map. Optionally overlay a collection of self-energies,
        e.g. a CreateSelfEnergies instance or any iterable of self-energy
        objects. Self-energies are *not* stored internally; they are used
        only for this plotting call.

        When self-energies are present and ``abscissa='momentum'``, their
        MDC maxima are overlaid with 95 % confidence intervals.

        The `plot_dispersions` argument controls bare-band plotting:

        - "full"   : use the full momentum range of the map (default)
        - "none"   : do not plot bare dispersions
        - "kink"   : for each self-energy, use the min/max of its own
          momentum range (typically its MDC maxima), with
          `len(abscissa axis)` points.
        - "domain" : for SpectralQuadratic, use only the left or right
          domain relative to `center_wavevector`, based on the self-energy
          attribute `side` ("left" / "right"); for other cases this behaves
          as "full".
        """
        import warnings
        from . import settings_parameters as xprs

        plot_disp_mode = plot_dispersions
        valid_disp_modes = ('full', 'none', 'kink', 'domain')
        if plot_disp_mode not in valid_disp_modes:
            raise ValueError(
                f"Invalid plot_dispersions '{plot_disp_mode}'. "
                f"Valid options: {valid_disp_modes}."
            )

        valid_abscissa = ('angle', 'momentum')
        valid_ordinate = ('kinetic_energy', 'electron_energy')

        if abscissa not in valid_abscissa:
            raise ValueError(
                f"Invalid abscissa '{abscissa}'. "
                f"Valid options: {valid_abscissa}"
            )
        if ordinate not in valid_ordinate:
            raise ValueError(
                f"Invalid ordinate '{ordinate}'. "
                f"Valid options: {valid_ordinate}"
            )

        if self_energies is not None:

            # MDC maxima are defined in momentum space, not angle space
            if abscissa == 'angle':
                raise ValueError(
                    "MDC maxima cannot be plotted against angles; they are "
                    "defined in momentum space. Use abscissa='momentum' "
                    "when passing self-energies."
                )

        ax, fig, plt = get_ax_fig_plt(ax=ax)

        n_abscissa = self.intensities.shape[1]

        if abscissa == 'angle':
            if self.angles is None:
                raise RuntimeError(
                    "BandMap has no angular axis. Use abscissa='momentum'."
                )

            Angl, Ekin = np.meshgrid(self.angles, self.ekin)
            ax.set_xlabel('Angle ($\\degree$)')
            if ordinate == 'kinetic_energy':
                mesh = ax.pcolormesh(
                    Angl, Ekin, self.intensities,
                    shading='auto',
                    cmap=plt.get_cmap('bone').reversed())
                ax.set_ylabel('$E_{\\mathrm{kin}}$ (eV)')
            elif ordinate == 'electron_energy':
                Enel = Ekin - self.hnuminPhi
                mesh = ax.pcolormesh(
                    Angl, Enel, self.intensities,
                    shading='auto',
                    cmap=plt.get_cmap('bone').reversed())
                ax.set_ylabel('$E-\\mu$ (eV)')

        elif abscissa == 'momentum':
            ax.set_xlabel(r'$k_{//}$ ($\mathrm{\AA}^{-1}$)')

            momenta = self.momenta
            if momenta is not None:
                momenta = np.asarray(momenta, dtype=float)
                if momenta.ndim != 1:
                    raise ValueError('momenta must be one-dimensional.')
                if len(momenta) != n_abscissa:
                    raise ValueError(
                        'Length of momenta axis must match the second '
                        'dimension of intensities.'
                    )

                Mome, Ekin = np.meshgrid(momenta, self.ekin)
                mome_min = np.min(momenta)
                mome_max = np.max(momenta)
                full_disp_momenta = momenta
                wlist = []
            else:
                if self.angles is None:
                    raise RuntimeError(
                        'BandMap has neither momenta nor angles to build '
                        'momentum coordinates.'
                    )
                angles = np.asarray(self.angles, dtype=float)
                if angles.ndim != 1:
                    raise ValueError('angles must be one-dimensional.')
                if len(angles) != n_abscissa:
                    raise ValueError(
                        'Length of angles axis must match the second '
                        'dimension of intensities.'
                    )

                Angl, Ekin = np.meshgrid(angles, self.ekin)
                with warnings.catch_warnings(record=True) as wlist:
                    warnings.filterwarnings(
                        "always",
                        message=(
                            "The input coordinates to pcolormesh are "
                            "interpreted as cell centers, but are not "
                            "monotonically increasing or decreasing."
                        ),
                        category=UserWarning,
                    )
                    Mome = np.sqrt(Ekin / PREF) * np.sin(np.deg2rad(Angl))
                mome_min = np.min(Mome)
                mome_max = np.max(Mome)
                full_disp_momenta = np.linspace(
                    mome_min, mome_max, n_abscissa
                )

            if ordinate == 'kinetic_energy':
                mesh = ax.pcolormesh(
                    Mome, Ekin, self.intensities,
                    shading='auto',
                    cmap=plt.get_cmap('bone').reversed())
                ax.set_ylabel('$E_{\\mathrm{kin}}$ (eV)')
            elif ordinate == 'electron_energy':
                Enel = Ekin - self.hnuminPhi
                mesh = ax.pcolormesh(
                    Mome, Enel, self.intensities,
                    shading='auto',
                    cmap=plt.get_cmap('bone').reversed())
                ax.set_ylabel('$E-\\mu$ (eV)')

            y_lims = ax.get_ylim()

            if any("cell centers" in str(w.message) for w in wlist):
                warnings.warn(
                    "Conversion from angle to momenta causes warping of the "
                    "cell centers. \n Cell edges of the mesh plot may look "
                    "irregular.",
                    UserWarning,
                    stacklevel=2,
                )

        if abscissa == 'momentum' and self_energies is not None:
            for self_energy in self_energies:

                mdc_maxima = getattr(self_energy, "mdc_maxima", None)

                # If this self-energy doesn't contain maxima, don't plot
                if mdc_maxima is None:
                    continue

                # Reserve a colour from the axes cycle for this self-energy,
                # and use it consistently for MDC maxima and dispersion.
                line_color = ax._get_lines.get_next_color()

                peak_sigma = getattr(
                    self_energy, "peak_positions_sigma", None
                )
                xerr = xprs.sigma_confidence * peak_sigma if peak_sigma is \
                    not None else None

                if ordinate == 'kinetic_energy':
                    y_vals = self_energy.ekin_range
                else:
                    y_vals = self_energy.enel_range

                x_vals = mdc_maxima
                label = getattr(self_energy, "label", None)

                # First plot the MDC maxima, using the reserved colour
                if xerr is not None:
                    ax.errorbar(
                        x_vals, y_vals, xerr=xerr, fmt='o',
                        linestyle='', label=label,
                        markersize=markersize,
                        color=line_color, ecolor=line_color,
                    )
                else:
                    ax.plot(
                        x_vals, y_vals, linestyle='',
                        marker='o', label=label,
                        markersize=markersize,
                        color=line_color,
                    )

                # Bare-band dispersion for SpectralLinear / SpectralQuadratic
                spec_class = getattr(
                    self_energy, "_class",
                    self_energy.__class__.__name__,
                )

                if (plot_disp_mode != 'none'
                        and spec_class in ("SpectralLinear",
                                           "SpectralQuadratic")):

                    # Determine momentum grid for the dispersion
                    if plot_disp_mode == 'kink':
                        x_arr = np.asarray(x_vals)
                        mask = np.isfinite(x_arr)
                        if not np.any(mask):
                            # No valid k-points to define a range
                            continue
                        k_min = np.min(x_arr[mask])
                        k_max = np.max(x_arr[mask])
                        disp_momenta = np.linspace(
                            k_min, k_max, n_abscissa
                        )
                    elif (plot_disp_mode == 'domain'
                          and spec_class == "SpectralQuadratic"):
                        side = getattr(self_energy, "side", None)
                        if side == 'left':
                            disp_momenta = np.linspace(
                                mome_min, self_energy.center_wavevector,
                                n_abscissa
                            )
                        elif side == 'right':
                            disp_momenta = np.linspace(
                                self_energy.center_wavevector, mome_max,
                                n_abscissa
                            )
                        else:
                            # Fallback: no valid side, use full range
                            disp_momenta = full_disp_momenta
                    else:
                        # 'full' or 'domain' for SpectralLinear
                        disp_momenta = full_disp_momenta

                    # --- Robust parameter checks before computing base_disp ---
                    if spec_class == "SpectralLinear":
                        fermi_vel = getattr(
                            self_energy, "fermi_velocity", None
                        )
                        fermi_k = getattr(
                            self_energy, "fermi_wavevector", None
                        )
                        if fermi_vel is None or fermi_k is None:
                            missing = []
                            if fermi_vel is None:
                                missing.append("fermi_velocity")
                            if fermi_k is None:
                                missing.append("fermi_wavevector")
                            raise TypeError(
                                "Cannot plot bare dispersion for "
                                "SpectralLinear: "
                                f"{', '.join(missing)} is None."
                            )

                        base_disp = (
                            fermi_vel * (disp_momenta - fermi_k)
                        )

                    else:  # SpectralQuadratic
                        bare_mass = getattr(
                            self_energy, "bare_mass", None
                        )
                        center_k = getattr(
                            self_energy, "center_wavevector", None
                        )
                        fermi_k = getattr(
                            self_energy, "fermi_wavevector", None
                        )

                        missing = []
                        if bare_mass is None:
                            missing.append("bare_mass")
                        if center_k is None:
                            missing.append("center_wavevector")
                        if fermi_k is None:
                            missing.append("fermi_wavevector")

                        if missing:
                            raise TypeError(
                                "Cannot plot bare dispersion for "
                                "SpectralQuadratic: "
                                f"{', '.join(missing)} is None."
                            )

                        dk = disp_momenta - center_k
                        base_disp = PREF * (dk ** 2 - fermi_k ** 2) / bare_mass
                    # --- end parameter checks and base_disp construction ---

                    if ordinate == 'electron_energy':
                        disp_vals = base_disp
                    else:  # kinetic energy
                        disp_vals = base_disp + self.hnuminPhi

                    band_label = getattr(self_energy, "label", None)
                    if band_label is not None:
                        band_label = f"{band_label} (bare)"

                    ax.plot(
                        disp_momenta, disp_vals,
                        label=band_label,
                        linestyle='--',
                        color=line_color,
                    )

            handles, labels = ax.get_legend_handles_labels()
            if any(labels):
                ax.legend()

            ax.set_ylim(y_lims)

        plt.colorbar(mesh, ax=ax, label='counts (-)')
        return fig

    def _resolve_abscissa_interval(self, angle_min, angle_max,
                                  momentum_min, momentum_max):
        """Resolve which abscissa axis and interval bounds to use."""
        using_momentum_bounds = (momentum_min is not None) \
            or (momentum_max is not None)
        using_angle_bounds = not (
            np.isneginf(angle_min) and np.isposinf(angle_max)
        )

        if using_momentum_bounds and using_angle_bounds:
            raise ValueError(
                'Please provide either angle_min/angle_max or '
                'momentum_min/momentum_max, not both.'
            )

        if using_momentum_bounds:
            if momentum_min is None or momentum_max is None:
                raise ValueError(
                    'Provide both momentum_min and momentum_max.'
                )
            if self.momenta is None:
                raise RuntimeError(
                    'Momentum bounds were provided, but this BandMap has '
                    'no momentum axis.'
                )
            return self.momenta, momentum_min, momentum_max, 'momentum'

        if using_angle_bounds:
            if self.angles is None:
                raise RuntimeError(
                    'Angle bounds were provided, but this BandMap has '
                    'no angle axis.'
                )
            return self.angles, angle_min, angle_max, 'angle'

        if self.momenta is not None:
            return self.momenta, -np.inf, np.inf, 'momentum'
        if self.angles is not None:
            return self.angles, -np.inf, np.inf, 'angle'

        raise RuntimeError('BandMap has no abscissa axis.')

    @add_fig_kwargs
    def fit_fermi_edge(self, hnuminPhi_guess, background_guess=0.0,
                       integrated_weight_guess=1.0, angle_min=-np.inf,
                       angle_max=np.inf, momentum_min=None,
                       momentum_max=None, ekin_min=-np.inf,
                       ekin_max=np.inf, ax=None, **kwargs):
        r"""Fits the Fermi edge of the band map and plots the result.
        Also sets hnuminPhi, the kinetic energy minus the work function in eV.
        The fitting includes an energy convolution with an abscissa range
        expanded by 5 times the energy resolution standard deviation.

        Parameters
        ----------
        hnuminPhi_guess : float
            Initial guess for kinetic energy minus the work function [eV]
        background_guess : float
            Initial guess for background intensity [counts]
        integrated_weight_guess : float
            Initial guess for integrated spectral intensity [counts]
        angle_min : float
            Minimum angle of integration interval [degrees].
            Use together with ``angle_max``.
        angle_max : float
            Maximum angle of integration interval [degrees].
            Use together with ``angle_min``.
        momentum_min : float, optional
            Minimum momentum of integration interval [Å⁻¹].
            Use together with ``momentum_max``.
        momentum_max : float, optional
            Maximum momentum of integration interval [Å⁻¹].
            Use together with ``momentum_min``.
        ekin_min : float
            Minimum kinetic energy of integration interval [eV]
        ekin_max : float
            Maximum kinetic energy of integration interval [eV]
        ax : Matplotlib-Axes / NoneType
            Axis for plotting the Fermi edge on. Created if not provided by
            the user.

        Other parameters
        ----------------
        **kwargs : dict, optional
            Additional arguments passed on to add_fig_kwargs.

        Returns
        -------
        fig : Matplotlib-Figure
            Figure containing the Fermi edge fit

        """
        if getattr(self, "_enel_explicit", False):
            raise RuntimeError(
                "BandMap was initialized with binding energy (enel). "
                "Fermi-edge fitting is not required."
            )

        from scipy.ndimage import gaussian_filter

        ax, fig, plt = get_ax_fig_plt(ax=ax)

        abscissa, abscissa_min, abscissa_max, _ = \
            self._resolve_abscissa_interval(
                angle_min=angle_min,
                angle_max=angle_max,
                momentum_min=momentum_min,
                momentum_max=momentum_max,
            )

        min_abscissa_index = np.argmin(np.abs(abscissa - abscissa_min))
        max_abscissa_index = np.argmin(np.abs(abscissa - abscissa_max))

        min_ekin_index = np.argmin(np.abs(self.ekin - ekin_min))
        max_ekin_index = np.argmin(np.abs(self.ekin - ekin_max))

        energy_range = self.ekin[min_ekin_index:max_ekin_index]

        integrated_intensity = np.trapezoid(
            self.intensities[min_ekin_index:max_ekin_index,
                min_abscissa_index:max_abscissa_index], axis=1)

        fdir_initial = FermiDirac(temperature=self.temperature,
                                  hnuminPhi=hnuminPhi_guess,
                                  background=background_guess,
                                  integrated_weight=integrated_weight_guess,
                                  name='Initial guess')

        parameters = np.array(
            [hnuminPhi_guess, background_guess, integrated_weight_guess])

        extra_args = (self.temperature,)

        popt, pcov, _ = fit_least_squares(
            p0=parameters, xdata=energy_range, ydata=integrated_intensity,
            function=fdir_initial, resolution=self.energy_resolution,
            yerr=None, bounds=None, extra_args=extra_args)

        # Update hnuminPhi; automatically sets self.enel
        self.hnuminPhi = popt[0]
        self.hnuminPhi_std = np.sqrt(np.diag(pcov)[0])

        fdir_final = FermiDirac(temperature=self.temperature,
                                hnuminPhi=self.hnuminPhi, background=popt[1],
                                integrated_weight=popt[2],
                                name='Fitted result')

        ax.set_xlabel(r'$E_{\mathrm{kin}}$ (-)')
        ax.set_ylabel('Counts (-)')
        ax.set_xlim([ekin_min, ekin_max])

        ax.plot(energy_range, integrated_intensity, label='Data')

        extend, step, numb = extend_function(energy_range,
                                             self.energy_resolution)

        initial_result = gaussian_filter(fdir_initial.evaluate(extend),
                         sigma=step)[numb:-numb if numb else None]

        final_result = gaussian_filter(fdir_final.evaluate(extend),
                       sigma=step)[numb:-numb if numb else None]

        ax.plot(energy_range, initial_result, label=fdir_initial.name)
        ax.plot(energy_range, final_result, label=fdir_final.name)

        ax.legend()

        return fig


    @add_fig_kwargs
    def correct_fermi_edge(self, hnuminPhi_guess=None, background_guess=0.0,
                       integrated_weight_guess=1.0, angle_min=-np.inf,
                       angle_max=np.inf, momentum_min=None,
                       momentum_max=None, ekin_min=-np.inf, ekin_max=np.inf,
                       slope_guess=0, offset_guess=None,
                           true_angle=0, ax=None, **kwargs):
        r"""TBD
        hnuminPhi_guess should be estimated at true angle

        Parameters
        ----------
        hnuminPhi_guess : float, optional
            Initial guess for kinetic energy minus the work function [eV].
        angle_min : float
            Minimum angle of integration interval [degrees].
        angle_max : float
            Maximum angle of integration interval [degrees].
        momentum_min : float, optional
            Minimum momentum of integration interval [Å⁻¹].
        momentum_max : float, optional
            Maximum momentum of integration interval [Å⁻¹].

        Other parameters
        ----------------
        **kwargs : dict, optional
            Additional arguments passed on to add_fig_kwargs.

        Returns
        -------
        fig : Matplotlib-Figure
            Figure containing the Fermi edge fit

        """
        if getattr(self, "_enel_explicit", False):
            raise RuntimeError(
                "BandMap was initialized with binding energy (enel). "
                "Fermi-edge correction is not required."
            )

        from scipy.ndimage import map_coordinates
        from . import settings_parameters as xprs

        if hnuminPhi_guess is None:
            raise ValueError('Please provide an initial guess for ' +
                             'hnuminPhi.')

        # Here some loop where it fits all the Fermi edges
        abscissa, abscissa_min, abscissa_max, abscissa_kind = \
            self._resolve_abscissa_interval(
                angle_min=angle_min,
                angle_max=angle_max,
                momentum_min=momentum_min,
                momentum_max=momentum_max,
            )

        abscissa_min_index = np.abs(abscissa - abscissa_min).argmin()
        abscissa_max_index = np.abs(abscissa - abscissa_max).argmin()

        ekin_min_index = np.abs(self.ekin - ekin_min).argmin()
        ekin_max_index = np.abs(self.ekin - ekin_max).argmin()

        Intensities = self.intensities[ekin_min_index:ekin_max_index + 1,
                                       abscissa_min_index:abscissa_max_index + 1]
        abscissa_range = abscissa[abscissa_min_index:abscissa_max_index + 1]
        energy_range = self.ekin[ekin_min_index:ekin_max_index + 1]

        nmps = np.zeros_like(abscissa_range, dtype=float)
        stds = np.zeros_like(abscissa_range, dtype=float)

        hnuminPhi_left = hnuminPhi_guess - (true_angle - abscissa_min) \
        * slope_guess

        fdir_initial = FermiDirac(temperature=self.temperature,
                      hnuminPhi=hnuminPhi_left,
                      background=background_guess,
                      integrated_weight=integrated_weight_guess,
                      name='Initial guess')

        parameters = np.array(
                [hnuminPhi_left, background_guess, integrated_weight_guess])

        extra_args = (self.temperature,)

        for indx in range(abscissa_max_index - abscissa_min_index + 1):
            edge = Intensities[:, indx]

            parameters, pcov, _ = fit_least_squares(
                p0=parameters, xdata=energy_range, ydata=edge,
                function=fdir_initial, resolution=self.energy_resolution,
                yerr=None, bounds=None, extra_args=extra_args)

            nmps[indx] = parameters[0]
            stds[indx] = np.sqrt(np.diag(pcov)[0])

        # Offset at true angle if not set before
        if offset_guess is None:
            offset_guess = hnuminPhi_guess - slope_guess * true_angle

        parameters = np.array([offset_guess, slope_guess])

        lin_fun = Linear(offset_guess, slope_guess, 'Linear')

        popt, pcov, _ = fit_least_squares(p0=parameters, xdata=abscissa_range,
                        ydata=nmps, function=lin_fun, resolution=None,
                                 yerr=stds, bounds=None)

        linsp = lin_fun(abscissa_range, popt[0], popt[1])

        # Update hnuminPhi; automatically sets self.enel
        self.hnuminPhi = lin_fun(true_angle, popt[0], popt[1])
        self.hnuminPhi_std = np.sqrt(true_angle**2 * pcov[1, 1] + pcov[0, 0]
                                     + 2 * true_angle * pcov[0, 1])

        if abscissa_kind == 'momentum':
            Absc, Ekin = np.meshgrid(self.momenta, self.ekin)
            x_label = r'$k_{//}$ ($\mathrm{\AA}^{-1}$)'
            abscissa_axis = self.momenta
        else:
            Absc, Ekin = np.meshgrid(self.angles, self.ekin)
            x_label = 'Angle ($\degree$)'
            abscissa_axis = self.angles

        ax, fig, plt = get_ax_fig_plt(ax=ax)

        ax.set_xlabel(x_label)
        ax.set_ylabel('$E_{\mathrm{kin}}$ (eV)')
        mesh = ax.pcolormesh(Absc, Ekin, self.intensities,
                       shading='auto', cmap=plt.get_cmap('bone').reversed(),
                             zorder=1)

        ax.errorbar(abscissa_range, nmps, yerr=xprs.sigma_confidence * stds, zorder=1)
        ax.plot(abscissa_range, linsp, zorder=2)

        cbar = plt.colorbar(mesh, ax=ax, label='counts (-)')

        # Fermi-edge correction
        rows, cols = self.intensities.shape
        shift_values = popt[1] * abscissa_axis / (self.ekin[0] - self.ekin[1])
        row_coords = np.arange(rows).reshape(-1, 1) - shift_values
        col_coords = np.arange(cols).reshape(1, -1).repeat(rows, axis=0)
        self.intensities = map_coordinates(self.intensities,
                [row_coords, col_coords], order=1)

        return fig
