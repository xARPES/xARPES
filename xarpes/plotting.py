# Copyright (C) 2024 xARPES Developers
# This program is free software under the terms of the GNU GPLv2 license.

# get_ax_fig_plt and add_fig_kwargs originate from pymatgen/util/plotting.py.
# Copyright (C) 2011-2024 Shyue Ping Ong and the pymatgen Development Team
# pymatgen is released under the MIT License.

# See also abipy/tools/plotting.py.
# Copyright (C) 2021 Matteo Giantomassi and the AbiPy Group
# AbiPy is free software under the terms of the GNU GPLv2 license.

"""Functions related to plotting."""

from functools import wraps
import matplotlib.pyplot as plt
import matplotlib as mpl

def my_plot_settings(name='default'):
    mpl.rc('xtick', labelsize=10, direction='in')
    mpl.rc('ytick', labelsize=10, direction='in')
    lw = dict(default=2.0, large=4.0)[name]
    mpl.rcParams['lines.linewidth'] = lw
    mpl.rcParams['lines.markersize'] = 3
    mpl.rcParams['xtick.major.size'] = 4
    mpl.rcParams['xtick.minor.size'] = 2
    mpl.rcParams['xtick.major.width'] = 0.8
    mpl.rcParams.update({'font.size': 16})


def get_ax_fig_plt(ax=None, **kwargs):
    r"""Helper function used in plot functions supporting an optional Axes
    argument. If ax is None, we build the `matplotlib` figure and create the
    Axes else. We return the current active figure.

    Args:
        ax (Axes, optional): Axes object. Defaults to None.
        kwargs: keyword arguments are passed to plt.figure if ax is not None.

      Returns:
        ax: :class:`Axes` object
        figure: matplotlib figure
        plt: matplotlib pyplot module.
    """
    if ax is None:
        fig = plt.figure(**kwargs)
        ax = fig.gca()
    else:
        fig = plt.gcf()

    return ax, fig, plt


def add_fig_kwargs(func):
    """Decorator that adds keyword arguments for functions returning matplotlib
    figures.

    The function should return either a matplotlib figure or None to signal
    some sort of error/unexpected event.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # pop the kwds used by the decorator.
        title = kwargs.pop('title', None)
        size_kwargs = kwargs.pop('size_kwargs', None)
        show = kwargs.pop('show', True)
        savefig = kwargs.pop('savefig', None)
        tight_layout = kwargs.pop('tight_layout', False)
        ax_grid = kwargs.pop('ax_grid', None)
        ax_annotate = kwargs.pop('ax_annotate', None)
        fig_close = kwargs.pop('fig_close', False)

        # Call func and return immediately if None is returned.
        fig = func(*args, **kwargs)
        if fig is None:
            return fig

        # Operate on matplotlib figure.
        if title is not None:
            fig.suptitle(title)

        if size_kwargs is not None:
            fig.set_size_inches(size_kwargs.pop('w'), size_kwargs.pop('h'),
                                **size_kwargs)

        if ax_grid is not None:
            for ax in fig.axes:
                ax.grid(bool(ax_grid))

        if ax_annotate:
            tags = ascii_letters
            if len(fig.axes) > len(tags):
                tags = (1 + len(ascii_letters) // len(fig.axes)) * ascii_letters
            for ax, tag in zip(fig.axes, tags):
                ax.annotate(f'({tag})', xy=(0.05, 0.95),
                            xycoords='axes fraction')

        if tight_layout:
            try:
                fig.tight_layout()
            except Exception as exc:
                # For some unknown reason, this problem shows up only on travis.
                # https://stackoverflow.com/questions/22708888/valueerror-when-using-matplotlib-tight-layout
                print('Ignoring Exception raised by fig.tight_layout\n',
                      str(exc))

        if savefig:
            fig.savefig(savefig)

        if show:
            plt.show()
        if fig_close:
            plt.close(fig=fig)

        return fig

    # Add docstring to the decorated method.
    doc_str = """\n\n
        Keyword arguments controlling the display of the figure:

        ================  ====================================================
        kwargs            Meaning
        ================  ====================================================
        title             Title of the plot (Default: None).
        show              True to show the figure (default: True).
        savefig           "abc.png" or "abc.eps" to save the figure to a file.
        size_kwargs       Dictionary with options passed to fig.set_size_inches
                          e.g. size_kwargs=dict(w=3, h=4)
        tight_layout      True to call fig.tight_layout (default: False)
        ax_grid           True (False) to add (remove) grid from all axes in fig.
                          Default: None i.e. fig is left unchanged.
        ax_annotate       Add labels to  subplots e.g. (a), (b).
                          Default: False
        fig_close         Close figure. Default: False.
        ================  ====================================================

"""

    if wrapper.__doc__ is not None:
        # Add s at the end of the docstring.
        wrapper.__doc__ += f'\n{doc_str}'
    else:
        # Use s
        wrapper.__doc__ = doc_str

    return wrapper
