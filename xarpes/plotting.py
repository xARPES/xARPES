# Copyright (C) 2024 xARPES Developers
# This program is free software under the terms of the GNU GPLv3 license.

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

def plot_settings(name='default'):
    mpl.rc('xtick', labelsize=10, direction='in')
    mpl.rc('ytick', labelsize=10, direction='in')
    lw = dict(default=2.0, large=4.0)[name]
    mpl.rcParams['lines.linewidth'] = lw
    mpl.rcParams['lines.markersize'] = 3
    mpl.rcParams['xtick.major.size'] = 4
    mpl.rcParams['xtick.minor.size'] = 2
    mpl.rcParams['xtick.major.width'] = 0.8
    mpl.rcParams.update({'font.size': 16})
    plt.rcParams['legend.frameon'] = False


def get_ax_fig_plt(ax=None, **kwargs):
    r"""Helper function used in plot functions supporting an optional `Axes`
    argument.

    If `ax` is `None`, we build the `matplotlib` figure and create the `Axes`.
    Else we return the current active figure.

    Parameters
    ----------
    ax : object
        `Axes` object. Defaults to `None`.
    **kwargs
        Keyword arguments are passed to `plt.figure` if `ax` is not `None`.

    Returns
    -------
    ax : object
        `Axes` object.
    figure : object
        `matplotlib` figure.
    plt : object
        `matplotlib.pyplot` module.
    """
    if ax is None:
        fig = plt.figure(**kwargs)
        ax = fig.gca()
    else:
        fig = plt.gcf()

    return ax, fig, plt

from functools import wraps
import matplotlib.pyplot as plt
import string

def add_fig_kwargs(func):
    """Decorator that adds keyword arguments for functions returning matplotlib
    figures.

    The function should return either a matplotlib figure or a tuple where the first element
    is a matplotlib figure, or None to signal some sort of error/unexpected event.
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
        fig_close = kwargs.pop('fig_close', True)

        # Call the original function
        result = func(*args, **kwargs)
        
        # Determine if result is a figure or a tuple with the first element as figure
        if isinstance(result, tuple):
            fig = result[0]
            rest = result[1:]
        else:
            fig = result
            rest = None

        # Return immediately if no figure is returned
        if fig is None:
            return result

        # Operate on the matplotlib figure
        if title is not None:
            fig.suptitle(title)

        if size_kwargs is not None:
            fig.set_size_inches(size_kwargs.pop('w'), size_kwargs.pop('h'), **size_kwargs)

        if ax_grid is not None:
            for ax in fig.axes:
                ax.grid(bool(ax_grid))

        if ax_annotate:
            tags = string.ascii_letters
            if len(fig.axes) > len(tags):
                tags = (1 + len(string.ascii_letters) // len(fig.axes)) * string.ascii_letters
            for ax, tag in zip(fig.axes, tags):
                ax.annotate(f'({tag})', xy=(0.05, 0.95), xycoords='axes fraction')

        if tight_layout:
            try:
                fig.tight_layout()
            except Exception as exc:
                print('Ignoring Exception raised by fig.tight_layout\n', str(exc))

        if savefig:
            fig.savefig(savefig)

        if show:
            plt.show()
        if fig_close:
            plt.close(fig=fig)

        # Reassemble the tuple if necessary and return
        if rest is not None:
            return (fig, *rest)
        else:
            return fig

    # Add docstring to the decorated method.
    doc_str = """\n\n

        notes
        -----

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
        ax_grid           True (False) to add (remove) grid from all axes in
                          fig.
                          Default: None i.e. fig is left unchanged.
        ax_annotate       Add labels to subplots e.g. (a), (b).
                          Default: False
        fig_close         Close figure. Default: True.
        ================  ====================================================

"""

    if wrapper.__doc__ is not None:
        wrapper.__doc__ += f'\n{doc_str}'
    else:
        wrapper.__doc__ = doc_str

    return wrapper
