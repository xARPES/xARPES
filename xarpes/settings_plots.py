# Copyright (C) 2025 xARPES Developers
# This program is free software under the terms of the GNU GPLv3 license.

"""Plotting and notebook behaviour settings for xARPES."""

import matplotlib.pyplot as plt

def plot_settings(name="default", register_pre_run=True):
    """Configure default plotting style for xARPES.

    Parameters
    ----------
    name : {"default", "large"}
        Select a predefined style.
    register_pre_run : bool
        If True, register a Jupyter pre-run hook that closes figures.
    """
    import matplotlib as mpl

    mpl.rc("xtick", labelsize=10, direction="in")
    mpl.rc("ytick", labelsize=10, direction="in")
    plt.rcParams["legend.frameon"] = False
    lw = dict(default=2.0, large=4.0)[name]

    mpl.rcParams.update({
        "lines.linewidth": lw,
        "lines.markersize": 3,
        "xtick.major.size": 4,
        "xtick.minor.size": 2,
        "xtick.major.width": 0.8,
        "font.size": 16,
        "axes.ymargin": 0.15,
    })

    if register_pre_run:
        _maybe_register_pre_run_close_all()


def _maybe_register_pre_run_close_all():
    """Register a pre_run_cell hook once, and only inside Jupyter."""
    from IPython import get_ipython

    if getattr(_maybe_register_pre_run_close_all, "_registered", False):
        return

    ip = get_ipython()
    if ip is None or ip.__class__.__name__ != "ZMQInteractiveShell":
        return

    def _close_all(_info):
        plt.close("all")

    ip.events.register("pre_run_cell", _close_all)
    _maybe_register_pre_run_close_all._registered = True