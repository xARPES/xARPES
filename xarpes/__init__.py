__version__ = "0.6.0"

from importlib import import_module

from .settings_parameters import parameter_settings
from .settings_plots import plot_settings


_LAZY_ATTR_MODULES = (
    "bandmap",
    "mdcs",
    "selfenergies",
    "distributions",
    "functions",
    "plotting",
    "constants",
)


def __getattr__(name):
    try:
        return globals()[name]
    except KeyError:
        pass

    for module in _LAZY_ATTR_MODULES:
        mod = import_module(f"{__name__}.{module}")
        if hasattr(mod, name):
            obj = getattr(mod, name)
            globals()[name] = obj
            return obj

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["__version__", "parameter_settings", "plot_settings"]