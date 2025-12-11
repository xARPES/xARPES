#!/usr/bin/env python3
"""
Generate .ipynb notebooks from .Rmd files using Jupytext.

- For every .Rmd (excluding hidden folders and .ipynb_checkpoints),
  create or overwrite a sibling <notebook>.ipynb using Jupytext.

Dependencies:
    pip install jupytext

Usage:
    Place Rmd2ipynb.py in the /examples directory, where it is .gitignored.
    $ python Rmd2ipynb.py  # run from anywhere; it operates where it exists.
"""

import os
import sys
from typing import Optional


def find_base_dir() -> str:
    """Return the directory where this script lives (or CWD in a REPL)."""
    return (os.path.dirname(os.path.abspath(__file__))
            if "__file__" in globals() else os.getcwd())


def is_hidden(name: str) -> bool:
    """Return True if a file or directory name is considered hidden."""
    return name.startswith(".")


def get_jupytext() -> Optional[object]:
    """
    Try to import jupytext and return the module, or None if unavailable.

    Prints a single warning if jupytext is not installed.
    """
    try:
        import jupytext  # type: ignore
        return jupytext
    except ImportError:
        print(
            "[WARN] 'jupytext' is not installed. "
            "Install it with 'pip install jupytext' to enable "
            ".Rmd -> .ipynb conversion.",
            file=sys.stderr,
        )
        return None


def convert_rmd_to_ipynb(rmd_path: str, jupytext) -> None:
    """
    Convert a single .Rmd file to a .ipynb notebook using Jupytext.

    The .ipynb file is written next to the .Rmd file. Existing notebooks
    are overwritten.
    """
    base, ext = os.path.splitext(rmd_path)
    if ext.lower() != ".rmd":
        return

    ipynb_path = base + ".ipynb"

    try:
        # Let Jupytext auto-detect the format from the extension
        nb = jupytext.read(rmd_path)
        jupytext.write(nb, ipynb_path)
        print(f"Converted: {rmd_path} -> {ipynb_path}")
    except Exception as exc:
        print(
            f"[ERROR] Failed to convert '{rmd_path}' to .ipynb: {exc}",
            file=sys.stderr,
        )


def main() -> None:
    base_dir = find_base_dir()
    jupytext = get_jupytext()
    if jupytext is None:
        # Nothing to do if we don't have jupytext
        return

    converted_any = False

    for path, folders, files in os.walk(base_dir, topdown=True):
        # Skip hidden folders and notebook checkpoint caches
        folders[:] = [
            name for name in folders
            if not is_hidden(name) and name != ".ipynb_checkpoints"
        ]

        for name in files:
            if is_hidden(name):
                continue
            if not name.endswith(".Rmd"):
                continue
            if ".ipynb_checkpoints" in path:
                continue

            rmd = os.path.join(path, name)
            convert_rmd_to_ipynb(rmd, jupytext)
            converted_any = True

    if not converted_any:
        print("No .Rmd files found to convert.")


if __name__ == "__main__":
    main()
