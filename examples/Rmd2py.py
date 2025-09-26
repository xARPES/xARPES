#!/usr/bin/env python3
"""
Update/generate .Rmd files from .ipynb using Jupytext, then convert all .Rmd to .py.

- Pass 1: For every .ipynb (excluding .ipynb_checkpoints and hidden folders),
          run `jupytext --to rmarkdown <notebook.ipynb>` to (re)write <notebook>.Rmd.
- Pass 2: For every .Rmd, produce a sibling .py using the user's original rules.

Dependencies:
    pip install jupytext

Usage:
    python Rmd2py.py   (run from anywhere; it operates relative to this file's directory)
"""

import os
import shutil
import subprocess
import sys

def find_base_dir() -> str:
    # Safe base directory (works both when run as script or in REPL)
    return os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()

def is_hidden(name: str) -> bool:
    return name.startswith(".")

def run_jupytext_ipynb_to_rmd(ipynb_path: str) -> bool:
    """
    Use the jupytext CLI to convert .ipynb -> .Rmd next to it.
    Returns True if conversion ran (and exited 0), False otherwise.
    """
    jupytext = shutil.which("jupytext")
    if not jupytext:
        print("[WARN] 'jupytext' not found on PATH. Skipping .ipynb -> .Rmd step for:",
              ipynb_path, file=sys.stderr)
        return False

    cmd = [jupytext, "--to", "rmarkdown", ipynb_path]
    try:
        res = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        # Optional: uncomment to see jupytext output
        # print(res.stdout, end="")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] jupytext failed for {ipynb_path}\n{e.stdout}", file=sys.stderr)
        return False

def convert_rmd_to_py(rmd_path: str) -> None:
    """
    Convert a single .Rmd file to a .py script according to the original rules.
    """
    py_path = rmd_path[:-3] + "py"  # replace .Rmd with .py
    with open(rmd_path, "r", encoding="utf-8") as lines, open(py_path, "w", encoding="utf-8") as text:
        text.write("#!/usr/bin/env python3\n")
        first_magic_comment = True
        in_yaml = False

        for line in lines:
            # Skip YAML header (between --- lines)
            if line.startswith("---"):
                in_yaml = not in_yaml
                continue
            if in_yaml:
                continue

            # Skip fenced code markers
            if line.startswith("```"):
                continue

            # Skip lines marked as "Jupyter only"
            if "Jupyter only" in line:
                continue

            # Remove IPython magics
            if "%matplotlib widget" in line or "%matplotlib inline" in line:
                continue

            # Replace first Jupyter magic comment with Qt5Agg backend
            if line.replace(" ", "").startswith("#%"):
                if first_magic_comment:
                    text.write("import matplotlib as mpl\n")
                    text.write("mpl.use('Qt5Agg')\n")
                    first_magic_comment = False
                continue

            # Otherwise write the line verbatim
            text.write(line)

def main() -> None:
    base_dir = find_base_dir()

    # ---- Pass 1: .ipynb -> .Rmd via jupytext ---------------------------------
    converted_any = False
    for path, folders, files in os.walk(base_dir, topdown=True):
        # Skip hidden folders and notebook checkpoint caches
        folders[:] = [name for name in folders if not is_hidden(name) and name != ".ipynb_checkpoints"]

        for name in files:
            if is_hidden(name):
                continue
            if not name.endswith(".ipynb"):
                continue
            if ".ipynb_checkpoints" in path:
                continue

            ipynb = os.path.join(path, name)
            ok = run_jupytext_ipynb_to_rmd(ipynb)
            converted_any = converted_any or ok

    if not converted_any:
        # Not an error—maybe there are no ipynb files, or jupytext isn't installed.
        pass

    # ---- Pass 2: .Rmd -> .py (your existing logic) ---------------------------
    for path, folders, files in os.walk(base_dir, topdown=True):
        folders[:] = [name for name in folders if not is_hidden(name)]
        for name in files:
            if is_hidden(name):
                continue
            if not name.endswith(".Rmd"):
                continue

            rmd = os.path.join(path, name)
            convert_rmd_to_py(rmd)

if __name__ == "__main__":
    main()
