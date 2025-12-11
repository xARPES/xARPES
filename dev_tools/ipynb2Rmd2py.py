#!/usr/bin/env python3
"""
Update/generate .Rmd files from .ipynb using Jupytext, then convert all .Rmd to .py.

- Pass 1: For every .ipynb (excluding .ipynb_checkpoints and hidden folders),
          run `jupytext --to rmarkdown <notebook.ipynb>` to (re)write <notebook>.Rmd.
- Pass 2: For every .Rmd, produce a sibling .py using the user's original rules.

Dependencies:
    pip install jupytext

Usage:
    Place Rmd2py.py in the /examples directory, where it is .gitignored.
    $ python Rmd2py.py # run from anywhere; it operates where it exists.
"""

import os
import re
import shutil
import subprocess
import sys


def find_base_dir() -> str:
    # Safe base directory (works both when run as script or in REPL)
    return (os.path.dirname(os.path.abspath(__file__))
            if "__file__" in globals()
            else os.getcwd())


def is_hidden(name: str) -> bool:
    return name.startswith(".")


def run_jupytext_ipynb_to_rmd(ipynb_path: str) -> bool:
    """
    Use the jupytext CLI to convert .ipynb -> .Rmd next to it.
    Returns True if conversion ran (and exited 0), False otherwise.
    """
    jupytext = shutil.which("jupytext")
    if not jupytext:
        print("[WARN] 'jupytext' not found on PATH. "
              "Skipping .ipynb -> .Rmd step for:",
              ipynb_path, file=sys.stderr)
        return False

    cmd = [jupytext, "--to", "rmarkdown", ipynb_path]
    try:
        res = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        # Optional: uncomment to see jupytext output
        # print(res.stdout, end="")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] jupytext failed for {ipynb_path}\n{e.stdout}",
              file=sys.stderr)
        return False


def convert_rmd_to_py(rmd_path: str) -> None:
    """
    Convert a single .Rmd file to a .py script:
    - YAML front matter is skipped
    - Markdown outside code fences is commented with '# '
    - Code inside fences (``` or ~~~) is written verbatim (with existing magic
      handling)
    """
    py_path = rmd_path[:-3] + "py"  # replace .Rmd with .py
    with open(rmd_path, "r", encoding="utf-8") as lines, \
            open(py_path, "w", encoding="utf-8") as text:
        text.write("#!/usr/bin/env python3\n")
        first_magic_comment = True
        in_yaml = False
        in_code = False  # track fenced code blocks

        for raw in lines:
            line = raw

            # ---- YAML front matter --------------------------------------------
            if line.startswith("---") and not in_code:
                in_yaml = not in_yaml
                continue
            if in_yaml:
                continue

            # ---- Fence open/close (```... or ~~~...) --------------------------
            # Rmd/Quarto code chunks: ```{python, echo=FALSE} ... ```
            if (line.lstrip().startswith("```") or
                    line.lstrip().startswith("~~~")) and not in_yaml:
                in_code = not in_code
                # Do not emit the fence line itself
                continue

            if in_code:
                # ---- Inside code fence: keep code, apply your filters ---------
                # Skip lines marked as "Jupyter only"
                if "Jupyter only" in line:
                    continue

                # Remove IPython magics
                if ("%matplotlib widget" in line or
                        "%matplotlib inline" in line):
                    continue

                # Remove global Jupyter hooks (e.g. get_ipython().events...)
                if "get_ipython" in line:
                    continue

                # Replace first Jupyter magic comment with Qt5Agg backend
                if line.replace(" ", "").startswith("#%"):
                    if first_magic_comment:
                        text.write("import matplotlib as mpl\n")
                        text.write("mpl.use('Qt5Agg')\n")
                        first_magic_comment = False
                    continue

                # Otherwise, write code verbatim
                text.write(line)

            else:
                # ---- Outside code fence: Markdown -> Python comments ----------
                if line.strip() == "":
                    # Preserve blank lines (safe in Python)
                    text.write("\n")
                else:
                    # Comment any Markdown/text line so it doesn't break
                    # execution
                    text.write("# " + line)


def normalise_rmd_metadata(rmd_path: str) -> None:
    """
    Standardise the YAML metadata of the Rmd file (string-based, no yaml lib):

    - jupytext_version: 1.15.2
    - kernelspec.display_name: Python 3 (ipykernel)
    - kernelspec.name: python3

    Only modifies existing lines in the YAML header. If the keys do not exist,
    nothing is added.
    """
    try:
        with open(rmd_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return

    if not lines:
        return

    # Find first two '---' lines that delimit the YAML front matter.
    if not lines[0].startswith("---"):
        return

    yaml_start = 0
    yaml_end = None
    for i in range(1, len(lines)):
        if lines[i].startswith("---"):
            yaml_end = i
            break

    if yaml_end is None:
        # Malformed header, bail out.
        return

    header_lines = lines[yaml_start + 1:yaml_end]
    header = "".join(header_lines)

    # Replace jupytext_version
    header = re.sub(
        r'^(\s*jupytext_version:\s*).*$',
        r'\g<1>1.15.2',
        header,
        flags=re.MULTILINE,
    )

    # Replace kernelspec.display_name
    header = re.sub(
        r'^(\s*display_name:\s*).*$',
        r'\g<1>Python 3 (ipykernel)',
        header,
        flags=re.MULTILINE,
    )

    # Replace kernelspec.name
    header = re.sub(
        r'^(\s*name:\s*).*$',
        r'\g<1>python3',
        header,
        flags=re.MULTILINE,
    )

    new_header_lines = header.splitlines(keepends=True)
    new_lines = (
        lines[:yaml_start + 1] +
        new_header_lines +
        lines[yaml_end:]
    )

    try:
        with open(rmd_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
    except OSError:
        # If we can't write, silently skip; script should still continue.
        return


def main() -> None:
    base_dir = find_base_dir()

    # ---- Pass 1: .ipynb -> .Rmd via jupytext -------------------------------
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
            if not name.endswith(".ipynb"):
                continue
            if ".ipynb_checkpoints" in path:
                continue

            ipynb = os.path.join(path, name)
            ok = run_jupytext_ipynb_to_rmd(ipynb)
            converted_any = converted_any or ok

    if not converted_any:
        # Not an error—maybe there are no ipynb files, or jupytext isn't
        # installed.
        pass

    # ---- Pass 2: .Rmd -> .py ----------------------------------------------
    for path, folders, files in os.walk(base_dir, topdown=True):
        folders[:] = [name for name in folders if not is_hidden(name)]
        for name in files:
            if is_hidden(name):
                continue
            if not name.endswith(".Rmd"):
                continue

            rmd = os.path.join(path, name)

            # Normalise jupytext/kernelspec metadata in YAML header
            normalise_rmd_metadata(rmd)

            # Convert .Rmd -> .py
            convert_rmd_to_py(rmd)


if __name__ == "__main__":
    main()
