![xARPES](https://xarpes.github.io/_images/xarpes.svg)

Repository for the code xARPES &ndash; extraction from angle resolved photoemission spectra.

# Installation

xARPES can be installed with `pip`:

	python3 -m pip install xarpes

Or with `conda`:

	conda install conda-forge::xarpes

More detailed instructions for installing the development version, tested for recent Ubuntu and Debian GNU/Linux, are provided below.

## pip

It is highly recommended to set up a pristine Python virtual environment. First, the `venv` module might have to be installed:

	sudo apt install python3-venv

Afterwards, one can activate a virtual environment named `<my_venv>` using:

	python3 -m venv <my_venv>

It has to be activated whenever installing/running xARPES:

	source <my_venv>/bin/activate

It is recommended to upgrade `pip` to the latest version:

	python3 -m pip install --upgrade pip

Finally, the installation can be performed:

	git clone git@github.com:xARPES/xARPES.git
	cd xARPES
	python3 -m pip install -e .

## conda

The user is assumed to be in a pristine virtual environment provided by conda. First, download the required version for your operating system from <https://docs.anaconda.com/free/miniconda/>. For example:

	wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

Start the installation:

	bash Miniconda3-latest-Linux-x86_64.sh

Then press Enter to get to the end of the license agreement, and answer `yes` to the following question:

	Do you accept the license terms? [yes|no]

Also specify your installation location.

It is convenient to also answer `yes` to the following, which will append new lines to your `~/.bashrc`:

	You can undo this by running `conda init --reverse $SHELL`? [yes|no]

A conda base environment is then activated with `source ~/.bashrc` or by starting a new terminal session.

Alternatively, you can answer `no` to the above question and activate conda whenever you need it:

	eval "$(<your_path>/miniconda3/bin/conda shell.<your_shell> hook)"

Next, we install `conda-build` for developing xARPES (answer `y` to questions):

	conda install conda-build

Finally, the following steps are executed for the installation &ndash; the `<my_env>` environment will have to be launched whenever running xARPES:

	git clone git@github.com:xARPES/xARPES.git
	cd xARPES
	conda create -n <my_env> -c defaults -c conda-forge
	conda activate <my_env>
        pip install -e .

Answer `y` to questions.

# Examples

After installation of xARPES, the `examples/` folder can be downloaded to the current directory:

	xarpes_download_examples

Equivalently:

	python3 -c "import xarpes; xarpes.download_examples()"

# Execution

It is recommended to use JupyterLab to analyse data. JupyterLab is launched using:

	jupyter lab

# License

Copyright (C) 2025 xARPES Developers

This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License, version 3, as published by the Free Software Foundation.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
