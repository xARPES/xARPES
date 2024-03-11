# Exubi

Private repository for the code Eliashberg eXtraction Using Bayesian Inference.

## Installation

Instructions provided here are for Linux Ubuntu v22 or later.

### pip

The user will have to set up a pristine virtual environment. First, ``python3-venv`` might have to be installed:

	sudo apt install python3-venv

Afterwards, one can activate a virtual environment named ``<my_venv>`` using:

	python3 -m venv <my_venv>

It has to be activated whenever installing/running Exubi:

	source <my_venv>/bin/activate

It is recommended to upgrade pip to the latest version:

	python3 -m pip install --upgrade pip

Finally, the installation can be performed:

	git clone git@github.com:TeetotalingTom/exubi.git
	python3 -m pip install -e exubi/

## conda

The user is assumed to be in a pristine virtual environment provided by conda. First, download the required version for your operating system from <https://docs.anaconda.com/free/miniconda/>. For example:

	wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

Start the installation:

	bash Miniconda3-latest-Linux-x86_64.sh

Then scroll down the license agreement and answer ``yes`` to the following question:

	Do you accept the license terms? [yes|no]

Also specify your installation location.

It is convenient to also answer ``yes`` to the following, which will append new lines to your ``~/.bashrc``:

	You can undo this by running `conda init --reverse $SHELL`? [yes|no]

A conda base environment is then activated with ``source ~/.bashrc`` or by starting a new terminal session.

Alternatively, you can answer ``no`` to the above question and activate conda whenever you need it:

	eval "$(/YOUR/PATH/TO/miniconda3/bin/conda shell.YOUR_SHELL_NAME hook)"

Next, we install ``conda-build`` for developing Exubi (anwer ``y`` to questions):

	conda install conda-build

Finally, the following steps are executed for the installation &ndash; the ``exubi`` environment will have to be launched whenever running Exubi:

	git clone git@github.com:TeetotalingTom/exubi.git
	conda create --name exubi --file exubi/requirements.txt
	conda activate exubi
	conda develop exubi/

Answer ``y`` to questions.

## Execution

It is recommended to use JupyterLab to analyse data. JupyterLab is launched using:

	jupyter lab
