# Exubi
Private repository for the code Eliashberg eXtraction Using Bayesian Inference

## Installation

Instructions provided here are for Linux Ubuntu V22 or later. 

### pip

The user will have to set up a pristine virtual environment. First, ``python3-venv`` might have to be installed:

	sudo apt install python3-pip python3-venv

Afterwards, one can activate a virtual environment named ``<my_venv>`` using:

	python3 -m venv <my_venv>
	
To be activated whenever installating/running Exubi:

	source <my_venv>/bin/activate

It is recommended to upgrade pip to the latest version:

	python3 -m pip install --upgrade pip
	
After which the installation can be performed:

	git clone git@github.com:TeetotalingTom/exubi.git
	cd exubi/
	python3 -m pip install -e .

## conda

The user is assumed to be in a pristine virtual environment provided by conda. First, download the required version for your operating system using:

	https://docs.anaconda.com/free/miniconda/
	
For example:

	wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
	bash Miniconda3-latest-Linux-x86_64.sh
	
Then scroll down the license agreement and answer ``yes`` to the following question:

	Do you accept the license terms? [yes|no]
	
Followed by specifying your installation location. It is most convenient to also answer ``yes`` to the following:

	You can undo this by running `conda init --reverse $SHELL`? [yes|no]
	
A conda base environment is then activated with ``source ~/.bashrc`` which runs the new lines appended to ``~/.bashrc``. Next, we install ``conda-build`` for developing Exubi (anwer ``y`` to questions):

	conda install conda-build
	
Next, the following steps are executed for the installation - the exubi environment will have to be launched whenever running exubi:

	git clone git@github.com:TeetotalingTom/exubi.git
	cd exubi/
	conda create --name exubi --file requirements.txt
	conda activate exubi
	conda develop .
	
And answer ``y`` to questions.

## Execution

It is recommended to user JupyterLab to analyse data. JupyterLab is launched using:

	jupyter lab
