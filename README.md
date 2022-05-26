# Preparation

## Install Conda environment
If you don't have conda package and environment manager you can install via the following steps for Linux OS:
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
export PATH="${PATH}:~/miniconda3/bin"
~/miniconda3/bin/conda init bash && source ~/.bashrc && conda config --set auto_activate_base false
```

## Prepare Conda environment
For refresh conda commands, you can take a look into [official Conda cheat sheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf). 

Commands below will install all needed software as a part of an environment named **fl**, or you can use the name that you prefer.

For **Windows OS, Linux OS** please use the following commands to prepare the environment:

```bash
conda create -n fl python=3.9.1 -y
conda install -n fl pytorch"=1.10.0" torchvision numpy cudatoolkit"=11.1" h5py"=3.6.0" coloredlogs matplotlib psutil pyqt pytest pdoc3 wandb -c pytorch -c nvidia -c conda-forge -y
```

For **Mac OS** CUDA is not currently available, please install PyTorch without CUDA support:

```bash
conda create -n fl python=3.9.1 -y
conda install -n fl pytorch"=1.10.0" torchvision numpy h5py"=3.6.0" coloredlogs matplotlib psutil pyqt pytest pdoc3 wandb -c pytorch -c nvidia -c conda-forge -y
```


## Install packages via pip and virtualenv

Alternatively, instead of using conda, you can use a standard virtual environment in Python, which sometimes is used instead of Conda.

```bash
python -m pip install virtualenv
python -m venv flpip
source flpip/bin/activate
python -m pip install -r requirements.txt -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

## Ability to edit UI files for Qt5

In case if you want to modify the user interface in the GUI tool via appending extra elements, please make one of the following:

1. Completely install Qt5 SDK, 
2. Install the package with Qt5 tools only, which includes Qt Designer.

# Launch experiments from command line

```bash
conda activate fl
cd ./simulator
./australian_distributed_experiment_1_client_per_round.sh
./australian_distributed_experiment_3_client_per_round.sh
./australian_distributed_experiment_6_client_per_round.sh
./australian_distributed_experiment_9_client_per_round.sh
```

The result of command execution are binary files available in the filesystem. By default computation are performed in CPU. If you want to carry computation in NVIDIA GPU please change the "--gpu" command line parameter to GPU number, starting from 0.

# Observe results of experiments

1 Activate you Python environment via "conda activate fl"
2. Change working directory to "./simulator/gui" and launch "start.py" with Python intepreter from install enviroment
3. Select "File->Load" and select ".bin" output files from the previous step
4. Open "Analysis" tab inside GUI, select experiments to visualize. Select need axis for OX and OY. For example for plots from the paper you should select "Sample gradient oracles (train)" for OX and "Norm of function gradient square(train)" for OY.
5. Press button "Plot graphics for selected experiments"


# Cleaning after work

Remove Conda environment in case if you are using Conda: 
```bash
conda remove --name fl --all
```
And please remove all not need files in the filesystem, including logs, checkpoints, saved experiments in binary format.


----

# About the License

The project is distributed under [Apache 2.0 license](LICENSE).
