These instructions are just for Microsoft Windows; see also the
[general instructions](./README.md).

# Installing Git

DySys is maintained on [GitHub](https://github.com/gdmcbain/DySys)
using [Git](https://git-scm.com).

# Installing conda

Installing DySys under Microsoft Windows with
[Miniconda](https://docs.conda.io/en/latest/miniconda.html) is
straightforward.  It is not implied that DySys won't work on Microsoft
Windows under other Python distributions, but only Anaconda and Miniconda have been
tested to date.

## Create a fresh conda environment

Open an *Anaconda Powershell Prompt* and
```shell
conda create -n DySys scipy pandas toolz git
conda activate DySys
conda config --add channels conda-forge
```
The name `DySys` here is arbitrary, but remember it.


## Install the local GitHub clone

```shell
git clone git@github.com:gdmcbain/DySys
pip install ./DySys
```
