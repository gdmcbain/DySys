These instructions are just for Anaconda or Miniconda; see also the [general
instructions](./README.md).

Like Anaconda/Miniconda, they should work on either GNU/Linux or Microsoft Windows.

This installation is to be done in an activated conda environment.  If already in one, skip this step.

```shell
conda create -n dysys # or whatever
conda activate dysys
```

Then:

```shell
conda install numpy pip
cd  # so that sources go in ~/src
pip install -e git+git@gitlab.memjet.local:msm/dysys.git#egg=dysys
```
