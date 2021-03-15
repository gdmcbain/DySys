These instructions are just for Anaconda or Miniconda; see also the [general
instructions](./README.md).

Like Anaconda/Miniconda, they should work on either GNU/Linux or Microsoft Windows.

This installation is to be done in an activated conda environment.  If already in one, skip this step.

```shell
conda create -n dysys numpy # or whatever
conda activate dysys
```

Then:

```shell
cd  # so that sources go in ~/src
git clone git@gitlab.memjet.local:msm/dysys
pip install -e ./dysys
```
