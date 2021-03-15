These instructions are just for Anaconda or Miniconda; see also the [general
instructions](./README.md).

Like Anaconda/Miniconda, they should work on either GNU/Linux or Microsoft Windows.

This installation is to be done in an activated conda environment.  If already in one, skip this step.

```shell
conda create -n dysys numpy # or whatever
conda activate dysys
```

Then change to the usual `src` directory.  This can be wherever you like but I
put it in `$HOME` on Linux and `$env:userprofile` under MS-Windows; thus either

```shell
cd ~/src
```
or
```shell
cd $env:userprofile\src
```
Then
```shell
git clone git@gitlab.memjet.local:msm/dysys
pip install -e ./dysys
```
