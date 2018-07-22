These instructions are just for Microsoft Windows; see also the
[general instructions](./README.md).

# Installing Git

DySys is maintained on [GitLab](https://gitlab.memjet.local/msm/DySys)
using [Git](https://git-scm.com).  Install Git from
<https://git-scm.com/download/win>;
e.g. [Git-2.18.0-64-bit.exe](https://github.com/git-for-windows/git/releases/download/v2.18.0.windows.1/Git-2.18.0-64-bit.exe).

## Configuring Git

Launch a shell from the Microsoft Windows start menu; e.g.
- *Command Prompt*
- *Git Bash* 
- *Git CMD* 

```shell
git config --global user.email %USERNAME%@memjet.com
git config --global http.sslVerify false
```

# Installing Anaconda

Installing DySys under Microsoft Windows with
[Anaconda3](https://www.anaconda.com/download/#windows) is
straightforward.  It is not implied that DySys won't work on Microsoft
Windows under other Python distributions, but only Anaconda has been
tested to date.  The latest tested is
[Anaconda3-5.0.1-Windows-x86_64.exe](
https://repo.continuum.io/archive/Anaconda3-5.0.1-Windows-x86_64.exe).
The latest seen but not yet tested is
[Anaconda3-5.2.0-Windows-x86_64.exe](https://repo.anaconda.com/archive/Anaconda3-5.2.0-Windows-x86_64.exe).

After installing Anaconda, launch the *Anaconda Navigator* desktop
application.  This can be done by hitting the Windows-symbol key and
typing 'Anaconda Navigator'.

## Create a fresh Anaconda environment

Click 'Environments/Create' and in the popup 'Create new environment' under 'Name' enter, e.g., DySys.  (Choose whatever name you like, but remember it.)

By default, the Packages are checked for Python (rather than R) and version 3.6; leave the defaults checked.

Click Create.

## Install Anaconda packages

Some of the Python packages required for DySys are provided by Anaconda.

On creation, DySys should have been appended to the list of environments, which by default consisted of just root.  Click on DySys to select it in the list of environments.

In the package menu, change the filter from 'Installed' to 'All' and then in the ‘search packages’ box, type:
* SciPy
* pandas
* toolz
* flake8

checking each and then installing them all together.

## Install the local GitLab packages

Open a terminal in the DySys Anaconda environment (by clicking on the
triangular ‘play’ symbol in the Navigator) and in it issue::

```shell
pip install git+http://gitlab.memjet.local/msm/DySys#egg=DySys
```
