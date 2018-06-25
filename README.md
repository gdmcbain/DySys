# DySys

DySys is a Python package to encapsulate some relatively abstract
classes for dynamical systems, particularly those of descriptor type
which cannot be simulated by
[`scipy.integrate.odeint`](https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.integrate.odeint.html).

## Installation

### Installation under GNU/Linux

It is recommended to install DySys in a virtualenv.

DySys requires Python 3; Python 2 is no longer supported.  It isn't necessary to
use exactly version 3.6; that's just for the example instructions.

Also the directory for the virtualenv needn't be called `DySys` and it
needed be kept in `$HOME/.py3.6`; it can be called whatever and placed
wherever.

```shell
python3.6 -m venv ~/.py3.6/DySys  # or wherever
. $_/bin/activate
pip install numpy  # Shouldn't really need this but do
pip install -e git+ssh://git@gitlab.memjet.local/msm/DySys#egg=DySys
```

## Testing

To test (optional)

```shell
pip install pytest-xdist
pytest --pyargs dysys -n auto
```

(Actually thus far the test suite is quite small so distributing the
tests amongst the local numprocessors is slower, but this is left here
to document the use of xdist for other packages.)

## Deactivation & reactivation

When finished:

```shell
deactivate
```

To recommence:

```shell
. DySys/bin/activate 
```

prepending the path to `DySys` if required because of having changed to a
different directory.

# Installation under Microsoft Windows

For Microsoft Windows, see
[Installation/Microsoft-Windows](https://gitlab.memjet.local/msm/DySys/wikis/installation/Microsoft-Windows).
