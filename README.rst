DySys
=====

DySys is a Python package to encapsulate some relatively abstract
classes for dynamical systems, particularly those of descriptor type
which cannot be simulated by `scipy.integrate.odeint
<https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.integrate.odeint.html>`_.

Installation
------------

DySys is maintained at https://gitlab.memjet.local/msm/DySys.


Installation under Windows
``````````````````````````

Install `Anaconda3 <https://www.continuum.io/downloads#windows>`_.

Launch the *Anaconda Navigator* and in it create a new environment.
Call it, e.g., 'DySys'.  Add the packages:

  * SciPy

  * pandas

  * toolz


Open a terminal in the DySys Anaconda environment and issue::

    pip install -e git+http://gitlab.memjet.local/msm/DySys#egg=DySys
