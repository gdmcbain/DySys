DySys
=====

DySys is a Python package to encapsulate some relatively abstract
classes for dynamical systems, particularly those of descriptor type
which cannot be simulated by `scipy.integrate.odeint
<https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.integrate.odeint.html>`_.

Installation
------------



Installation under GNU/Linux
````````````````````````````

It is recommended to install pyMMJ in a virtualenv, as for its prerequisites

* `DySys <https://gitlab.memjet.local/msm/DySys>`_ and 
* `Beamel <https://gitlab.memjet.local/msm/Beamel>`_.

pyMMJ requires Python 3; Python 2 is no longer supported.  It isn't necessary to
use exactly version 3.6; that's just for the example instructions.

Also the directory for the virtualenv needn't be called `pyMMJ`; it can be
called whatever and placed wherever.::

   python3.6 -m venv --without-pip pyMMJ
   . $_/bin/activate
   wget https://bootstrap.pypa.io/get-pip.py
   python get-pip.py
   rm get-pip.py
   pip install -e git+git@gitlab.memjet.local:msm/DySys#egg=DySys

To test (optional)::

   pip install nose
   nosetests dysys

(Running `nose` before before activating the virtualenv can confuse
bash; if this happens, try `hash -r`.  See `nose-devs#973
<https://github.com/nose-devs/nose/issues/973>`_.)
   
When finished::

   deactivate
   
To recommence::

   . pyMMJ/bin/activate 
   
prepending the path to `pyMMJ` if required because of having changed to a
different directory.

Installation under GNU/Linux
````````````````````````````

For Microsoft Windows, see `INSTALL-MSWindows.rst
<./INSTALL-MSWindows.rst>`_.
