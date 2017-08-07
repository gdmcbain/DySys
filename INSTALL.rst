Installation
============

These instructions are for GNU/Linux systems; for Microsoft Windows,
see `INSTALL-MSWindows.rst <./INSTALL-MSWindows.rst>`_.

These instructions are for Python 3; Python 2 is no longer supported.

virtualenv
----------

Create a virtualenv and launch it::

  mkdir -p ~/projects/python/envs
  virtualenv ~/projects/python/envs/DySys
  . ~/projects/python/envs/DySys/bin/activate

When done, deactivate the virtualenv with::

  deactivate

but the rest of these instructions assume that the same virtualenv is
activated (or reactivated).

git
---

::
   git config --global --bool http.sslVerify false

Install
-------

::
   pip install scikit-sparse
   pip install -e git+http://gitlab.memjet.local/msm/DySys#egg=DySys
