#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Encapsulate Cholesky decomposition.

Fall back on SciPy if CHOLMOD is unavailable.

scipy.linalg has cho_factor and cho_solve, but they're dense;
sksparse.choldmod is sparse, but not easily installed, especially
under MS-Windows

:author: G. D. McBain <gmcbain>

:created: 2016-09-13

'''

from __future__ import absolute_import, division, print_function

from functools import partial
from warnings import warn

from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import issparse


def dense_cholesky(a):
    '''Compute the Cholesky decomposition of a

    with the property that calling it on a right-hand side vector
    b returns the solution x of a.dot(x) = b.

    :param a: np.ndarray, ndim==2, symmetric, positive-definite

    :rtype: function (np.ndarray, ndim=1, len=n) -> (np.ndarray,
    ndim=1, len=n), linear

    '''

    return partial(cho_solve,
                   cho_factor(a.toarray() if issparse(a) else a))


try:

    # TRICKY gmcbain 2017-07-31: sksparse.cholmod is not available for
    # Microsoft Windows.
    
    from sksparse.cholmod import cholesky as sparse_cholesky
except ImportError:

    # TODO gmcbain 2017-07-31: Consider adopting the pure Python
    # implementation from msmdir.010831/cholesky.py.
    
    warn('could not import cholesky from sksparse.cholmod,'
         ' falling back on scipy.linalg, which is dense', UserWarning)
    sparse_cholesky = dense_cholesky


def cholesky(a):
    return (sparse_cholesky if issparse(a) else dense_cholesky)(a)
