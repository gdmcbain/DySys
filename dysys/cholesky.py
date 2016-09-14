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

from scipy.sparse import issparse

try:
    from sksparse.cholmod import cholesky
except ImportError:
    warn('could not import cholesky from sksparse.cholmod,'
         ' falling back on scipy.linalg, which is dense', UserWarning)
    from scipy.linalg import cho_factor, cho_solve

    def cholesky(a):
        '''Compute the Cholesky decomposition of a

        with the property that calling it on a right-hand side vector
        b returns the solution x of a.dot(x) = b.

        :param a: np.ndarray, ndim==2, symmetric, positive-definite

        :rtype: function (np.ndarray, ndim=1, len=n) -> (np.ndarray,
        ndim=1, len=n), linear

        '''

        c = cho_factor(a.todense() if issparse(a) else a)

        return partial(cho_solve, c)
