#!/usr/bin/env python

'''
:author: G. D. McBain <gmcbain>
:created: 2013-08-26

'''

from __future__ import absolute_import, division, print_function

import itertools as it

import numpy as np
from scipy.sparse.linalg import spsolve


def fixed_point(iteration, tol=np.MachAr().eps, maxiter=np.iinfo(np.int).max):
    '''

    :param: iteration, iterable generating pairs of values and
    convergence test values

    :param: tol, positive float

    :param: maxiter, positive integer

    When called (without arguments), the object returns the first
    element of the first pair for which the second value is less than
    tol (in the sense of numpy.linalg.norm)

    '''

    return next(y for y, h in it.islice(iteration, maxiter)
                if np.linalg.norm(h) < tol)


def newton(residual, jacobian, x, *args, **kwargs):
    '''eliminate the residual by Newton-iteration

    :param: residual, a function taking a tuple with first term a
    one-dimensional numpy.array to another array of the same len

    :param: jacobian, a function like residual but returning a square
    numpy.array of corresponding shape

    :param: x, a tuple with first term a one-dimensional numpy.array
    of the length expected by the residual and jacobian functions

    Any other positional or keyword arguments are passed on to
    fixed_point; of particular interest are tol and maxiter.

    '''

    def iteration(x):
        while True:
            dx = solve(jacobian(x), residual(x))
            x = x - dx
            yield x, dx

    return fixed_point(iteration(x), *args, **kwargs)


def solve(A, b):
    try:
        return spsolve(A, b)
    # except ValueError:
    #             return b / A
    except IndexError:
        return b / A.toarray()[0, 0]
