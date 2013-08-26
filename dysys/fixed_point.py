#!/usr/bin/env python

'''
:author: G. D. McBain <gmcbain>
:created: 2013-08-26

'''

import itertools as it

import numpy as np
from scipy.sparse.linalg import spsolve


class FixedPoint(object):

    '''fixed-point iteration'''

    def __init__(self, iteration,
                 tol=np.MachAr().eps, maxiter=np.iinfo(np.int).max):

        '''

        :param: iteration, iterable generating pairs of values and
        convergence test values

        :param: tol, positive float

        :param: maxiter, positive integer

        When called (without arguments), the object returns the first
        element of the first pair for which the second value is less
        than tol (in the sense of numpy.linalg.norm)

        '''

        self.iteration = iteration
        self.tol = tol
        self.maxiter = maxiter

    def __iter__(self):
        return self

    def next(self):
        while True:
            y, h = next(it.islice(self.iteration, self.maxiter))
            self.maxiter -= 1
            if np.linalg.norm(h) < self.tol:
                break
        return y

    def __call__(self):
        return next(self)


def newton(residual, jacobian, x, *args, **kwargs):
    '''eliminate the residual by Newton-iteration

    :param: residual, a function taking a tuple with first term a
    one-dimensional numpy.array to another array of the same len

    :param: jacobian, a function like residual but returning a square
    numpy.array of corresponding shape

    :param: x, a tuple with first term a one-dimensional numpy.array
    of the length expected by the residual and jacobian functions

    Any other positional or keyword arguments are passed on to the
    FixedPoint constructor; of particular interest are tol and
    maxiter.

    '''

    def iteration(x):
        while True:
            try:
                dx = spsolve(jacobian(x), residual(x))
            except ValueError:
                dx = residual(x) / jacobian(x)
            x = (x[0] - dx,) + x[1:]
            yield x, dx

    return FixedPoint(iteration(x), *args, **kwargs)()
