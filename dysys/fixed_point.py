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
        self.iteration = iteration
        self.tol = tol
        self.maxiter = maxiter

    def __iter__(self):
        return self

    def next(self):
        while True:
            y, h = next(it.islice(self.iteration, self.maxiter))
            print self.maxiter, y
            self.maxiter -= 1
            if np.linalg.norm(h) < self.tol:
                break
        return y

    def __call__(self):
        return next(self)


def newton(residual, jacobian, x, *args, **kwargs):
    'eliminate the residual by Newton-iteration'
    def iteration(x):
        while True:
            try:
                dx = spsolve(jacobian(x), residual(x))
            except ValueError:
                print 'residual: {0} ({1}, {2})'.format(
                    residual(x), type(residual(x)), residual(x).shape)
                print 'jacobian: {0} ({1}, {2})'.format(
                    jacobian(x), type(jacobian(x)), jacobian(x).shape)
                dx = residual(x) / jacobian(x)
            x = (x[0] - dx,) + x[1:]
            yield x, dx

    return FixedPoint(iteration(x), *args, **kwargs)()
