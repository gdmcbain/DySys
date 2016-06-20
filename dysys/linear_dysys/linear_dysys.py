#!/usr/bin/env python

'''a module for linear 'descriptor' systems

:author: G. D. McBain <gmcbain>

:created: 2013-01-11

'''

from __future__ import absolute_import, division, print_function

from functools import partial

from dysys import DySys


class LinearDySys(DySys):

    def __init__(self, M, D, f=None, theta=1.0, definite=False):
        '''a DySys defined by mass and damping operators

        and a time-dependent forcing function, according to (something
        like)

            M * x' + D * x = f (t, d)

        though this class is still virtual since it depends on:

          . the implementation of the M & D operators (e.g. as sparse)

          . the discretization of the temporal derivative.

        Since occasionally the steady-state D * x = f (inf) is of
        interest, M may be None.

        :param M: mass operator (abstract)

        :param D: damping operator (abstract)

        :param f: function of time and dict of discrete dynamical
        variables, returning right-hand side (default zero function)

        :param theta: float, parameter of theta time-stepping method,
        default 1.0 for backward Euler, 0.5 for trapezoidal, 0 for
        forward Euler

        :param definite: bool, for if system is (positive-)definite

        '''

        self.M, self.D, self.f = M, D, f
        self.theta = theta
        self.definite = definite

    def __len__(self):
        return self.D.shape[0]

    def constrain(self, known, xknown=None, vknown=None):
        '''return a new DySys with constrained degrees of freedom

        having the same class as self.

        :param known: sequence of indices of known degrees of freedom

        :param xknown: corresponding sequence of their values
        (default: zeros)

        :param vknown: corresponding sequence of their rates of change

        '''

        U, K = self.node_maps(known)
        M, D = [None if A is None else U.T * A * U for A in [self.M, self.D]]
        sys = self.__class__(
            M,
            D,
            lambda *args: U.T.dot(
                (0 if self.f is None else self.f(*args)) -
                (0 if xknown is None else self.D.dot(K.dot(xknown))) -
                (0 if vknown is None else self.M.dot(K.dot(vknown)))))

        sys.reconstitute = partial(self.reconstituter, U, K, xknown)
        return sys
