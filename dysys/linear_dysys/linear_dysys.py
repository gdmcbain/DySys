#!/usr/bin/env python

'''a module for linear 'descriptor' systems

:author: G. D. McBain <gmcbain>

:created: 2013-01-11

'''

from __future__ import absolute_import, division, print_function

from functools import partial

from dysys import DySys


class LinearDySys(DySys):

    def __init__(self, M, D, f=None, theta=1.0, definite=False,
                 **kwargs):
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

        Further keyword parameters are passed on to DySys.__init__; in
        particular: 'parameters' and 'master'.

        '''

        self.M, self.D, self.f = M, D, f
        self.theta = theta
        self.definite = definite
        super(LinearDySys, self).__init__(**kwargs)

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
        project = partial(self.projector, U)

        M, D = [None if A is None else project(A * U)
                for A in [self.M, self.D]]
        sys = self.__class__(
            M,
            D,
            lambda *args: project(
                (0 if self.f is None else self.f(*args)) -
                (0 if xknown is None else self.D.dot(K.dot(xknown))) -
                (0 if vknown is None else self.M.dot(K.dot(vknown)))),
            self.theta,
            self.definite)

        sys.reconstitute = partial(self.reconstituter, U, K, xknown)
        sys.project = project

        return sys

    def harmonic(self, omega):
        '''return the complex harmonic solution

        :param omega: sequence of floats (typically positive)

        '''

        # M x' + D x - f (t) = 0, with f(t) = F exp (j w t), X = s exp (j w t)

        # (D + j w M) X - F = 0

        return [self.__class__(None,
                               self.D + 1j * w * self.M,
                               self.f).equilibrium()
                 for w in omega]

