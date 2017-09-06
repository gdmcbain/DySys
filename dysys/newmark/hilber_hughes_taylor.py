#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''This implements the Hilber-Hughes-Taylor (a.k.a. alpha-) method

(Hilber 1976; Hilber, Hughes, & Taylor 1977; Cook, Malkus, & Plesha
1989, p. 409; Hughes 2000, p. 532; Craveur 2008, §18.4.5), as a
generalization of the Newmark method

:author: G. D. McBain <gmcbain>

:created: 2016-04-06

'''

from __future__ import absolute_import, division, print_function

from functools import partial

import numpy as np
from scipy.interpolate import interp1d

from .newmark import Newmark


class HilberHughesTaylor(Newmark):
    '''a dynamical system advancing with a Hilber-Hughes-Taylor method

    (a.k.a. alpha-method), like the Newmark method

    '''

    def __init__(self, M, K, C=None, f=None, alpha=0., definite=False,
                 **kwargs):
        ''':param M: mass scipy.sparse matrix

        :param K: stiffness scipy.sparse

        :param C: damping scipy.sparse matrix, or None, in which case
        it is constructed as like M but with no nonzero entries

        :param f: function of time and dict of discrete dynamical
        variables, returning forcing vector, or None in which case a
        null function is substituted

        :param alpha: float, Hilber-Hughes-Taylor method parameter,
        should be in [-1/3, 0]

        :param definite: bool, for if system is (positive-)definite

        Further keyword parameters are passed on to DySys.__init__; in
        particular: 'parameters' and 'master'.

        '''

        self.alpha = alpha
        super(HilberHughesTaylor, self).__init__(
            M, K, C, f, (1 - alpha)**2 / 4., (1 - 2 * alpha) / 2., definite,
            **kwargs)

    def step(self, t, h, x, d, *args):
        'evolve from displacement x at time t to t+h'

        self.prestep(t, h, x, d, *args)

        xt = (x[0] + h * (x[1] + h * (.5 - self.beta) * self.a),
              x[1] + (1 - self.gamma) * h * self.a)

        rhs = -self.K.dot(interp1d([-1, 0],
                                   np.vstack([x[0], xt[0]]).T)(self.alpha))

        if self.C is not None:
            rhs -= self.C.dot(interp1d([-1, 0],
                                       np.vstack([x[1], xt[1]]).T)(self.alpha))

        rhs += interp1d([-1, 0],
                        np.vstack(self.forcing(
                            t, h, x, d, *args)).T)(self.alpha)

        self.a = self.solve(rhs)
        return (xt[0] + self.beta * h**2 * self.a,
                xt[1] + self.gamma * h * self.a)

    def setA(self, h):
        super(HilberHughesTaylor, self).setA(h, self.alpha)

    def constrain(self, known, xknown=None, vknown=None, aknown=None):
        '''return a new DySys with constrained degrees of freedom

        having the same class as self.

        :param known: sequence of indices of known degrees of freedom

        :param xknown: corresponding sequence of their values
        (default: zeros)

        :param vknown: corresponding sequence of their rates of change

        :param aknown: corresponding sequence of their second
        derivatives

        '''

        # TODO gmcbain 2016-07-27: Refactor!

        U, Kn = self.node_maps(known)
        project = partial(self.projector, U)

        M, K, C = [None if A is None else project(A * U)
                   for A in [self.M, self.K, self.C]]
        sys = self.__class__(
            M,
            K,
            C,
            lambda *args: project(
                (0 if self.f is None else self.f(*args)) -
                (0 if xknown is None else self.K.dot(Kn.dot(xknown))) -
                (0 if vknown is None else self.C.dot(Kn.dot(vknown))) -
                (0 if aknown is None else self.M.dot(Kn.dot(aknown)))),
            self.alpha, self.definite)

        reconstituter = partial(self.reconstituter, U, Kn)
        def reconstitute(xv):
            return (reconstituter(xknown, xv[0]),
                    reconstituter(vknown, xv[1]))
            
        sys.reconstitute = reconstitute
        sys.project = project

        return sys
