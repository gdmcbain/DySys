#!/usr/bin/env python

'''

:author: G. D. McBain <gmcbain>
:created: 2013-04-09

'''

from __future__ import absolute_import, division, print_function

import numpy as np

from dysys import node_maps
from .linear_dysys import LinearDySys
from .fixed_point import newton


class NonlinearSparseDySys(LinearDySys):

    def __init__(self, F, M, D, n=None):
        '''an alternative to SparseNFDySys

        The system evolves according to the more general F(t, x, x') = 0.

        :param: F(t, x, v), where v is understood to be the rate of
        change of x

        :param: M(t, x, v), returning the partial derivative of F w.r.t. v

        :param: D(t, x, v), returning the partial derivative of F w.r.t. x

        :param: n, order of system, i.e. len of F; calculated from
        D(0, [], []) if omitted (which will only work if D doesn't
        inspect its arguments)

        '''

        self.F, self.M, self.D = F, M, D
        self.n = D(0, [], []).shape[0] if n is None else n

    def __len__(self):
        return self.n

    def step(self, t, xold, h, tol=1e-3):
        '''take a backward-Euler step'''

        if h == 0:
            raise ZeroDivisionError

        def arg_map(x):
            return t + h, x, (x[0] - xold[0]) / h

        def residual(x):
            '''approximate the rate of change using backward Euler'''
            return self.F(*arg_map(x))

        def jacobian(x):
            # r(x + dx) = F(t, x + dx, (x + dx - xold) / h)

            #          ~= r(x) + (F_x + F_v / h) dx

            # Thus J = F_x + F_v / h.

            return self.M(*arg_map(x)) / h + self.D(*arg_map(x))

        return newton(residual, jacobian, xold, tol)

    def equilibrium(self, x0, tol=1e-3):
        '''take an infinitely long backward-Euler step'''

        def residual(x):
            # r(x) = F(oo, x, 0)
            return self.F(np.inf, x, np.zeros(x[0].shape))

        def jacobian(x):
            # r(x+dx) = F(oo, x + dx, 0) ~ r(x) + D(t, x) dx
            return self.D(np.inf, x)

        return newton(residual, jacobian, x0, tol)

    def constrain(self, known, xknown=None, vknown=None):
        '''return a new NonlinearSparseDySys with constrained DoFs

        :param known: sequence of indices of known degrees of freedom

        :param xknown: corresponding sequence of their values
        (default: zeros)

        :param vknown: corresponding sequence of their rates of change

        The returned system is attributed the U and K matrices from
        self.node_maps and therefore can use :method reconstitute:.

        '''
        
        U, K = node_maps(known, len(self))

        def reconstitute(u):
            '''put back the known degrees of freedom constrained out'''

            return ((U.dot(u[0]) + (0 if xknown is None else K.dot(xknown))),
                    ) + u[1:]

        def arg_map(t, u, u1):
            '''transform the arguments for the constraining'''
            return (t,
                    reconstitute(u),
                    U.dot(u1) + (0 if vknown is None else K.dot(vknown)))

        sys = self.__class__(
            lambda t, u, u1: U.T.dot(self.F(*arg_map(t, u, u1))),
            lambda t, u, u1: U.T.dot(self.M(*arg_map(t, u, u1))).dot(U),
            lambda t, u, u1: U.T.dot(self.D(*arg_map(t, u, u1))).dot(U),
            U.shape[1])

        sys.reconstitute = reconstitute

        return sys

