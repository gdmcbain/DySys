#!/usr/bin/env python

'''

:author: G. D. McBain <gmcbain>
:created: 2013-04-09

'''

from __future__ import absolute_import, division, print_function

from .linear_dysys import LinearDySys
from .fixed_point import newton


class NonlinearSparseDySys(LinearDySys):
    
    def __init__(self, F, M, D):
        '''an alternative to SparseNFDySys 

        The system evolves according to the more general F(t, x, x') = 0.

        :param: F(t, x, v), where v is understood to be the rate of
        change of x

        :param: M(t, x), returning the partial derivative of F w.r.t. v

        :param: D(t, x), returning the partial derivative of F w.r.t. x

        '''

        self.F, self.M, self.D = F, M, D

    def step(self, t, xold, h, tol=1e-3):
        '''take a backward-Euler step'''

        if h == 0:
            raise ZeroDivisionError

        def residual(x):
            '''approximate the rate of change using backward Euler'''
            return self.F(t + h, x, (x[0] - xold[0]) / h)

        def jacobian(x):
            # r(x + dx) = F(t, x + dx, (x + dx - xold) / h) 

            #          ~= r(x) + (F_x + F_v / h) dx

            # Thus J = F_x + F_v / h.
            return self.M(t + h, x) / h + self.D(t + h, x)

        return newton(residual, jacobian, xold, tol)
