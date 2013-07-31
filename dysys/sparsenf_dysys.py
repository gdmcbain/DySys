#!/usr/bin/env python

'''
:author: G. D. McBain <gmcbain>
:created: 2013-01-23

'''

from copy import copy

import numpy as np

from scipy.optimize import fsolve
from scipy.sparse.linalg import spsolve

from linear_dysys import LinearDySys


class SparseNFDySys(LinearDySys):

    '''a DySys with state and time-dependent forcing force and

    sparse mass and damping operators

            M * x' + D * x = f(t, x)

    The residual for a backward Euler step of h from x to y is

           (M / h + D) * y - f(t, y) - M / h * x

    as returned by self.residual and the Jacobian of that with respect
    to y is

            (M / h + D) - (df/dy)(t, y)

    as returned by self.jacobian.

    '''

    def __init__(self, M, D, f, f1=None):
        '''

        :param f1: the Jacobian derivative of f with respect to x; if
        provided, it will be used to compute the jacobian in the step
        and equilibrium methods

        '''

        self.f1 = f1
        super(SparseNFDySys, self).__init__(M, D, f)

    def step(self, t, xold, h, tol=1e-3):

        # KLUDGE gmcbain 2013-04-08: An unpythonic LBYL check is used
        # here because a division by zero inside the residual and
        # jacobian functions defined below seems to be handled
        # internally, being converted to a warning.  Here is it
        # checked and the exception raised to be caught by the
        # dysys.dysys.stepper decorator of dysys.DySys._step.

        if h == 0:
            raise ZeroDivisionError

        def residual(x):
            return ((self.M / h + self.D) * x[0] -
                    self.f(t, x) - self.M / h * xold[0])

        def jacobian(x):
            return (self.M / h + self.D) - self.f1(t, x)

        return (fsolve(residual, xold) if self.f1 is None
                else newton(residual, jacobian, xold, tol))

    def equilibrium(self, x0, tol=1e-3):
        '''solve for a steady-state equilibrium

        using Newton iteration if the Jacobian has been provided in
        the f1 data member (e.g. during initialization), otherwise
        scipy.optimize.fsolve

        The steady-state problem is

            D x = f (inf, x) ,

        so the residual is

            r (x) = D x - f (inf, x)

        with Jacobian

            r' (x) = D - f_x (inf, x) .

        To set up the Newton iteration, for a given x, try to make

            r (x - dx) = 0 ,

        expand in a Taylor series to get

            r (x - dx) = r (x) - r' (x) dx + O (dx**2)

        and solve that to first order for dx, i.e.

            r' (x) dx = r (x) .

        '''

        # This is very much like the step method with the time t and
        # time-step h going to infinity; i.e. the equilibrium is
        # construed as the state eventually reached after a history of
        # forcing which tends asymptotically to a constant value.

        def residual(x):
            return self.D * x[0] - self.f(np.inf, x)  # t -> np.inf

        def jacobian(x):
            return self.D - self.f1(np.inf, x)

        x = x0 if type(x0) is tuple else (x0,)

        return (fsolve(residual, x) if self.f1 is None
                else newton(residual, jacobian, x, tol))

    def constrain(self, *args, **kwargs):
        '''extends the method from the super-class

        Say we have the residual

        r(x) = (M/h+D)x - f(t,x) - (M/h) x0

        and jacobian

        J(x) = (M/h+D) - (df/dx)(t,x)

        and then constrain x = U u + K k so that

        r' = {U.T (M/h+D) U} u - U.T f(t, Uu+Kk) - U.T (M/h) (U u0 + K k)

        (though we expect U.T K = 0, so the last term should drop out)

        r' = {U.T (M/h+D) U} u - U.T f(t, Uu+Kk) - {U.T (M/h) U} u0

           = (M'/h+D') u - df'/dx - (M'/h) u0

        J' = U.T (M/h+D) U - U.T (df/dx) U

           = (M'/h+D') - df'/du

        where M' = U.T * M * U, D' = U.T * D * U, and

        df'/du = U.T * (df/dx) * U

        The addition is that if f is changed to U.T * (f - M * K *
        vknown - D * K * xknown) then its derivative f1 needs to be
        changed to U.T * f1 * U.

        '''

        sys = super(SparseNFDySys, self).constrain(*args, **kwargs)
        sys.f1 = (
            None if self.f1 is None else
            lambda t, x: sys.U.T * self.f1(t, self.reconstitute(x)) * sys.U)
        return sys


def newton(residual, jacobian, x, tol=np.MachAr().eps):

    '''eliminate the residual by Newton-iteration'''

    def iteration(x):
        while True:
            dx = spsolve(jacobian(x), residual(x))
            x = (x[0] - dx,) + x[1:]
            yield x, dx

    # TODO gmcbain 2013-07-29: I think that the fixed-point iteration
    # which follows in the return line might be more easily
    # generalized if the iteration function were not passed the
    # initial data inside the generator expression; instead require
    # the variable 'iteration' to be an iterable

    return next(y for y, h in iteration(x) if np.linalg.norm(h) < tol)
