from functools import partial

import numpy as np

from scipy.optimize import root

from .linear_dysys import LinearDySys
from ..fixed_point import newton


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
        super().__init__(M, D, f)

    def step(self, t, h, xold, d, tol=1e-3):

        # KLUDGE gmcbain 2013-04-08: An unpythonic LBYL check is used
        # here because a division by zero inside the residual and
        # jacobian functions defined below seems to be handled
        # internally, being converted to a warning.  Here is it
        # checked and the exception raised to be caught by the
        # dysys.dysys.stepper decorator of dysys.DySys._step.

        if h == 0:
            raise ZeroDivisionError

        def residual(x):
            return ((self.M / h + self.D) @ x -
                    self.f(t, x, d) - self.M @ xold / h)

        def jacobian(x):
            return (self.M / h + self.D) - self.f1(t, x, d)

        return (root(residual, xold).x if self.f1 is None
                else newton(residual, jacobian, xold, tol))

    def equilibrium(self, x0, d=None, **kwargs):
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

        kwargs.setdefault('tol', 1e-3)

        def residual(x):
            return self.D @ x - self.f(np.inf, x, d)  # t -> np.inf

        def jacobian(x):
            return self.D - self.f1(np.inf, x, d)

        return (root(residual, x0, **kwargs).x if self.f1 is None
                else newton(residual, jacobian, x0, **kwargs))

    def constrain(self, known, xknown=None, vknown=None):
        '''return a new SparseNFDySys with constrained degrees of freedom

        :param known: sequence of indices of known degrees of freedom

        :param xknown: corresponding sequence of their values
        (default: zeros)

        :param vknown: corresponding sequence of their rates of change

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

        U, K = self.node_maps(known)
        project = partial(self.projector, U)

        M, D = [None if A is None else project(A * U)
                for A in [self.M, self.D]]
        sys = self.__class__(
            M,
            D,
            lambda *args: project(
                (0 if self.f is None else self.f(*args)) -
                (0 if xknown is None else self.D @ K @ xknown) -
                (0 if vknown is None else self.M @ K @ vknown)),
            None if self.f1 is None else
            (lambda t, x, d: U.T @ (self.f1(t, sys.reconstitute(x), d) @ U)))

        sys.reconstitute = partial(self.reconstituter, U, K, xknown)
        sys.project = project

        return sys
