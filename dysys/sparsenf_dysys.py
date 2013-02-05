#!/usr/bin/env python

'''
:author: G. D. McBain <gmcbain>
:created: 2013-01-23

'''

import numpy as np

from scipy.optimize import fsolve, newton_krylov
from scipy.sparse.linalg import spsolve

from dysys import DySys

class SparseNFDySys(DySys):

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

        If the Jacobian derivative of f with respect to x is provided,
        it will be used by self.jacobian to compute the jacobian of
        the evolution matrix and a linear operator carrying out its
        inverse passed as the preconditioner to newton_krylov for the
        time-stepping.

        '''

        self.M, self.D, self.f, self.f1 = M, D, f, f1

    def residual(self, h, xold, x, t):
        return (self.M / h + self.D) * x - self.f(t, x) - self.M / h * xold

    def jacobian(self, t, x, h):
        return (self.M / h + self.D) - self.f1(t, x)

    def step(self, t, xold, h, tol=1e-3):
        if self.f1 is None:
            return fsolve(lambda y: self.residual(h, xold, y, t), xold)
        else:
            # return newton_krylov(
            #     lambda y: self.residual(h, x, y, t), x, 
            #     inner_M=LinearOperator(
            #         [len(x)]*2,
            #         lambda y: spsolve(self.jacobian(t, x, h), y)))
            x = np.copy(xold)
            while True:         # Newton iteration
                dx = spsolve(self.jacobian(t, x, h),
                             self.residual(h, xold, x, t))
                x -= dx
                if np.linalg.norm(dx) < tol:
                    break
            return x
