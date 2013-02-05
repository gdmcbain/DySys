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

    def step(self, t, xold, h, tol=1e-3):

        def residual(x):
            return (self.M / h + self.D) * x - self.f(t, x) - self.M / h * xold

        def jacobian(x):
            return (self.M / h + self.D) - self.f1(t, x)

        if self.f1 is None:
            x = fsolve(lambda x: residual(x), xold)
        else:
            x = np.copy(xold)
            while True:         # Newton iteration
                dx = spsolve(jacobian(x), residual(x))
                x -= dx
                if np.linalg.norm(dx) < tol:
                    break
        return x

    def equilibrium(sys, x0, tol=1e-3):
        '''solve for a steady-state equilibrium

        using Newton iteration if the Jacobian has been provided in
        the f1 data member (e.g. during initialization), otherwise
        scipy.optimize.fsolve

        '''
        
        def residual(x):
            return sys.D * x - sys.f(np.inf, x) # t -> np.inf

        def jacobian(x):
            return sys.D - sys.f1(np.inf, x)
            
        if sys.f1 is None:
            x = fsolve(residual, x0)
        else:
            x = x0
            while True:
                dx = spsolve(jacobian(x), residual(x))
                x -= dx
                if np.linalg.norm(dx) < tol:
                    break
        return x

