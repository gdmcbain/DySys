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

        # KLUDGE gmcbain 2013-04-08: An unpythonic LBYL check is used
        # here because a division by zero inside the residual and
        # jacobian functions defined below seems to be handled
        # internally, being converted to a warning.  Here is it
        # checked and the exception raised to be caught by the
        # dysys.dysys.stepper decorator of dysys.DySys._step.

        if h == 0:
            raise ZeroDivisionError

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
            return self.D * x - self.f(np.inf, x) # t -> np.inf

        def jacobian(x):
            return self.D - self.f1(np.inf, x)
            
        if self.f1 is None:
            x = fsolve(residual, x0)
        else:
            x = x0
            while True:
                dx = spsolve(jacobian(x), residual(x))
                x -= dx
                if np.linalg.norm(dx) < tol:
                    break
        return x

