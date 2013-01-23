#!/usr/bin/env python

'''
:author: G. D. McBain <gmcbain>
:created: 2013-01-23

'''


from scipy.optimize import fsolve, newton_krylov
from scipy.sparse.linalg import LinearOperator, spsolve

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

    def residual(self, h, x, y, t):
        return (self.M / h + self.D) * y - self.f(t, y) - self.M / h * x

    def jacobian(self, h, x, t):
        return (self.M / h + self.D) - self.f1(t, x)

    def step(self, t, x, h):
        if self.f1 is None:
            Jinv = None
        else:
            Jinv = LinearOperator([len(x)]*2,
                                  lambda y: spsolve(self.jacobian(h, x, t), y))

        return newton_krylov(lambda y: self.residual(h, x, y, t), x, 
                             inner_M=Jinv)
