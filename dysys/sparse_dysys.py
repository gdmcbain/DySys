#!/usr/bin/env python

'''
:author: G. D. McBain <gmcbain>
:created: 2013-01-11

'''

from scipy.sparse.linalg import spsolve

from linear_dysys import LinearDySys
from dysys import stepper

class SparseDySys(LinearDySys):
    '''a LinearDySys using sparse matrices and backward Euler

    The matrices might be individually singular, but the pencil should
    be regular (see Yip & Sincovec 1981); i.e. M/h+D should be
    invertible for positive time-step h.
    
    '''

    @stepper
    def step(self, t, x, h):
        '''estimate the next state using backward Euler'''
        return spsolve(self.M / h + self.D, 
                           self.f(t) + self.M / h * x) 

if __name__ == '__main__':

    import itertools as it

    import numpy as np
    from scipy.sparse import eye

    class Decay(SparseDySys):
    
        "tau x' + x = 0, which decays exponentially with timescale tau."

        def __init__(self, tau=0.7):
            self.tau = tau
            D = eye(1, 1)
            super(Decay, self).__init__(tau * D, D, lambda t: np.zeros(1))

        def exact(self, t, ic):
            return ic * np.exp(-t / self.tau)

    system = Decay()
    ic = 1.0

    history = system.march(np.array([ic]), 0.1)

    for t, x in it.takewhile(lambda event: event[1] > ic/9, history):
        print '{0:.1f}  {1:.3f}  {2:.3f}'.format(t, x[0], system.exact(t, ic))
