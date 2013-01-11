'''
:author: G. D. McBain <gmcbain>
:created: 2013-01-11

'''

from scipy.sparse.linalg import spsolve

from linear_dysys import LinearDySys

class SparseDySys(LinearDySys):
    '''a LinearDySys using sparse matrices and backward Euler

    The matrices might be individually singular, but the pencil should
    be regular (see Yip & Sincovec 1981); i.e. M/h+D should be
    invertible for positive time-step h.
    
    '''

    def step(self, t, x, h):
        '''estimate the next state using backward Euler'''
        return spsolve(self.M / h + self.D, 
                       self.f(t) + self.M / h * x) 

