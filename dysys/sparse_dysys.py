#!/usr/bin/env python

'''
:author: G. D. McBain <gmcbain>
:created: 2013-01-11

'''

from warnings import warn

import numpy as np
from scipy.linalg import eig
from scipy.sparse.linalg import (spsolve, eigs)

from linear_dysys import LinearDySys
from dysys import stepper

class SparseDySys(LinearDySys):
    '''a LinearDySys using sparse matrices and backward Euler

    The matrices might be individually singular, but the pencil should
    be regular (see Yip & Sincovec 1981); i.e. M/h+D should be
    invertible for positive time-step h.
    
    '''

    def step(self, t, x, h):
        '''estimate the next state using backward Euler'''
        return spsolve(self.M / h + self.D, 
                       (0 if self.f is None else self.f(t)) + self.M / h * x) 

    def equilibrium(self):
        '''return the eventual steady-state solution'''
        return spsolve(self.D, 
                       np.zeros(self.D.shape[0]) 
                       if self.f is None else self.f(np.inf))

    def spectrum(self):
        'return the complete spectrum of the system'
        return eig(-self.D.todense(), self.M.todense(), right=False)

    def modes(self, *args, **kwargs):
        '''return the first few modes of the system,

        being the modes with temporal eigenvalues of least magnitude.
        This is achieved using scipy.sparse.linalg.eigs by
        shift-inverting on sigma=0 and then seeking the eigenvalues of
        largest magnitude, since apparently ARPACK is better at
        seeking large eigenvalues than small.

        The returned quantities are the complex amplification factors,
        in units of inverse-time; they occur in complex conjugate
        pairs and their negative parts are twice pi times the
        frequency.  The real parts should be negative (if the system
        is stable) and represent the reciprocal of decay time
        constants.

        For very small systems, delegate to :method spectrum: and
        compute all the modes of the discrete system (which involves
        converting to dense form).

        All positional and keyword arguments are passed to eigs, in
        particular k:
        
        :param k: number of modes to return (counting complex
        conjugate pairs together), defaulting to 6 (the current
        default of scipy.sparse.linalg.eigs)

        :rtype: pair, being np.array of (generally complex)
        eigenvalues

        
        '''

        try:
            kwargs.update({'M': self.M, 'sigma': 0, 'which': 'LM',
                           'return_eigenvectors': False})
            return eigs(-self.D.tocsc(), *args, **kwargs)
        except ValueError as too_small:
            warn('system too small, converting to dense', UserWarning)
            return self.spectrum()


if __name__ == '__main__':

    import itertools as it

    import numpy as np
    from scipy.sparse import eye

    class Decay(SparseDySys):
    
        "tau x' + x = 0, which decays exponentially with timescale tau."

        def __init__(self, tau=0.7):
            self.tau = tau
            D = eye(1, 1)
            super(Decay, self).__init__(tau * D, D)

        def exact(self, t, ic):
            return ic * np.exp(-t / self.tau)

    system = Decay()
    ic = 1.0

    t, x = system.march_while(lambda state: state > ic / 9, np.array([ic]), 0.1)

    print np.array((t, x, system.exact(np.array(t), ic))).T

    print 'Equilibrium: ', system.equilibrium()
    print 'Spectrum: {0} (exact: {1})'.format(system.spectrum(), -1/system.tau)
