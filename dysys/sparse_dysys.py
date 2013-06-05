#!/usr/bin/env python

'''
:author: G. D. McBain <gmcbain>
:created: 2013-01-11

'''

from warnings import warn

import numpy as np
from scipy.linalg import eig
from scipy.sparse import eye
from scipy.sparse.linalg import (spsolve, eigs)

from linear_dysys import LinearDySys

class SparseDySys(LinearDySys):
    '''a LinearDySys using sparse matrices and backward Euler

    The matrices might be individually singular, but the pencil should
    be regular (see Yip & Sincovec 1981); i.e. M/h+D should be
    invertible for positive time-step h.
    
    '''

    def step(self, t, x, h):
        '''estimate the next state using backward Euler'''
        return (spsolve(self.M / h + self.D,
                        self.M / h * x[0] +
                        (0 if self.f is None else self.f(t))),) + x[1:]

    def equilibrium(self):
        '''return the eventual steady-state solution'''
        return (spsolve(self.D, 
                        np.zeros(len(self)) 
                        if self.f is None else self.f(np.inf)),)

    def eig(self, *args, **kwargs):
        '''return the complete spectrum of the system

        Any positional and keyword arguments are passed on to
        scipy.linalg.eig.

        '''

        return eig(-self.D.todense(), self.M.todense(), *args, **kwargs)

    def eigs(self, *args, **kwargs):
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

        # TODO gmcbain 2013-05-16: Think of a general way to enable
        # actually optionally returning eigenvectors.

        kw = {'M': self.M, 
              'sigma': 0, 
              'which': 'LM',
              'return_eigenvectors': False}
        try:
            kwargs.update(kw)
            return eigs(-self.D.tocsc(), *args, **kwargs)
        except ValueError as too_small:
            warn('system too small, converting to dense', UserWarning)
            for key in kw.keys():
                del kwargs[key]
            return self.eig(*args, **kwargs)

def main():
    # see msmdir.003744 for an archived run
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

    history = system.march_while(lambda state: state[0] > ic / 9, 
                                 np.array([ic]), 
                                 0.1,
                                 pandas=True)

    history.columns = ['DySys']
    history['exact'] = system.exact(np.array(history.index, dtype=float), ic)
    print history

    print 'Equilibrium: ', system.equilibrium()
    print 'Spectrum: {0} (exact: {1})'.format(
        np.real_if_close(system.eig(right=False)), 
        -1/system.tau)

if __name__ == '__main__':
    main()
