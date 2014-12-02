#!/usr/bin/env python

'''
:author: G. D. McBain <gmcbain>
:created: 2013-01-11

'''

from __future__ import absolute_import, division, print_function
from warnings import warn

import numpy as np

from scipy.linalg import eig
from scipy.sparse import identity
from scipy.sparse.linalg import eigs

from .fixed_point import solve
from .linear_dysys import LinearDySys


class SparseDySys(LinearDySys):
    '''a LinearDySys using sparse matrices and backward Euler

    The matrices might be individually singular, but the pencil should
    be regular (see Yip & Sincovec 1981); i.e. M/h+D should be
    invertible for positive time-step h.

    '''

    def step(self, t, h, x, d):
        '''estimate the next state using backward Euler'''
        # TRICKY gmcbain 2013-06-28: A very nasty workaround is
        # required here to accommodate changes to
        # scipy.sparse.linalg.spsolve between 0.10.1 and 0.12.0, for
        # handling trivial 1x1 systems which fall foul of being
        # squeezed, since then the have a shape which is an empty
        # tuple and that can't be indexed!
        b = (self.M.dot(x) / h +
             (0 if self.f is None else self.f(t, d)))

        # TODO gmcbain 2014-05-08: factor out this wrapping of
        # spsolve, perhaps in fixed_point?

        # try:
        #     return spsolve(self.M / h + self.D, b)
        # except IndexError:              # singleton system?
        #     return b / (self.M / h + self.D)[0, 0]
        
        return solve(self.M / h + self.D, b)

    def equilibrium(self, d=None):
        '''return the eventual steady-state solution'''
        b = np.zeros(len(self)) if self.f is None else self.f(np.inf, d)
        try:
            return solve(self.D, b)
        except IndexError:      # singleton system?
            return np.array(b / (self.D)[0, 0])

    def harmonic(self, omega):
        '''return the complex harmonic solution

        :param: omega, float (typically positive)

        # TODO gmcbain 20140508: let omega be a sequence

        '''

        # M x' + D x - f (t) = 0, with f(t) = F exp (j w t), X = s exp (j w t)

        # (D + j w M) X - F = 0

        return [sys.equilibrium() for sys in
                (self.__class__(None, self.D + 1j * w * self.M, self.f)
                 for w in omega)]

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

        if 'return_eigenvectors' not in kwargs:
            kwargs['return_eigenvectors'] = False
        kwargs['M'] = self.M

        try:
            return eigs(-self.D.tocsc(), *args, **kwargs)
        except ValueError as too_small:
            warn('system too small, converting to dense', UserWarning)
            if 'k' in kwargs:
                del kwargs['k']
            del kwargs['M']
            kwargs['right'] = kwargs.pop('return_eigenvectors')
            return self.eig(*args, **kwargs)


def demo():
    # see msmdir.003774 for an archived run

    import pandas as pd

    class Decay(SparseDySys):

        "tau x' + x = 0, which decays exponentially with timescale tau."

        def __init__(self, tau=0.7):
            self.tau = tau
            D = identity(1, format='csr')
            super(Decay, self).__init__(tau * D, D)

        def exact(self, t, ic):
            return ic * np.exp(-t / self.tau)

    system = Decay()
    ic = 1.0

    history = pd.Series(
        dict((t, s[0][0]) for (t, s) in
             system.march_while(lambda state: state[0][0] > ic / 9,
                                np.array([ic]),
                                0.1)))

    history = pd.DataFrame({'DySys': history,
                            'exact': system.exact(np.array(history.index,
                                                           dtype=float), ic)})
    print(history)

    print('Equilibrium: ', system.equilibrium())
    Print('Spectrum: {0} (exact: {1})'.format(
          np.real_if_close(system.eig(right=False)),
          -1 / system.tau))
