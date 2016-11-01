#!/usr/bin/env python

'''
:author: G. D. McBain <gmcbain>
:created: 2013-01-11

'''

from __future__ import absolute_import, division, print_function

from warnings import warn

import numpy as np

from scipy.linalg import eig
from scipy.sparse import identity, linalg as sla

from ...cholesky import cholesky
from ...fixed_point import solve
from ..linear_dysys import LinearDySys


class SparseDySys(LinearDySys):
    '''a LinearDySys using sparse matrices and backward Euler

    The matrices might be individually singular, but the pencil should
    be regular (see Yip & Sincovec 1981); i.e. M/h+D should be
    invertible for positive time-step h.

    '''

    def step(self, t, h, x, d):
        '''estimate the next state using theta method

        memoizing the incomplete-LU factors of the evolution matrix
        for fast time-stepping

        '''

        if not hasattr(self, '_memo') or h != self._memo['h']:
            if h == 0.:
                raise ZeroDivisionError

            M = self.M / h - (1 - self.theta) * self.D

            self._memo = {'h': h, 'M': M}

            M1 = M + self.D
            if self.definite:
                self._memo['solve'] = cholesky(M1)
            else:

                def solver(rhs):
                    x1, info = sla.lgmres(
                        M1, rhs, x0=x,
                        M=sla.LinearOperator(M.shape, sla.spilu(M1).solve))
                    if info == 0:
                        return x1
                    else:
                        if info > 0:
                            raise RuntimeError(
                                'convergence to tolerance not achieved '
                                'in %s iterations' % info)
                        else:
                            raise ValueError('info %d' % info)

                self._memo['solve'] = solver

        return self._memo['solve'](
            self._memo['M'].dot(x) +
            np.array([1 - self.theta, self.theta]).dot(
                self.forcing(t, h, x, d)))

    def equilibrium(self, x=None, d=None, **kwargs):
        '''return the eventual steady-state solution

        :param x: initial condition, optional, passed on to self.f

        :param d: dict, discrete dynamical variables, optional, passed
        on to self.f

        Further keyword arguments passed on to solve.

        '''

        # TODO gmcbain 2016-11-01: Adopt DySys.forcing to enable
        # slavish behaviour.

        return solve(self.D, self.f(np.inf, x, d), **kwargs)

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
            return sla.eigs(-self.D.tocsc(), *args, **kwargs)
        except ValueError:
            warn('system too small, converting to dense', UserWarning)
            for k in ['k', 'M', 'which']:
                if k in kwargs:
                    del kwargs[k]
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
    print('Spectrum: {0} (exact: {1})'.format(
          np.real_if_close(system.eig(right=False)),
          -1 / system.tau))
