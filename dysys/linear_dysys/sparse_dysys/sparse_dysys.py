#!/usr/bin/env python

'''
:author: G. D. McBain <gmcbain>
:created: 2013-01-11

'''

from __future__ import absolute_import, division, print_function

from functools import partial
from warnings import warn

import numpy as np

from scipy.interpolate import interp1d
from scipy.linalg import eig
from scipy.sparse import linalg as sla

from toolz import dissoc, keymap, merge

from ...cholesky import cholesky
from ...fixed_point import solve
from ..linear_dysys import LinearDySys


class SparseDySys(LinearDySys):
    '''a LinearDySys using sparse matrices and backward Euler

    The matrices might be individually singular, but the pencil should
    be regular (see Yip & Sincovec 1981); i.e. M/h+D should be
    invertible for positive time-step h.

    '''

    def __len__(self):
        return self.D.shape[0]

    def step(self, t, h, x, d=None, inputs=None):
        '''estimate the next state using theta method

        :param t: float, time

        :param h: float > 0, time-step

        :param x: numpy.ndarray, state, initial condition

        :param d: dict, passed to self.forcing

        :param inputs: optional pair, being inputs at t and t + h
        [default: None]

        Attempt fast time-stepping, reusing factors if the time-step
        is the same as on the previous call.  Use Cholesky if
        self.definite; otherwise incomplete-LU and LGMRES.

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

                # def solver(rhs):
                #     x1, info = sla.lgmres(
                #         M1, rhs, x0=x, tol=1e-12,
                #         M=sla.LinearOperator(M.shape, sla.spilu(M1).solve))
                #     if info == 0:
                #         return x1
                #     else:
                #         if info > 0:
                #             warn('convergence to tolerance not achieved '
                #                  'in %s iterations' % info, RuntimeWarning)
                #             return solve(M1, rhs)                        
                #         else:
                #             raise ValueError('info %d' % info)

                # self._memo['solve'] = solver
                self._memo['solve'] = partial(solve, M1)

        return self._memo['solve'](
            self._memo['M'].dot(x) +
            interp1d([0, 1],
                     np.vstack(self.forcing(t, h, x, d, inputs)).T)(self.theta))

    def equilibrium(self, x=None, d=None, *args, **kwargs):
        '''return the eventual steady-state solution

        :param x: initial condition, optional, passed on to self.f

        :param d: dict, discrete dynamical variables, optional, passed
        on to self.forcing

        Further positional arguments are also passed on to
        self.forcing.

        Further keyword arguments passed on to solve.

        '''

        return solve(self.D, self.forcing(0, np.inf, x, d, *args)[1],
                     **kwargs)

    def eig(self, *args, **kwargs):
        '''return the complete spectrum of the system

        Any positional and keyword arguments are passed on to
        scipy.linalg.eig.

        '''

        return eig(-self.D.todense(), self.M.todense(),
                   *args, **(dissoc(kwargs, 'sigma')))

    def eigs(self, *args, **kwargs) -> np.ndarray:
        '''return the first few modes of the system,

        being the modes with temporal eigenvalues of least magnitude.
        This is achieved using :function scipy.sparse.linalg.eigs: by
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

        :rtype: as per scipy.sparse.linalg.eigs or self.eig, if self
        is too small to be treated by the former

        '''

        try:
            return sla.eigs(-self.D.tocsc(), *args,
                            **merge({'M': self.M, 'sigma': 0.}, kwargs))
        except ValueError:
            warn('system too small, converting to dense', UserWarning)
            return self.eig(
                *args, **keymap(
                    lambda k: 'right' if k == 'return_eigenvectors' else k,
                    dissoc(kwargs, 'k', 'M', 'which')))


def demo():
    # see msmdir.003774 for an archived run

    import pandas as pd
    from scipy.sparse import identity

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
