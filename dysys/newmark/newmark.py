#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''

:author: G. D. McBain <gmcbain>

:created: 2013-02-08

'''

from __future__ import absolute_import, division, print_function

from functools import partial
from warnings import warn

import numpy as np
from scipy.sparse import block_diag, bmat, identity, linalg as sla

from ..cholesky import cholesky
from ..dysys import DySys
from ..fixed_point import solve
from ..linear_dysys import SparseDySys


class Newmark(DySys):
    '''a dynamical system advancing with a Newmark method

    having constant sparse mass, damping, and stiffness matrices and a
    forcing function depending on time

    A Newmark system evolves with the "displacement" as the
    dynamical variable but also has "state" in the form of the
    velocity and acceleration, the former being required as the
    system is of second order while the latter is merely convenient.

    '''

    def __init__(self, M, K, C=None, f=None, beta=0.25, gamma=0.5,
                 definite=False):
        ''':param M: mass scipy.sparse matrix

        :param K: stiffness scipy.sparse

        :param C: damping scipy.sparse matrix, or None, in which case
        it is constructed as like M but with no nonzero entries

        :param f: function of (time, state (typically ignored), dict
        of discrete dynamical variables), returning forcing vector, or
        None in which case a ternary zero-function is substituted

        :param beta: Newmark method parameter, default 0.25 (which,
        with gamma=0.5, is the implicit and unconditionally stable
        "average acceleration" method: Hughes 2000, p. 493)

        :param gamma: Newmark method parameter, default 0.5 (as
        required for second order accuracy: Hughes 2000, Table 9.1.1,
        note 3)

        :param definite: bool, for if system is (positive-)definite

        '''

        self.M, self.K, self.C = M, K, C
        self.f = f or (lambda _, __, ___: np.zeros(len(self)))
        self.beta, self.gamma = beta, gamma
        self.definite = definite

    def __len__(self):
        return self.K.shape[0]

    def equilibrium(self, x=None, d=None, **kwargs):
        '''return the eventual steady-state solution

        using self.f(np.inf, d)

        :param x: initial guess, passed on to self.f, optional

        :param d: dict, passed on to self.f, optional

        Further keyword arguments are passed on to solve

        '''

        # TODO gmcbain 2016-11-01: Adopt DySys.forcing to enable
        # slavish behaviour.
        

        return solve(self.K, self.f(np.inf, x, d), **kwargs)

    def forcing(self, t, h, x, d):

        # TRICKY gmcbain 2016-11-01: Only use value at end of
        # time-step.
        
        return super(Newmark, self).forcing(t, h, x, d)[1]

    def step(self, t, h, x, d):
        'evolve from displacement x at time t to t+h'

        xt = x + h * (self.v + h * (.5 - self.beta) * self.a)
        vt = self.v + (1 - self.gamma) * h * self.a

        rhs = self.forcing(t, h, x, d) - self.K.dot(xt)

        if self.C is not None:
            rhs -= self.C.dot(vt)

        self.a = self.solve(rhs)
        self.v = vt + self.gamma * h * self.a
        return xt + self.beta * h**2 * self.a

    def setA(self, h, alpha=0.):
        '''set the acceleration evolution matrix

        :param h: float > 0, time-step

        :param alpha: optional float, for use by the
        HilberHughesTaylor subclass, defaulting to 0., in which case
        Hilber, Hughes, & Taylor's alpha-method degenerates to Newmark

        '''

        A = self.M + (1 + alpha) * h**2 * self.beta * self.K
        if self.C is not None:
            A += (1 + alpha) * h * self.gamma * self.C
        self.solve = cholesky(A) if self.definite else partial(solve, A)

    def march(self, h, x, d=None, *args, **kwargs):
        '''evolve from displacement x[0] and velocity x[1] with time-step h

        This involves setting the internal variables for velocity and
        acceleration (v and a, respectively), and, for convenience,
        the evolution matrix A, and then deferring to the march method
        of the super-class, DySys.

        '''

        d = {} if d is None else d

        self.v = x[1]
        rhs = self.forcing(0., 0., x[0], d) - self.K.dot(x[0])

        if self.C is not None:
            rhs -= self.C.dot(x[1])

        self.a = solve(self.M, rhs)

        self.setA(h)

        # TRICKY gmcbain 2016-04-08: Return the rate of change of the
        # solution too

        try:
            fnew = kwargs.pop('f')
            kwargs['f'] = lambda x: fnew((x, self.v))
        except KeyError:
            kwargs['f'] = lambda x: (x, self.v)

        return super(Newmark, self).march(h, x[0], d, *args, **kwargs)

    def constrain(self, known, xknown=None, vknown=None, aknown=None):
        '''return a new DySys with constrained degrees of freedom

        having the same class as self.

        :param known: sequence of indices of known degrees of freedom

        :param xknown: corresponding sequence of their values
        (default: zeros)

        :param vknown: corresponding sequence of their rates of change

        :param aknown: corresponding sequence of their second
        derivatives

        '''

        # TODO gmcbain 2016-07-27: Refactor!

        U, Kn = self.node_maps(known)
        project = partial(self.projector, U)

        M, K, C = [None if A is None else project(A * U)
                   for A in [self.M, self.K, self.C]]
        sys = self.__class__(
            M,
            K,
            C,
            lambda *args: project(
                (0 if self.f is None else self.f(*args)) -
                (0 if xknown is None else self.K.dot(Kn.dot(xknown))) -
                (0 if vknown is None else self.C.dot(Kn.dot(vknown))) -
                (0 if aknown is None else self.M.dot(Kn.dot(aknown)))),
            self.beta, self.gamma, self.definite)

        sys.reconstitute = partial(self.reconstituter, U, Kn, xknown)
        sys.project = project

        return sys

    def eigs(self, *args, **kwargs):
        '''return the first few modes of the system'''

        if self.C is None:
            kwargs['M'] = self.M
            if 'sigma' not in kwargs:  # inverse iteration
                kwargs['sigma'] = 0.   # Hughes (2000, §10.5.2)
            try:
                w, v = ((sla.eigsh if self.definite else sla.eigs)
                        (-self.K, *args, **kwargs))
                return np.sqrt(-w), v
            except ValueError:
                warn('system too small, converting to dense', UserWarning)
                for k in ['k', 'M', 'which']:
                    if k in kwargs:
                        del kwargs[k]
                kwargs['right'] = kwargs.pop('return_eigenvectors')
                return self.eig(*args, **kwargs)
        else:

            return self.to_sparse_dysys().eigs(*args, **kwargs)

    def to_sparse_dysys(self):
        '''return an equivalent SparseDySys

        by introducing the rate of change as an auxiliary variable

        '''

        # TODO gmcbain 2016-10-11: Add f etc.

        return SparseDySys(block_diag([identity(self.M.shape[0]),
                                       self.M]),
                           bmat([[None, -identity(self.M.shape[0])],
                                 [self.K, self.C]]))


# Define special cases, as per Hughes (2000, Table 9.1.1, p. 493)

trapezoidal = partial(Newmark, beta=.25, gamma=.5)

linear_acceleration = partial(Newmark, beta=1/6, gamma=.5)

fox_goodwin = partial(Newmark, beta=1/12, gamma=.5)

central_difference = partial(Newmark, beta=0, gamma=.5)
