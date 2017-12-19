#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''This implements the Hilber-Hughes-Taylor (a.k.a. alpha-) method

(Hilber 1976; Hilber, Hughes, & Taylor 1977; Cook, Malkus, & Plesha
1989, p. 409; Hughes 2000, p. 532; Craveur 2008, §18.4.5), as a
generalization of the Newmark method

:author: G. D. McBain <gmcbain>

:created: 2016-04-21

'''

from __future__ import absolute_import, division, print_function

from .dysys import DySys
from .fixed_point import solve


class HilberHughesTaylor_d(DySys):
    '''a dynamical system advancing with the alpha-method in d-form

    See also: HilberHughesTaylor

    '''

    def __init__(self, M, C, K, f, alpha=0.):

        ''':param M: mass scipy.sparse matrix

        :param C: damping scipy.sparse matrix

        :param K: stiffness scipy.sparse

        :param f: function of time and dict of discrete dynamical
        variables, returning forcing vector

        :param alpha: float, Hilber-Hughes-Taylor method parameter,
        should be in [-1/3, 0]

        '''

        self.M, self.C, self.K = M, C, K
        self.f = f
        self.alpha = alpha
        self.beta = (1 - alpha)**2 / 4.
        self.gamma = (1 - 2 * alpha) / 2.

    def setKeff(self, h):
        self.Keff = (self.M / self.beta / h**2 +
                     (1 + self.alpha) * (self.gamma * self.C / self.beta / h +
                                         self.K))

    def step(self, t, h, x, d):
        'evolve from displacement x at time t to t+h'

        x1 = solve(self.Keff,
                   (1 + self.alpha) * self.f(t + h, d) -
                   self.alpha * self.f(t, d) +
                   self.M @ (x / self.beta / h**2 +
                             self.v / self.beta / h +
                             (1 - 2 * self.beta) / 2. / self.beta * self.a) +
                   self.C @ ((1 + self.alpha) * self.gamma /
                             self.beta / h * x +
                             ((1 + self.alpha) * self.gamma / self.beta - 1) *
                             self.v -
                             (1 + self.alpha) *
                             (1 - self.gamma / 2. / self.beta) * h * self.a) +
                   self.K @ (self.alpha * x))

        self.v, self.a = ((self.gamma / self.beta / h * (x1 - x) +
                           (1 - self.gamma / self.beta) * self.v +
                           h * (1 - .5 * self.gamma / self.beta) * self.a),
                          ((x1 - x - h * self.v) / self.beta / h**2 -
                           (.5 / self.beta - 1) * self.a))

        return x1

    def march(self, h, x, d=None, *args, **kwargs):

        d = {} if d is None else d

        self.v = x[1]

        self.a = solve(self.M,
                       self.f(0., d) - self.C @ x[1] - self.K @ x[0])

        self.setKeff(h)

        if 'f' not in kwargs:
            # TRICKY gmcbain 2016-04-08: Return the rate of change of
            # the solution too
            kwargs['f'] = lambda x: (x, self.v)

        return super(self.__class__, self).march(h, x[0], d, *args, **kwargs)
