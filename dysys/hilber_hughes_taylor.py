#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''This implements the Hilber-Hughes-Taylor (a.k.a. alpha-) method

(Hilber 1976; Hilber, Hughes, & Taylor 1977; Cook, Malkus, & Plesha
1989, p. 409; Hughes 2000, p. 532; Craveur 2008, §18.4.5), as a
generalization of the Newmark method

:author: G. D. McBain <gmcbain>

:created: 2016-04-06

'''

from __future__ import absolute_import, division, print_function

from .newmark import Newmark
from .fixed_point import solve


class HilberHughesTaylor(Newmark):
    '''a dynamical system advancing with a Hilber-Hughes-Taylor method

    (a.k.a. alpha-method), like the Newmark method
    
    '''

    def __init__(self, M, C, K, f, alpha=0.):
        ''':param M: mass scipy.sparse matrix

        :param C: damping scipy.sparse matrix

        :param K: stiffness scipy.sparse

        :param f: function of time returning forcing vector

        :param alpha: float, Hilber-Hughes-Taylor method parameter,
        should be in [-1/3, 0]


        '''

        super(self.__class__, self).__init__(M, C, K, f,
                                             (1 - alpha)**2 / 4.,
                                             (1 - 2 * alpha) / 2.)
        self.alpha = alpha

    def step(self, t, h, x, d):
        'evolve from displacement x at time t to t+h'

        xt = x + h * (self.v + h * (.5 - self.beta) * self.a)
        vt = self.v + (1 - self.gamma) * h * self.a

        self.a = solve(self.A,
                       (1 + self.alpha) * self.f(t + h, d) -
                       self.alpha * self.f(t, d) -
                       self.C.dot((1 + self.alpha) * vt -
                                  self.alpha * self.v) -
                       self.K.dot((1 + self.alpha) * xt -
                                  self.alpha * x))
        self.v = vt + self.gamma * h * self.a
        return xt + self.beta * h**2 * self.a

    def setA(self, h):
        self.A = self.M + (1 + self.alpha) * h * (self.gamma * self.C +
                                                  self.beta * h * self.K)
