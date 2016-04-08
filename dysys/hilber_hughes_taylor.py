#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''This implements the Hilber-Hughes-Taylor (a.k.a. alpha-) method

(Hilber 1976; Hilber, Hughes, & Taylor 1977; Cook, Malkus, & Plesha
1989, p. 409; Hughes 2000, p. 532; Craveur 2008, §18.4.5), as a
generalization of the Newmark method

:author: G. D. McBain <gmcbain>

:created: 2016-04-06

References
----------

* Cook, R.D., Malkus, D.S., & Plesha, M.E. (1989). Concepts and
  Applications of Finite Elements Analysis (3rd ed.). New York: Wiley
  & Sons

* Craveur, J.-C. (2008). Modélisation par éléments finis (Third
  ed.). Paris: Dunod

* Hilber, H. M. (1976). Analysis and design of numerical integration
  methods in structural dynamics. Technical Report UCB/EERC-76/29,
  Earthquake Engineering Research Center, University of California
  Berkeley (unsighted, cited by Hughes 2000, p. 532)

* Hilber, H. M., T. J. R. Hughes, & R. L. Taylor (1977). Improved
  numerical dissipation for time integration algorithms in structural
  dynamics. Earthquake Engineering & Structural Dynamics 5 (3),
  283-292 (unsighted, cited by Cook, Malkus, & Plesha 1989, p. 409;
  Hughes 2000, p. 532)

* Hughes, T. J. R. (2000). The Finite Element Method. Mineola, New
  York: Dover

'''

from __future__ import absolute_import, division, print_function

from .dysys import DySys
from .fixed_point import solve


class HilberHughesTaylor(DySys):
    '''a dynamical system advancing with a Hilber-Hughes-Taylor method

    having constant sparse mass, damping, and stiffness matrices and a
    forcing function depending on time

    '''

    # A Hilber-Hughes-Taylor system evolves with the "displacement" as
    # the dynamical variable but also has "state" in the form of the
    # velocity and acceleration, the former being required as the
    # system is of second order while the latter is merely convenient.

    def __init__(self, M, C, K, f, alpha=0.):
        ''':param M: mass scipy.sparse matrix

        :param C: damping scipy.sparse matrix

        :param K: stiffness scipy.sparse

        :param f: function of time returning forcing vector

        :param alpha: float, Hilber-Hughes-Taylor method parameter,
        should be in [-1/3, 0]


        '''

        self.M, self.C, self.K = M, C, K
        self.f = f
        self.alpha = alpha

        self.beta = (1 - alpha)**2 / 4.
        self.gamma = (1 - 2 * alpha) / 2.

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

    def march(self, h, x, d=None, *args, **kwargs):
        '''evolve from displacement x[0] and velocity x[1] with time-step h

        This involves setting the internal variables for velocity and
        acceleration (v and a, respectively), and, for convenience,
        the evolution matrix A, and then deferring to the march method
        of the super-class, DySys.

        '''

        d = {} if d is None else d

        self.v = x[1]
        self.a = solve(self.M,
                       self.f(0., d) - self.C.dot(x[1]) - self.K.dot(x[0]))

        self.A = self.M + (1 + self.alpha) * h * (self.gamma * self.C +
                                                  self.beta * h * self.K)

        if 'f' not in kwargs:
            # TRICKY gmcbain 2016-04-08: Return the rate of change of
            # the solution too
            kwargs['f'] = lambda x: (x, self.v)
            
        return super(self.__class__, self).march(h, x[0], d, *args, **kwargs)
