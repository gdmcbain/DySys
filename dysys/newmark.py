#!/usr/bin/env python

'''

:author: G. D. McBain <gmcbain>

:created: 2013-02-08

'''

from __future__ import absolute_import, division, print_function

from functools import partial

from .dysys import DySys
from .fixed_point import solve


class Newmark(DySys):
    '''a dynamical system advancing with a Newmark method

    having constant sparse mass, damping, and stiffness matrices and a
    forcing function depending on time

    A Newmark system evolves with the "displacement" as the
    dynamical variable but also has "state" in the form of the
    velocity and acceleration, the former being required as the
    system is of second order while the latter is merely convenient.

    '''

    def __init__(self, M, C, K, f, beta=0.25, gamma=0.5):
        ''':param M: mass scipy.sparse matrix

        :param C: damping scipy.sparse matrix

        :param K: stiffness scipy.sparse

        :param f: function of time and dict of discrete dynamical
        variables, returning forcing vector

        :param beta: Newmark method parameter, default 0.25 (which,
        with gamma=0.5, is the implicit and unconditionally stable
        "average acceleration" method: Hughes 2000, p. 493)

        :param gamma: Newmark method parameter, default 0.5 (as
        required for second order accuracy: Hughes 2000, Table 9.1.1,
        note 3)

        '''

        self.M, self.C, self.K = M, C, K
        self.f = f
        self.beta, self.gamma = beta, gamma

    def step(self, t, h, x, d):
        'evolve from displacement x at time t to t+h'

        xt = x + h * (self.v + h * (.5 - self.beta) * self.a)
        vt = self.v + (1 - self.gamma) * h * self.a

        self.a = solve(self.A,
                       self.f(t + h, d) - self.C.dot(vt) - self.K.dot(xt))
        self.v = vt + self.gamma * h * self.a
        return xt + self.beta * h**2 * self.a

    def setA(self, h):
        self.A = self.M + h * (self.gamma * self.C + self.beta * h * self.K)

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

        self.setA(h)

        if 'f' not in kwargs:
            # TRICKY gmcbain 2016-04-08: Return the rate of change of
            # the solution too
            kwargs['f'] = lambda x: (x, self.v)

        return super(Newmark, self).march(h, x[0], d, *args, **kwargs)

### Define special cases, as per Hughes (2000, Table 9.1.1, p. 493)

trapezoidal = partial(Newmark, beta=.25, gamma=.5)

linear_acceleration = partial(Newmark, beta=1/6, gamma=.5)

fox_goodwin = partial(Newmark, beta=1/12, gamma=.5)

central_difference = partial(Newmark, beta=0, gamma=.5)

    # TODO gmcbain 2013-05-16: Provide modal analysis methods eig and
    # eigs (like those of SparseDySys).  One way to formulate this is
    # to convert the second-order system to a first-order block system
    # by introducing an auxiliary variable for the temperature, then
    # the eigenvalue problem is not quadratic but linear, the standard
    # form of the generalized algebraic eigenvalue problem accepted by
    # scipy.linalg.eig and scipy.sparse.linalg.eigs.
