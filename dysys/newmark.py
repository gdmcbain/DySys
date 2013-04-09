#!/usr/bin/env python

'''
:author: G. D. McBain <gmcbain>
:created: 2013-02-08

'''

from scipy.sparse.linalg import spsolve

from dysys import DySys

class Newmark(DySys):
    '''a dynamical system advancing with a Newmark method

    having constant sparse mass, damping, and stiffness matrices and a
    forcing function depending on time

    Reference: Hughes, T. J. R. (2000). The Finite Element
    Method. Mineola, New York: Dover, ch. 9 "Algorithms for Hyperbolic
    and Parabolic-Hyperbolic Problems"

    '''

    # A Newmark system evolves with the "displacement" as the
    # dynamical variable but also has "state" in the form of the
    # velocity and acceleration, the former being required as the
    # system is of second order while the latter is merely convenient.

    def __init__ (self, M, C, K, f, beta=0.25, gamma=0.5):
        ''':param M: mass scipy.sparse matrix

        :param C: damping scipy.sparse matrix

        :param K: stiffness scipy.sparse

        :param f: function of time returning forcing vector

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

    def step(self, t, d, h):
        'evolve from displacement d at time t to t+h'

        d += h * self.v + h*h*(1-2*self.beta)*self.a / 2
        self.v += (1 - self.gamma) * h * self.a
        self.a = spsolve(self.A,
                         self.f(t) - self.C * self.v - self.K * d)
        self.v += self.gamma * h * self.a
        return d + self.beta * h * h * self.a

    def march(self, d, v, h):
        '''evolve from displacement d and velocity v with time-step h

        This involves setting the internal variables for velocity and
        acceleration (v and a, respectively), and, for convenience,
        the evolution matrix A, and then deferring to the march method
        of the super-class, DySys.

        '''

        self.v = v
        self.a = spsolve(self.M, self.f(0.) - self.C * v - self.K * d)

        self.A = self.M + self.gamma * h * self.C + self.beta * h * h * self.K
        
        return super(Newmark, self).march(d, h)

