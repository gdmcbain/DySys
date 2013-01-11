#!/usr/bin/python

'''
A demonstration of using the backward Euler method of marching a
sparse-linear differential-algebraic system in time.

:author: G. D. McBain <gmcbain>
:created: 2012-10-11
'''

import numpy as np

class DySys(object):
    '''virtual base class for dynamical systems

    which can be marched in time, generating an infinite sequence of
    states

    '''

    def step(self, t, x, h):
        '''abstract method to be overridden by subclasses

        which should return the state at time t+h given that the
        initial condition x at time t

        '''

        raise NotImplementedError

    def march(self, x0, h):
        '''generate the sequence of pairs of times and states

        from the initial condition x0 at time 0.0 with constant
        time-step h, using the step method

        '''

        t, x = 0.0, x0
        while True:
            yield t, x
            t, x = t + h, self.step(t, x, h)
