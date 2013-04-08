#!/usr/bin/python
# -*- coding: latin-1 -*-


'''
A demonstration of using the backward Euler method of marching a
sparse-linear differential-algebraic system in time.

:author: G. D. McBain <gmcbain>
:created: 2012-10-11
'''

import numpy as np

def stepper(stepping_function):
    '''decorator to do nothing for steps of zero length

    i.e. instead of returning a new state, return the old one, which
    is argument 2, following the DySys object and the time

    '''

    def wrapper(*args):
        try:
            return stepping_function(*args)
        except ZeroDivisionError: # assume step is zero
            return args[2]
    return wrapper

class DySys(object):
    '''virtual base class for dynamical systems

    which can be marched in time, generating an infinite sequence of
    states

    '''

    def step(self, t, y, h):
        '''abstract method to be overridden by subclasses

        which should return the state at time t+h given the initial
        condition y at time t

        The basic idea is that: '...the nature of time-stepping is
        inherently sequential or local; given the "state" y(t), the
        method is a procedure for computing an approximation to y(t+h)
        a time-step h>0 ahead.  The size of h is used to trade
        accuracy for efficiency and vice versa, and is therefore the
        principal _internal_ means of controlling the error'
        (Söderlind 2002, S. 1)

        Söderlind, G. (2002). Automatic control and adaptive
        Time-Stepping. Numerical Algorithms 31(1-4):281-310

        '''

        raise NotImplementedError

    @stepper
    def _step(self, t, y, h):
        'wrap the step method as universally required'
        return self.step(t, y, h)

    def simple_march(self, x0, h):
        '''generate the sequence of pairs of times and states

        from the initial condition x0 at time 0.0 with constant
        time-step h, using the step method

        '''

        # TODO 2013-03-01 gmcbain: Maybe add an optional filter f to
        # yield t, f(x) instead, since "Usually, only a small portion
        # of data needs to be saved in order to concisely record the
        # pertinent features of the dynamics." (PyDSTool Project
        # overview)
        
        t, x = 0.0, x0
        while True:
            yield t, x
            t, x = t + h, self.step(t, x, h)

    def march(self, x0, h, events=None):
        '''like simple_march, but punctated by a sorted iterable of events

        each of which is a pair of the time at which it is scheduled
        and its mapping of the old state to the new

        '''

        if events is None:
            events = []
            
        t, x = 0.0, x0
        for event in events:
            while True:
                yield t, x
                if t + h > event[0]:
                    t, x = event[0], self._step(t, x, event[0] - t)
                    yield t, x      # step to just before event
                    x = event[1](x)
                    break
                else:
                    t, x = t + h, self._step(t, x, h)
        while True:             # events exhausted
            yield t, x
            t, x = t + h, self._step(t, x, h)

    march_punctuated = march    # backwards-compatibility alias
    
        
