#!/usr/bin/python
# -*- coding: latin-1 -*-


'''
A demonstration of using the backward Euler method of marching a
sparse-linear differential-algebraic system in time.

:author: G. D. McBain <gmcbain>
:created: 2012-10-11
'''

import itertools as it

import numpy as np
import pandas as pd

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
    def _step(self, t, y, h, substeps=1):
        'wrap the step method as universally required'

        # TODO 2013-04-11 gmcbain: The option of substeps here does
        # not do much except suppress output for the intermediate
        # steps; however, the idea of introducing the feature was that
        # this might be useful for developing "extrapolation method"
        # (Brenan, Campbell, & Petzold 1996, S. 4.6, pp. 108--114).
        # It might also be useful for the estimation of error and
        # step-size control.

        # Brenan, K. E., S. L. Campbell, & L. R. Petzold
        # (1996). Numerical solution of initial-value problems in
        # differential-algebraic equations, Volume 14 of Classics in
        # Applied Mathematics. Philadelphia: Society for Industrial
        # and Applied Mathematics

        h /= substeps
        for i in xrange(substeps):
            t, y = t + h, self.step(t, y, h)
        return y

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

    def march(self, x0, h, events=None, substeps=1):
        '''like simple_march, but punctated by a sorted iterable of events

        each of which is a pair of the time at which it is scheduled
        and its mapping of the old state to the new

        :param x0: initial condition (typically an np.array)

        :param h: time-step (float)

        :param events: iterable of pairs of time and mapping on time
        and state, state having the same type as x0 (default: empty
        list)

        :param substeps: number of equal substeps to take to make up
        each time-step (default: 1)

        :rtype: yield indefinitely pairs of time and state at end of
        time-steps

        See also: march_till, march_while

        '''

        t, x = 0.0, x0

        # TRICKY gmcbain 2013-05-09: Append an event at infinite time
        # so that the events iterable is never exhausted.  The
        # associated function will never be called; np.asarray is
        # chosen as it is near enough to an identity.

        for event in it.chain([] if events is None else events, 
                              [(np.inf, np.asarray)]):
            while True:
                yield t, x
                if t + h > event[0]:
                    t, x = event[0], self._step(t, x, event[0] - t, substeps)
                    yield t, x      # step to just before event
                    x = event[1](t, x)
                    break
                else:
                    t, x = t + h, self._step(t, x, h, substeps)

    def march_truncated(self, condition, *args, **kwargs):
        '''truncate a march when condition fails

        :param condition: a predicate on pairs of time and state
        
        :param Series: define this keyword as not False to get the
        result as a pandas.Series

        :rtype: pair of sequence of times and corresponding sequence
        of states, unless Series is defined unfalse

        See also: march_till, march_while

        '''

        series = kwargs.pop('Series', False)
        filtre = lambda h: (pd.Series(dict(h)) if series else zip(*list(h)))
        return filtre(it.takewhile(condition, self.march(*args, **kwargs)))

    def march_till(self, endtime, *args, **kwargs):
        '''march until the time passes endtime

        :param endtime: float

        :rtype: pair of sequence of times and corresponding sequence
        of states, unless the keyword argument Series is defined as
        not False in which case it is a pandas.Series

        See also: march, march_while, march_truncated

        '''

        return self.march_truncated(lambda event: event[0] < endtime,
                                     *args, **kwargs)

    def march_while(self, predicate, *args, **kwargs):
        '''march until the state fails the predicate

        :param predicate: boolean function of state

        :rtype: pair of sequence of times and corresponding sequence
        of states, unless the keyword argument Series is defined as
        not False in which case it is a pandas.Series

        See also: march, march_till, march_truncated

        '''

        return self.march_truncated(lambda event: predicate(event[1]),
                                    *args, **kwargs)
