#!/usr/bin/python
# -*- coding: latin-1 -*-


'''A demonstration of using the backward Euler method 

of marching a sparse-linear differential-algebraic system in time.


:author: G. D. McBain <gmcbain>

:created: 2012-10-11

'''

from __future__ import absolute_import, division, print_function

import itertools as it

import numpy as np
from scipy.sparse import identity


def stepper(stepping_function):
    '''decorator to do nothing for steps of zero length

    i.e. instead of returning a new state, return the old one, which
    is argument 2, following the DySys object and the time

    '''

    def wrapper(*args, **kwargs):
        try:
            return stepping_function(*args, **kwargs)
        except ZeroDivisionError:  # assume step is zero
            return args[3]
    return wrapper


class DySys(object):
    '''virtual base class for dynamical systems

    which can be marched in time, generating an infinite sequence of
    states

    '''

    def step(self, t, h, y, d):
        '''abstract method to be overridden by subclasses

        which should return the state at time t+h given the initial
        condition y at time t; d is an optional object containing
        discrete dynamical parameters

        The basic idea is that: '...the nature of time-stepping is
        inherently sequential or local; given the "state" y(t), the
        method is a procedure for computing an approximation to y(t+h)
        a time-step h>0 ahead.  The size of h is used to trade
        accuracy for efficiency and vice versa, and is therefore the
        principal _internal_ means of controlling the error'
        (Söderlind 2002, S. 1)

        '''

        raise NotImplementedError

    @stepper
    def _step(self, t, h, y, d=None, substeps=1):
        'wrap the step method as universally required'

        # TODO 2013-04-11 gmcbain: The option of substeps here does
        # not do much except suppress output for the intermediate
        # steps; however, the idea of introducing the feature was that
        # this might be useful for developing "extrapolation methods"
        # (Hairer, Lubich, & Roche 1989, p. 16; Brenan, Campbell, &
        # Petzold 1996, S. 4.6, pp. 108--114).  It might also be
        # useful for the estimation of error and step-size control.

        h /= substeps
        for i in range(substeps):
            t, y = t + h, self.step(t, h, y, d)
        return y

    # def simple_march(self, x0, h):
    #     '''generate the sequence of pairs of times and states

    #     from the initial condition x0 at time 0.0 with constant
    #     time-step h, using the step method

    #     '''

    #     t, x = 0.0, x0
    #     while True:
    #         yield t, x
    #         t, x = t + h, self.step(t, h, x)

    def march(self, h, x, d=None, events=None, substeps=1, f=None):
        '''like simple_march, but punctated by a sorted iterable of events

        each of which is a pair of the time at which it is scheduled
        and its mapping of the old state to the new

        :param x: initial condition, a numpy.array or a pair of an
        numpy.array and a dict of discrete variables

        :param h: time-step (float)

        :param d: dict, for discrete dynamical variables, e.g. to be
        accessed by the functions in events

        :param events: iterable of pairs of time and mapping on time,
        x and d

        :param substeps: number of equal substeps to take to make up
        each time-step (default: 1)

        :param f: if provided, this maps the states before output.
        This can be used to reconstitute the results from a
        constrained LinearDySys, or for postprocessing: "Usually, only
        a small portion of data needs to be saved in order to
        concisely record the pertinent features of the dynamics."
        (PyDSTool Project overview)

        :rtype: yield indefinitely triples (time, continuous state,
         discrete state) at ends of time-steps

        See also: march_till, march_while

        '''

        t, d = 0.0, {} if d is None else d

        # TRICKY gmcbain 2013-05-09: Append an event at infinite time
        # so that the events iterable is never exhausted.  The
        # associated function will never be called; np.asarray is
        # chosen as it is near enough to an identity.

        for event in it.chain([] if events is None else events,
                              [(np.inf, np.asarray)]):
            while True:
                yield t, (x if f is None else f(x)), d
                if t + h > event[0]:
                    # step to just before event
                    x = self._step(t, event[0] - t, x, d, substeps)
                    t = event[0]
                    yield t, (x if f is None else f(x)), d

                    # event
                    x, d = event[1](t, x, d)
                    break
                else:
                    t, x = t + h, self._step(t, h, x, d, substeps)

    def march_truncated(self, condition, *args, **kwargs):
        '''truncate a march when condition fails

        :param condition: a predicate on pairs of time and state

        :rtype: like march but truncated

        For immediate inspection, the output is conveniently passed to
        dict and then perhaps pandas.DataFrame, but more usually would
        be mapped between those two steps.

        See also: march_till, march_while

        '''

        return it.takewhile(condition, self.march(*args, **kwargs))

    def march_till(self, endtime, *args, **kwargs):
        '''march until the time passes endtime

        :param endtime: float

        :rtype: like march_truncated

        See also: march, march_while, march_truncated

        '''

        return self.march_truncated(lambda event: event[0] < endtime,
                                    *args, **kwargs)

    def march_while(self, predicate, *args, **kwargs):
        '''march until the state fails the predicate

        :param predicate: boolean function of continuous state and
        discrete state

        :rtype: like march_truncated

        See also: march, march_till, march_truncated

        '''

        return self.march_truncated(
            lambda event: predicate(event[1], **event[2]), *args, **kwargs)


def node_maps(known, size):
    '''return the matrices mapping the unknown and knowns

    to the global nodes.

    This concerns the imposition of nodal degree-of-freedom
    constraints as inspired by the comments of Roy Stogner in the
    Libmesh-users list thread "interaction between subdomain_id
    and dof constraints?"  (2012-02-25).  The idea is to represent
    the column vector of all unknowns x as U * xu + K * xk, where
    xu are unknown and xk are known whose lengths together add to
    that of x and U and K are rectangular matrices, typically
    columns of the identity.

    '''

    # KLUDGE: gmcbain 2013-01-29: I don't know how to deal with
    # arrays with zero rows or columns in scipy.sparse, so I need
    # to treat it as a special case.  Yuck.  GNU Octave does the
    # obvious right thing.  I think the problem applies to NumPy
    # too.

    # TRICKY gmcbain 2013-06-28: Between versions 0.10.1 and
    # 0.12.0, SciPy switched from having scipy.sparse.identity
    # return csr_matrix to dia_matrix.  This broke the code
    # below since the latter does not support indexing!
    # (i.e. raises TypeError: dia_matrix object has no
    # attribute __getitem__)

    I = identity(size, format='csr')
    if len(known) > 0:
        U = I[:, np.setdiff1d(np.arange(size),
                              np.mod(known, size))]
        return U, I[:, known]
    else:
        return [I, np.zeros((size, 0))]
