#!/usr/bin/python
# -*- coding: latin-1 -*-


'''A dynamical system

taking approximate discrete steps in continuous time

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

    def __init__(self, parameters=None, master=None):
        self.parameters = {} if parameters is None else parameters
        self.master = master

    @property
    def zero(self):
        '''return the zero element of the vector space'''
        return np.zeros(len(self))

    @property
    def identity(self):
        '''return the (CSR sparse) identity matrix

        :rtype: scipy.sparse.csr_matrix


        '''

        return identity(len(self), format='csr')

    def as_master(self, x=None, d=None, f=None):
        '''return a dict representing self as a master-system

        :param x: state, optional (default self.zero)

        :param d: dict, discrete dynamical variables (default empty)

        :param f: (DySys, t, x, d) -> ?

        :rtype: dict

        '''

        return {'system': self,
                'state': self.zero if x is None else x,
                'd': {} if d is None else d,
                'f': f}

    def equilibrium(self, y0=None, d=None, **kwargs):
        '''return an eventual steady-state solution

        :param y0: initial guess, maybe optional, maybe ignored

        :param d: dict, discrete dynamical variables, optional

        Further keyword-arguments may be passed on to the back-end
        solver.

        '''

        raise NotImplementedError

    def step(self, t, h, y, d):
        '''abstract method to be overridden by subclasses

        which should return the state at time t+h given the initial
        condition y at time t; d is an optional object containing
        discrete dynamical parameters

        Note that d is not to be returned; it is only modified by
        'events' during self.march.

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

    def handle_event(self, f, t, x, d):
        '''handle event

        :param f: function of self, t, x, d that returns x, d, possibly
        modified

        :param t: float > 0, time

        :param x: ndarray of continuous dynamical variables

        :param d: dict of discrete dynamical variables

        Subclass designers: Override if more needs to be done, using
        super to re-call this.

        '''

        return f(self, t, x, d)

    def march(self, h, x=None, d=None, events=None, substeps=1, f=None):
        '''generate the evolution of the system in time,

        continuously according to the differential equation, but also
        punctated by a sorted iterable of events

        each of which is a pair of the time at which it is scheduled
        and its mapping of the old state to the new

        :param h: time-step (float)

        :param x: initial condition, typically a one-dimensional
        numpy.ndarray, but may vary with subclass

        :param d: dict, for discrete dynamical variables, e.g. to be
        accessed by the functions in events

        :param events: iterable of pairs of time and mapping on time,
        x and d; the second term may be None for the identity mapping,
        if it is just desired to force a step at that time

        :param substeps: number of equal substeps to take to make up
        each time-step (default: 1)

        :param f: optional (x -> anything).  If provided, this maps
        the states x before output.  This can be used to reconstitute
        the results from a constrained LinearDySys, or for
        postprocessing: "Usually, only a small portion of data needs
        to be saved in order to concisely record the pertinent
        features of the dynamics."  (PyDSTool Project overview)

        :rtype: yield indefinitely triples (time, continuous state,
         discrete state) at ends of time-steps

        See also: march_till, march_while

        '''

        t = 0.
        x = self.zero if x is None else x
        d = {} if d is None else d

        # TRICKY gmcbain 2013-05-09: Append an event at infinite time
        # so that the events iterable is never exhausted.  The
        # associated function will never be called; np.asarray is
        # chosen as it is near enough to an identity.

        for epoch, change in it.chain([] if events is None else events,
                                      [(np.inf, np.asarray)]):
            while True:
                yield t, (x if f is None else f(x)), d
                if t + h > epoch:

                    x = self._step(t, epoch - t, x, d, substeps)
                    t = epoch
                    yield t, (x if f is None else f(x)), d

                    if change is not None:
                        x, d = self.handle_event(change, t, x, d)
                    break
                else:
                    t, x = t + h, self._step(t, h, x, d, substeps)

    def march_truncated(self, condition, *args, **kwargs):
        '''truncate a march when condition fails

        :param condition: a predicate on triples of time, continuous
        state, and dict of discrete dynamical variables

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
            lambda event: predicate(event[1], event[2]), *args, **kwargs)

    def node_maps(self, known):

        '''return the matrices mapping the unknown and knowns

        to the global nodes

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

        size = len(self)
        I = self.identity       # noqa E741

        return ((I, np.zeros((size, 0))) if len(known) == 0
                else
                (I[:, np.setdiff1d(np.arange(size), np.mod(known, size))],
                 I[:, known]))

    @staticmethod
    def reconstituter(U, K, x, u):
        '''reinsert the known degrees of freedom stripped out by constrain

        This is an identity mapping if the system is not constrained
        (determined by assuming that the system will only have the
        attribute U if its constrain method has been called).

        '''

        return U @ u + (0 if x is None else K @ x)

    @staticmethod
    def projector(U, x):
        '''map to constrained space

        using the left-inverse of U from self.node_maps

        :param U: linear operator from constrained space to total
        space, as returned by self.node_maps

        :param x: vector in total space

        The idea is that if x = U @ u + K @ k, then U.T @ x =
        (U.T @ U) @ u, assuming U.T @ K = 0.

        Further assuming that U is orthogonal in the sense that
        U.T @ U is the identity, we have u = U.T @ x.

        '''

        return U.T @ x

    def forcing(self, t, h, x, d, inputs=None):
        '''return forcing at start and end of time-step

        :param t: float, time

        :param h: float > 0, time-step

        :param x: state

        :param d: dict, discrete dynamical variables

        :param inputs: pair, inputs at the start and end of step
        [optional: default None]

        '''

        d = d or {}

        if inputs:
            [fold, fnew] = map(lambda t, y: self.f(self, t, x, d, y),
                               [t, t + h], inputs)
        elif self.f is not None:
            [fold, fnew] = map(lambda t: self.f(self, t, x, d), [t, t + h])
        elif self.master is not None:
            yold = self.master.pop('state')
            try:
                ynew = self.master['state'] = self.master['system'].step(
                    t, h, yold, self.master.get('d'))
            except ZeroDivisionError:
                ynew = self.master['state'] = yold
            [fold, fnew] = map(
                lambda y: self.master['f'](self, t, y, self.master.get('d')),
                [yold, ynew])
        else:
            fold = fnew = self.zero

        return [fold, fnew]

    def eig(self, *args, **kwargs):

        '''return the complete spectrum of the system

        Designed for small dense systems; see self.eigs for large
        sparse systems.

        '''

        return NotImplemented

    def eigs(self, *args, **kwargs):
        '''return the first few modes of the system

        Designed for large sparse systems; default to self.eig, converting
        to dense, if the system is too small.

        '''

        return NotImplemented


# def node_maps(known, size):
#     '''return the matrices mapping the unknown and knowns

#     to the global nodes.

#     This concerns the imposition of nodal degree-of-freedom
#     constraints as inspired by the comments of Roy Stogner in the
#     Libmesh-users list thread "interaction between subdomain_id
#     and dof constraints?"  (2012-02-25).  The idea is to represent
#     the column vector of all unknowns x as U * xu + K * xk, where
#     xu are unknown and xk are known whose lengths together add to
#     that of x and U and K are rectangular matrices, typically
#     columns of the identity.

#     '''

#     # KLUDGE: gmcbain 2013-01-29: I don't know how to deal with
#     # arrays with zero rows or columns in scipy.sparse, so I need
#     # to treat it as a special case.  Yuck.  GNU Octave does the
#     # obvious right thing.  I think the problem applies to NumPy
#     # too.

#     # TRICKY gmcbain 2013-06-28: Between versions 0.10.1 and
#     # 0.12.0, SciPy switched from having scipy.sparse.identity
#     # return csr_matrix to dia_matrix.  This broke the code
#     # below since the latter does not support indexing!
#     # (i.e. raises TypeError: dia_matrix object has no
#     # attribute __getitem__)

#     I = identity(size, format='csr')
#     if len(known) > 0:
#         U = I[:, np.setdiff1d(np.arange(size),
#                               np.mod(known, size))]
#         return U, I[:, known]
#     else:
#         return [I, np.zeros((size, 0))]
