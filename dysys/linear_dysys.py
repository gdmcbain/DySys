#!/usr/bin/env python

'''a module for linear 'descriptor' systems

:author: G. D. McBain <gmcbain>

:created: 2013-01-11

'''

from __future__ import absolute_import, division, print_function

import numpy as np

from dysys import DySys, node_maps


class LinearDySys(DySys):

    def __init__(self, M, D, f=None):
        '''a DySys defined by mass and damping operators and
        a time-dependent forcing function, according to (something like)

            M * x' + D * x = f (t)

        though this class is still virtual since it depends on:

          . the implementation of the M & D operators (e.g. as sparse)

          . the discretization of the temporal derivative.

        Since occasionally the steady-state D * x = f (inf) is of
        interest, M may be None.

        '''

        self.M, self.D, self.f = M, D, f

    def __len__(self):
        return self.D.shape[0]

    def constrain(self, known, xknown=None, vknown=None):
        '''return a new DySys with constrained degrees of freedom

        having the same class as self.

        :param known: sequence of indices of known degrees of freedom

        :param xknown: corresponding sequence of their values
        (default: zeros)

        :param vknown: corresponding sequence of their rates of change

        The returned system is attributed the U and K matrices from
        self.node_maps and therefore can use :method reconstitute:.

        '''

        U, K = node_maps(known, len(self))
        M, D = [None if A is None else U.T * A * U for A in [self.M, self.D]]
        sys = self.__class__(
            M,
            D,
            lambda *args: U.T.dot(
                (0 if self.f is None else self.f(*args)) -
                (0 if xknown is None else self.D.dot(K.dot(xknown))) -
                (0 if vknown is None else self.M.dot(K.dot(vknown)))))

        def reconstitute(u):
            '''reinsert the known degrees of freedom stripped out by constrain

            This is an identity mapping if the system is not constrained
            (determined by assuming that the system will only have the
            attribute U if its constrain method has been called).

            '''
            return U.dot(u) + (0 if xknown is None else K.dot(xknown))

        sys.reconstitute = reconstitute
        return sys
