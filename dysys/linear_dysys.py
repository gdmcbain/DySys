#!/usr/bin/env python

'''a module for linear 'descriptor' systems

:author: G. D. McBain <gmcbain>

:created: 2013-01-11

'''

import numpy as np
from scipy.sparse import identity

from dysys import DySys

class LinearDySys(DySys):
    
    def __init__(self, M, D, f):
        '''a DySys defined by mass and damping operators and
        a time-dependent forcing function, according to (something like)

            M * x' + D * x = f (t)

        though this class is still virtual since it depends on:

          . the implementation of the M & D operators (e.g. as sparse)

          . the discretization of the temporal derivative.

        '''

        self.M, self.D, self.f = M, D, f

    def __len__(self):
        return self.D.shape[0]

    def node_maps(self, known):
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
        
        return ([identity(len(self))[:,c] for c in
                 (np.setdiff1d(np.arange(len(self)), known), known)]
                if len(known) > 0 else
                [identity(self.nodes), None])

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

        U, K = self.node_maps(known)
        (M, D) = [None if A is None else U.T * A * U for A in [self.M, self.D]]
        sys = self.__class__(
            M, 
            D,
            lambda t: U.T * (
                self.f(t) -
                (0 if xknown is None else self.D * K * xknown) -
                (0 if vknown is None else self.M * K * vknown)))
        sys.U, sys.K, sys.xknown = U, K, xknown
        return sys

    def reconstitute(self, x):
        "don't try this except on systems returned by constrain"
        return (self.U * x +
                (0 if self.xknown is None else self.K * self.xknown))
        
