#!/usr/bin/env python

'''a module for linear 'descriptor' systems

:author: G. D. McBain <gmcbain>

:created: 2013-01-11

'''

import numpy as np
from scipy.sparse import identity

from dysys import DySys

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

        def node_maps(known):
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

        U, K = node_maps(known)
        (M, D) = [None if A is None else U.T * A * U for A in [self.M, self.D]]
        sys = self.__class__(
            M, 
            D,
            lambda *args: U.T * (
                (0 if self.f is None else self.f(*args)) -
                (0 if xknown is None else self.D * K * np.array(xknown)) -
                (0 if vknown is None else self.M * K * np.array(vknown))))
        sys.U, sys.K, sys.xknown, sys.vknown = U, K, xknown, vknown
        return sys

    # TODO gmcbain 2013-05-17: It would be nice for LinearDySys to
    # override the march method from DySys so that it wasn't necessary
    # to pass sys.reconstitute, but I haven't figured out how to
    # override a generator function, still invoking the inherited one
    # with super, despite having read
    # http://stackoverflow.com/questions/8076312 'Subclassing and
    # overriding a generator function in python' (which is for Python
    # 3 and doesn't seem to work anyway).

    def reconstitute(self, x):
        '''reinsert the known degrees of freedom stripped out by constrain

        This is an identity mapping if the system is not constrained
        (determined by assuming that the system will only have the
        attribute U if its constrain method has been called).

        '''

        return ((self.U * x[0] +
                 (0 if self.xknown is None else self.K * self.xknown))
                if hasattr(self, 'U') else x[0],) + x[1:]
