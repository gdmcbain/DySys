#!/usr/bin/env python

'''

:author: G. D. McBain <gmcbain>
:created: 2013-04-09

'''

from __future__ import absolute_import, division, print_function

import numpy as np
from scipy.sparse import identity

from .linear_dysys import LinearDySys
from .fixed_point import newton


class NonlinearSparseDySys(LinearDySys):
    
    def __init__(self, F, M, D, n=None):
        '''an alternative to SparseNFDySys 

        The system evolves according to the more general F(t, x, x') = 0.

        :param: F(t, x, v), where v is understood to be the rate of
        change of x

        :param: M(t, x, v), returning the partial derivative of F w.r.t. v

        :param: D(t, x, v), returning the partial derivative of F w.r.t. x

        :param: n, order of system, i.e. len of F; calculated from
        D(0, [], []) if omitted (which will only work if D doesn't
        inspect its arguments)

        '''

        self.F, self.M, self.D = F, M, D
        self.n = D(0, [], []).shape[0] if n is None else n

    def __len__(self):
        return self.n

    def step(self, t, xold, h, tol=1e-3):
        '''take a backward-Euler step'''

        if h == 0:
            raise ZeroDivisionError

        def rate_of_change(x):
            return (x[0] - xold[0]) / h

        def residual(x):
            '''approximate the rate of change using backward Euler'''
            return self.F(t + h, x, rate_of_change(x))

        def jacobian(x):
            # r(x + dx) = F(t, x + dx, (x + dx - xold) / h) 

            #          ~= r(x) + (F_x + F_v / h) dx

            # Thus J = F_x + F_v / h.
            v = rate_of_change(x)
            return self.M(t + h, x, v) / h + self.D(t + h, x, v)

        return newton(residual, jacobian, xold, tol)

    def equilibrium(self, x0, tol=1e-3):
        '''take an infinitely long backward-Euler step'''
        
        def residual(x):
            # r(x) = F(oo, x, 0)
            return self.F(np.inf, x, np.zeros(x[0].shape))

        def jacobian(x):
            # r(x+dx) = F(oo, x + dx, 0) ~ r(x) + D(t, x) dx
            return self.D(np.inf, x)

        return newton(residual, jacobian, x0, tol)

    def constrain(self, known, xknown=None, vknown=None):
        '''return a new NonlinearSparseDySys with constrained DoFs

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

            # TRICKY gmcbain 2013-06-28: Between versions 0.10.1 and
            # 0.12.0, SciPy switched from having scipy.sparse.identity
            # return csr_matrix to dia_matrix.  This broke the code
            # below since the latter does not support indexing!
            # (i.e. raises TypeError: dia_matrix object has no
            # attribute __getitem__)

            if len(known) > 0:
                I = identity(len(self), format='csr')
                U = I[:, np.setdiff1d(np.arange(len(self)),
                                      np.mod(known, len(self)))]
                return U, I[:, known]
            else:
                return [identity(len(self)), None]

        U, K = node_maps(known)
        
        def arg_map(t, u, u1):
            '''transform the arguments for the constraining'''
            return (t, 
                    (U.dot(u[0]) + (0 if xknown is None else K.dot(xknown)),),
                    U.dot(u1) + (0 if vknown is None else K.dot(vknown)))

        sys = self.__class__(
            lambda t, u, u1: U.T.dot(self.F(*arg_map(t, u, u1))),
            lambda t, u, u1: U.T.dot(self.M(*arg_map(t, u, u1))).dot(U),
            lambda t, u, u1: U.T.dot(self.D(*arg_map(t, u, u1))).dot(U),
            U.shape[1])
        sys.U, sys.K, sys.xknown, sys.vknown = U, K, xknown, vknown
        return sys

    def reconstitute(self, u):
        '''put back the known degrees of freedom constrained out

        '''

        return ((self.U.dot(u[0]) +
                 (0 if self.xknown is None else self.K.dot(self.xknown)))
                ,) + u[1:]
