from functools import partial
from typing import Callable, Optional
from warnings import warn

import numpy as np
from scipy.sparse import block_diag, bmat, linalg as sla, spmatrix
from scipy.sparse.linalg import splu

from dysys.dysys import DySys
from dysys.fixed_point import solve
from dysys.linear_dysys import SparseDySys


class Newmark(DySys):
    """a dynamical system advancing with a Newmark method

    having constant sparse mass, damping, and stiffness matrices and a
    forcing function depending on time

    A Newmark system evolves with the "displacement" as the
    dynamical variable but also has "state" in the form of the
    velocity and acceleration, the former being required as the
    system is of second order while the latter is merely convenient.

    The state variable x is a pair of numpy.ndarrays, being
    displacement and velocity; the acceleration is kept in an
    attribute Newmark.a.

    """

    def __init__(self,
                 M: spmatrix,
                 K: spmatrix,
                 C: Optional[spmatrix]=None,
                 f: Optional[Callable]=None,
                 beta: float=0.25,
                 gamma: float=0.5,
                 definite: bool=False):
        """:param M: mass scipy.sparse matrix

        :param K: stiffness scipy.sparse

        :param C: damping scipy.sparse matrix, or None, in which case
        it is constructed as like M but with no nonzero entries

        :param f: function of (time, state (typically ignored), dict
        of discrete dynamical variables), returning forcing vector, or
        None in which case a ternary zero-function is substituted

        :param beta: Newmark method parameter, default 0.25 (which,
        with gamma=0.5, is the implicit and unconditionally stable
        "average acceleration" method: Hughes 2000, p. 493)

        :param gamma: Newmark method parameter, default 0.5 (as
        required for second order accuracy: Hughes 2000, Table 9.1.1,
        note 3)

        :param definite: bool, for if system is (positive-)definite

        """

        self.M, self.K, self.C = M, K, C
        self.f = f or (lambda *args: self.zero[0])
        self.beta, self.gamma = beta, gamma
        self.definite = definite

    def __len__(self):
        return self.K.shape[0]

    @property
    def zero(self):
        return (np.zeros(len(self)),)*2

    def equilibrium(self,
                    x: np.ndarray=None,
                    d: np.ndarray=None,
                    *args, **kwargs) -> (np.ndarray,
                                         np.ndarray):
        """return the eventual steady-state solution

        using self.forcing(np.inf, np.inf, x, d)

        :param x: optional initial guess, passed on to self.forcing,
        where it should be ignored [default: None]

        :param d: optional dict, passed on to self.forcing [default:
        None]

        Further positional arguments are passed on to self.forcing;
        keyword arguments to solve.

        """

        return (solve(self.K,
                      self.forcing(np.inf, np.inf, x, d, *args)[1],
                      **kwargs),
                self.zero[1])

    def prestep(self, t, h, x, d, *args):
        if not hasattr(self, '_memo') or self._memo['h'] != h:
            rhs = self.forcing(t, h, x, d, *args)[1] - self.K @ x[0]
            if self.C is not None:
                rhs -= self.C @ x[1]
            self.a = solve(self.M, rhs)
            self.setA(h)
            self._memo = {'h': h}

    def step(self, t, h, x, d, *args):
        'evolve from displacement x at time t to t+h'

        self.prestep(t, h, x, d, *args)

        xt = (x[0] + h * (x[1] + h * (.5 - self.beta) * self.a),
              x[1] + (1 - self.gamma) * h * self.a)

        rhs = self.forcing(t, h, x, d, *args)[1] - self.K @ xt[0]

        if self.C is not None:
            rhs -= self.C @ xt[1]

        self.a = self.solve(rhs)
        return (xt[0] + self.beta * h**2 * self.a,
                xt[1] + self.gamma * h * self.a)

    def setA(self, h, alpha=0.):
        """set the acceleration evolution matrix

        :param h: float > 0, time-step

        :param alpha: optional float, for use by the
        HilberHughesTaylor subclass, defaulting to 0., in which case
        Hilber, Hughes, & Taylor's alpha-method degenerates to Newmark

        """

        A = self.M + (1 + alpha) * h**2 * self.beta * self.K
        if self.C is not None:
            A += (1 + alpha) * h * self.gamma * self.C
        self.solve = splu(A).solve

    def constrain(self, known, xknown=None, vknown=None, aknown=None):
        """return a new DySys with constrained degrees of freedom

        having the same class as self.

        :param known: sequence of indices of known degrees of freedom

        :param xknown: corresponding sequence of their values
        (default: zeros)

        :param vknown: corresponding sequence of their rates of change

        :param aknown: corresponding sequence of their second
        derivatives

        """

        # TODO gmcbain 2016-07-27: Refactor!

        U, Kn = self.node_maps(known)
        project = partial(self.projector, U)

        M, K, C = [None if A is None else project(A * U)
                   for A in [self.M, self.K, self.C]]
        sys = self.__class__(
            M,
            K,
            C,
            lambda *args: project(
                (0 if self.f is None else self.f(*args)) -
                (0 if xknown is None else self.K @ Kn @ xknown) -
                (0 if vknown is None else self.C @ Kn @ vknown) -
                (0 if aknown is None else self.M @ Kn @ aknown)),
            self.beta, self.gamma, self.definite)

        reconstituter = partial(self.reconstituter, U, Kn)

        def reconstitute(xv):
            return (reconstituter(xknown, xv[0]),
                    reconstituter(vknown, xv[1]))

        sys.reconstitute = reconstitute
        sys.project = project

        return sys

    def eigs(self, *args, **kwargs):
        """return the first few modes of the system"""

        if 'sigma' not in kwargs:  # inverse iteration
            kwargs['sigma'] = 0.   # Hughes (2000, §10.5.2)
        if self.C is None:
            kwargs['M'] = self.M
            try:
                retval = ((sla.eigsh if self.definite else sla.eigs)
                          (-self.K, *args, **kwargs))
                if kwargs.get('return_eigenvectors', False):
                    return np.sqrt(-retval[0]), retval[1]
                else:
                    return np.sqrt(-retval)
            except ValueError:
                warn('system too small, converting to dense', UserWarning)
                for k in ['k', 'M', 'which']:
                    if k in kwargs:
                        del kwargs[k]
                kwargs['right'] = kwargs.pop('return_eigenvectors')
                return self.eig(*args, **kwargs)
        else:

            return self.to_sparse_dysys().eigs(*args, **kwargs)

    def to_sparse_dysys(self, theta: float=0.5) -> SparseDySys:
        """return an equivalent SparseDySys

        by introducing the rate of change as an auxiliary variable

        """

        return SparseDySys(block_diag([self.identity,
                                       self.M]).tocsc(),
                           bmat([[None, -self.identity],
                                 [self.K, self.C]]),
                           lambda *fargs: np.concatenate([self.zero[0],
                                                          self.f(*fargs)]),
                           theta)


# Define special cases, as per Hughes (2000, Table 9.1.1, p. 493)

trapezoidal = partial(Newmark, beta=.25, gamma=.5)

linear_acceleration = partial(Newmark, beta=1/6, gamma=.5)

fox_goodwin = partial(Newmark, beta=1/12, gamma=.5)

central_difference = partial(Newmark, beta=0, gamma=.5)
