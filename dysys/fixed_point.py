import itertools as it

import numpy as np
from scipy.sparse import issparse, spmatrix
from scipy.sparse.linalg import spsolve


def fixed_point(iteration, tol=np.finfo(float).eps, maxiter=np.iinfo(int).max):
    '''

    :param: iteration, iterable generating pairs of values and
    convergence test values

    :param: tol, positive float

    :param: maxiter, positive integer

    When called (without arguments), the object returns the first
    element of the first pair for which the second value is less than
    tol (in the sense of numpy.linalg.norm)

    '''

    # TODO gmcbain 2015-03-23: Investigate replacing this with
    # scipy.optimize.fixed_point. DySys#38

    return next(y for y, h in it.islice(iteration, maxiter)
                if np.linalg.norm(h) < tol)


def newton(residual, jacobian, x, *args, **kwargs):
    '''eliminate the residual by Newton-iteration

    :param: residual, a function taking a one-dimensional
    numpy.ndarray to another array of the same len

    :param: jacobian, a function like residual but returning a square
    numpy.array of corresponding shape

    :param: x, a one-dimensional numpy.ndarray of the length expected
    by the residual and jacobian functions

    Any other positional or keyword arguments are passed on to
    fixed_point; of particular interest are tol and maxiter.

    '''

    # TODO gmcbain 2016-10-28: Can we really not use
    # scipy.optimize.root?  DySys#39

    def iteration(x):
        while True:
            dx = solve(jacobian(x), residual(x))
            x = x - dx
            yield x, dx

    return fixed_point(iteration(x), *args, **kwargs)


def solve(A: spmatrix,
          b: np.ndarray,
          *args, **kwargs) -> np.ndarray:
    '''solve the linear system A x = b

    Further positional and keyword arguments are passed on to
    scipy.sparse.linalg.spsolve or numpy.linalg.solve depending on
    whether scipy.sparse.issparse(A).

    '''

    if issparse(A):
        try:
            return spsolve(A, b, *args, **kwargs)
        # except ValueError:
        #             return b / A
        except IndexError:
            return b / A.toarray()[0, 0]
    else:
        return np.linalg.solve(A, b, *args, **kwargs)
