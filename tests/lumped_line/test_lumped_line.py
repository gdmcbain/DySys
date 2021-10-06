import numpy as np
from numpy.lib.ufunclike import fix
from scipy.sparse import bmat, diags

from dysys import SparseDySys

from pytest import fixture


class LumpedLine:

    """Consider a single lumped hydraulic conduit (two-port network)

    with serial resistance R and inertance L, shunt compliance C and
    duty Q and fixed inlet pressure

              R         L
    IN o---^^^^^^^---%%%%%%%%--T----T-----o OUT
             q->               |    |
                               |    | ^
                             C =    8 | Q
                               |    |
                               _    _
                               -    -
                               .    .


    """

    def __init__(self):
        self.pin = -981.0
        self.R = 4.680  # Pa.s/uL9
        self.C = 2.4519e-3  # uL/Pa
        self.L = 0.073557  # Pa.s^2/uL
        self.Q = -267.48  # uL/s

        M = diags([[0, self.C, self.L]], [0])
        B = bmat([[+1], [-1]])
        D = bmat([[None, B], [-B.T, np.array([self.R])]])
        self.sys = SparseDySys(M, D, lambda *_: [0, self.Q, 0])


@fixture
def lumped_line():
    return LumpedLine()


def test_equilibrium(lumped_line):
    """pressure-drop is all across the resistance

    the branch serial flow-rate being absorbed by the duty

    """

    sys = lumped_line.sys.constrain([0], [lumped_line.pin])
    soln = sys.reconstitute(sys.equilibrium())
    p, q = soln[:2], soln[2]
    np.testing.assert_array_almost_equal(
        p, lumped_line.pin + np.array([0, lumped_line.Q * lumped_line.R])
    )
    np.testing.assert_array_almost_equal(q, -lumped_line.Q)


def test_harmonic(lumped_line):
    f = np.array([14.0])
    omega = 2 * np.pi * f
    sys = lumped_line.sys.constrain([0], [0.0])
    soln = sys.reconstitute(sys.harmonic(omega)[0])
    p, q = soln[:2], soln[2]

    Z = lumped_line.R + 1j * omega * lumped_line.L
    Y = 1j * omega * lumped_line.C + 1 / Z

    np.testing.assert_array_almost_equal(
        p, np.concatenate(np.broadcast_arrays([0], lumped_line.Q / Y))
    )
    np.testing.assert_array_almost_equal(q, -lumped_line.Q / (Y * Z))
