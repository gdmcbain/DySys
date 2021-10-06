from functools import lru_cache

import numpy as np
from pandas import Series

from dysys import ScalarLinearDySys


from pytest import fixture


class Scalar:

    """Consider a single lumped hydraulic conduit (two-port network)

    with serial resistance R and inertance L, shunt compliance C and
    duty Q and fixed inlet pressure

              R
    IN o---^^^^^^^-------------T----------o OUT
             q->               |
                               |
                             C =
                               |
                               _
                               -
                               .

    Treating p[0] as given, after eliminating q = (p[1] - p[0]) / R,
    the governing equation is RC dp[1]/dt + p[1] = p[0].

    This is described by ScalarLinearDySys(RC, 1, lambda *_: p[0]).

    """

    R, C = 2.0, 3.0
    p_in = 5.0
    
    @property
    def system(self):
        return ScalarLinearDySys(self.R * self.C, 1.0, lambda *_: self.p_in, theta=0.5)


@fixture
def scalar():
    s = Scalar()
    s.sys = s.system
    return s


def test_equilibrium(scalar):
    """The exact solution is p[1] = p[0]."""
    np.testing.assert_almost_equal(scalar.sys.equilibrium(), scalar.p_in)

def test_harmonic(scalar):
    """See Roadstrum & Wolaver (1987)

    (§5.5 'Frequency response of first-order circuits').

    """

    omega = np.linspace(0, 5 / scalar.R / scalar.C)
    np.testing.assert_almost_equal(
        scalar.sys.harmonic(omega), scalar.p_in / (1 + 1j * omega * scalar.R * scalar.C)
    )

def test_march(scalar):
    """See Roadstrum & Wolaver (1987)

    (§6.2 'Differential equations').

    """

    p = Series(
        {
            t / scalar.R / scalar.C: x
            for t, x, _ in scalar.sys.march_till(
                5 * scalar.R * scalar.C, scalar.R * scalar.C / 1e3
            )
        }
    )

    np.testing.assert_array_almost_equal(p, scalar.p_in * (1 - np.exp(-p.index)))
