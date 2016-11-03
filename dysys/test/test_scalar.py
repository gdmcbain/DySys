#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests on a single lump.

:author: gmcbain

:created: 2016-11-03

'''

from __future__ import absolute_import, division, print_function

import unittest

import numpy as np
from pandas import Series

from dysys import ScalarLinearDySys


class TestScalar(unittest.TestCase):

    '''Consider a single lumped hydraulic conduit (two-port network)

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

    '''

    @classmethod
    def setUpClass(cls):
        cls.R, cls.C = 2., 3.
        cls.p_in = 5.
        cls.sys = ScalarLinearDySys(cls.R * cls.C, 1.,
                                    lambda *_: cls.p_in,
                                    theta=0.5)

    def test_equilibrium(self):
        '''The exact solution is p[1] = p[0].'''
        np.testing.assert_almost_equal(self.sys.equilibrium(), self.p_in)

    def test_harmonic(self):
        '''See Roadstrum & Wolaver (1987)

        (§5.5 'Frequency response of first-order circuits').

        '''

        omega = np.linspace(0, 5 / self.R / self.C)
        np.testing.assert_almost_equal(
            self.sys.harmonic(omega),
            self.p_in / (1 + 1j * omega * self.R * self.C))

    def test_march(self):
        '''See Roadstrum & Wolaver (1987)

        (§6.2 'Differential equations').

        '''

        p = Series({t / self.R / self.C: x
                    for t, x, _ in self.sys.march_till(5 * self.R * self.C,
                                                       self.R * self.C / 1e3)})

        np.testing.assert_array_almost_equal(
            p, self.p_in * (1 - np.exp(-p.index)))
