#!/usr/bin/env python

'''Tests on a single branch.

Based on corresponding test in Millihydraulics.

:author: G. D. McBain <gmcbain>

:created: 2014-05-08

'''

from __future__ import absolute_import, division, print_function

from unittest import TestCase, main

import numpy as np
from scipy.sparse import bmat, coo_matrix, diags

from dysys import SparseDySys


class TestLumpedLine(TestCase):

    '''Consider a single lumped hydraulic conduit 

    with serial resistance R and inertance L, shunt compliance C and
    duty Q and fixed inlet pressure

    '''

    
    @classmethod
    def setUpClass(cls):
        cls.pin = -981.0
        cls.R = 4.680          # Pa.s/uL9
        cls.C = 2.4519e-3      # uL/Pa
        cls.L = 0.073557       # Pa.s^2/uL
        cls.Q = -267.48        # uL/s

        cls.M = diags([[0, cls.C, cls.L]], [0])
        B = bmat([[+1], [-1]])
        cls.D = bmat([[None, B], 
                      [-B.T, np.array([cls.R])]])
        cls.sys = SparseDySys(cls.M, cls.D,
                              lambda t: [0, cls.Q, 0])

    def test_equilibrium(self):
        '''the branch serial flow-rate absorbs the duty and the
        pressure-drop is all across the resistance

        '''

        sys = self.sys.constrain([0], [self.pin])
        soln = sys.reconstitute(sys.equilibrium())[0]
        p, q = soln[:2], soln[2]
        np.testing.assert_array_almost_equal(
            p,
            self.pin + np.array([0, self.Q * self.R]))
        np.testing.assert_array_almost_equal(q, -self.Q)
            

if __name__ == '__main__':
    main()
