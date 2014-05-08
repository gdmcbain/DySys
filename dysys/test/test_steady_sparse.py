#!/usr/bin/env python

'''Test the equilibrium method of SparseDySys.

Derived from msmdir.005291.

:author: G. D. McBain <gmcbain>

:created: 2014-05-08

:keywords: hydraulics, analytic circuit theory

'''

from __future__ import absolute_import, division, print_function

from unittest import TestCase, main

import numpy as np
from scipy.sparse import coo_matrix, diags

from dysys import SparseDySys

class TestLadder(TestCase):

    @classmethod
    def setUpClass(cls):
        nodes = 8
        branches = 10
        
        G = -coo_matrix((np.ones(branches),
                         ([0, 1, 2, 4, 5, 6, 0, 1, 2, 3],
                          [1, 2, 3, 5, 6, 7, 4, 5, 6, 7])),
                        [nodes] * 2)
        G = G + G.T
        G = G - diags(np.array(G.sum(0)), [0])
        cls.sys = SparseDySys(None, G)

    def test(self):
        sys = self.sys.constrain([0, -1], [0.0, -5.0])
        V = sys.reconstitute(sys.equilibrium())[0]
        np.testing.assert_array_almost_equal(V[[4, 1]], [-1, -5 / 3])

if __name__ == '__main__':
    main()

        
