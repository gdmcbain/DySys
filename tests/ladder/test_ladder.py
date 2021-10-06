from unittest import TestCase, main

import numpy as np
from scipy.sparse import coo_matrix, diags

from dysys import SparseDySys


class TestLadder(TestCase):

    """ Consider the three-cell ladder network with (8) nodes and [10]
    equal branches, with one corner node (0) vented and the one
    diagonally opposite (7) at a constant suction of -5 kPa.

    .|--(0)-[6]->-(4)
         |         |
        [0]       [3]
         |         |
         v         v
         |         |
        (1)-[7]->-(5)
         |         |
        [1]       [4]
         |         |
         v         v
         |         |
        (2)-[8]->-(6)
         |         |
        [2]       [5]
         |         |
         v         v
         |         |
        (3)-[9]->-(7)=<-5 kPa>-|.

    In the exact solution, nodes (4) and (1) have pressures -1 and -5
    / 3.

    """

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
        V = sys.reconstitute(sys.equilibrium())
        np.testing.assert_array_almost_equal(V[[4, 1]], [-1, -5 / 3])


if __name__ == '__main__':
    main()
