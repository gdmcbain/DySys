#!/usr/bin/env python

'''
:author: gmcbain

:created: 2013-08-26

'''

from unittest import (TestCase, main)

import numpy as np
from scipy.sparse import spdiags

from dysys import newton


class TestNewton(TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    def test_sqrt(self, x=(2 * np.arange(1, 3),), decimals=5):

        def res(y):
            return y[0] ** 2 - x[0]

        def jac(y):
            return spdiags(2 * y[0], 0, *(len(x[0]),)*2)
            
        print 'answer: ', newton(res, jac, x)

        np.testing.assert_array_almost_equal(
            newton(res, jac, x, tol=10 ** -decimals),
            np.sqrt(x),
            decimals)


if __name__ == '__main__':
    main()
