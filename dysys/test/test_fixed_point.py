#!/usr/bin/env python

'''
:author: gmcbain

:created: 2013-08-26

'''

from __future__ import absolute_import, division, print_function

from unittest import TestCase, main

import numpy as np
from scipy.sparse import spdiags

from dysys import newton


class TestNewton(TestCase):

    def test_sqrt(self, x=(2 * np.arange(1, 3),), decimals=5):

        '''r(y) = y**2 - x

        r(y + s) = (y + s)**2 - 2 = y**2 + 2 * y * s + O(s**2) - x

        ~= r(y) + J(y) * s

        J(y) = 2 * y

        '''

        def res(y):
            return y[0]**2 - x[0]

        def jac(y):
            return spdiags(2 * y[0], 0, *(len(x[0]),)*2).tocsc()

        np.testing.assert_array_almost_equal(
            newton(res, jac, x, tol=10**-(decimals/2))**2,
            x,
            decimals)


if __name__ == '__main__':
    main()
