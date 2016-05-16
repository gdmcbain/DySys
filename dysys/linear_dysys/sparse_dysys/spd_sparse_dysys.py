#!/usr/bin/env python

'''Symmetric positive-definite sparse dynamical system

:author: G. D. McBain <gmcbain>

:created: 2016-05-16

'''

from __future__ import absolute_import, division, print_function

from sksparse.cholmod import cholesky

from dysys import SparseDySys

class SPDSparseDySys(SparseDySys):

    def step(self, t, h, x, d):
        '''estimate the next state using theta method

        memoizing the Cholesky decomposition for fast time-stepping


        '''

        if not hasattr(self, '_memo') or h != self._memo['h']:
            M = self.M / h - (1 - self.theta) * self.D
            M1 = M + self.D

            self._memo = {'h': h,
                          'M': M,
                          'solve': cholesky(M + self.D)}

        b = self._memo['M'].dot(x)
        if self.f is not None:
            b += (self.theta * self.f(t + h, d) +
                  (1 - self.theta) * self.f(t, d))

        return self._memo['solve'](b)
