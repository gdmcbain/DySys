#!/usr/bin/env python

'''
:author: G. D. McBain <gmcbain>
:created: 2015-02-09

'''

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from scipy.integrate import ode

from dysys import DySys


class ODySys(DySys):

    def __init__(self, f, jac=None):
        self.f, self.jac = f, jac
        self.r = ode(f, jac)

    def step(self, t, h, x, d):
        '''estimate the next state'''
        self.r.set_initial_value(x, t)
        return self.r.integrate(self.r.t + h)
