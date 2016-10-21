#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''

:author: gmcbain

:created: 2015-02-09

'''

from __future__ import absolute_import, division, print_function

from scipy.integrate import ode

from dysys import DySys


class ODySys(DySys):

    def __init__(self, f, jac=None):
        self.f, self.jac = f, jac
        self._ode = ode(f, jac)

    def __getattr__(self, name):
        '''delegate to ode'''
        return getattr(self._ode, name)

    def step(self, t, h, x, d):
        '''estimate the next state'''
        self.set_initial_value(x, t)
        return self.integrate(self.t + h)
