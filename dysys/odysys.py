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

    def __init__(self, f, jac=None, f_args=None, jac_args=None):

        '''Encapsulate scipy.integrate.ode for DySys

        :param f: 

        :param jac:

        '''
        
        self.f, self.jac = f, jac
        self._ode = ode(self.f, self.jac)
        self.set_f_params(*(f_args or []))
        self.set_jac_params(*(jac_args or []))

    def __getattr__(self, name):
        '''delegate to ode'''
        return getattr(self._ode, name)

    def handle_event(self, f, t, x, d):
        x, d = super(ODySys, self).handle_event(f, t, x, d)
        self.set_f_params(*d.get('f_args', []))
        self.set_jac_params(*d.get('jac_args', []))
        return x, d

    def step(self, t, h, x, d):
        '''estimate the next state'''
        self.set_initial_value(x, t)

        if h == 0:
            raise ZeroDivisionError
        
        xnext = self.integrate(self.t + h)
        if self.successful():
            return xnext
        else:
            raise ZeroDivisionError
