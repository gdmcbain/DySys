#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''

:author: gmcbain

:created: 2015-02-09

'''

from __future__ import absolute_import, division, print_function

from functools import partial

import numpy as np
from scipy.integrate import ode
from scipy.optimize import root

from . import DySys


class ODySys(DySys):

    def __init__(self, f, jac=None, f_args=None, jac_args=None):

        '''Encapsulate scipy.integrate.ode for DySys

        :param f: function of time, state, and possibly other
        arguments, listed in f_args

        :param jac: optional function of time, state, and possibly
        other arguments, listed in jac_args

        :param f_args: optional list of additional positional
        arguments for f

        :param jac_args: optional list of additional positional
        arguments for jac

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
            raise RuntimeError

    def equilibrium(self, y0, d=None, **kwargs):
        '''return a steady-state solution

        :param y0: one-dimensional numpy.ndarray, initial guess

        :param d: dict, discrete dynamical variables, currently
        ignored

        Further keyword-arguments passed on to scipy.optimize.root.

        '''

        return root(partial(self.f, np.inf),
                    y0,
                    self.f_params,
                    jac=lambda y: self.jac(np.inf, y, *self.jac_params),
                    **kwargs).x
