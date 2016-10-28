#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''A class for scalar linear 'descriptor' systems.

Useful for keeping simple demonstrations simple, rather than having to
set up sparse matrices of shape (1, 1).

:author: gmcbain

:created: 2016-10-26

'''

from __future__ import absolute_import, division, print_function

from .linear_dysys import LinearDySys

from numpy import inf


class ScalarLinearDySys(LinearDySys):
    '''a LinearDySys with scalar mass and damping

    and a force which is a function of time

    '''

    def __len__(self):
        return 1

    def driven_step(self, t, tnew, y, f, fnew, theta=0.5):
        '''return the state at tnew given state y at t

        :param t: float, time

        :param h: float > 0, time-step

        :param y: float, state

        :param f: float, RHS forcing term at t

        :param fnew: float, RHS forcing term at t + h

        :param theta: float, in (0, 1), parameter of theta-method;
        default 0.5 for Crank-Nicolson, 1 gives backward Euler

        :rtype: float

        '''

        h = tnew - t
        return ((theta * fnew + (1 - theta) * f +
                 (self.M / h - (1 - theta) * self.D) * y) /
                (self.M / h + theta * self.D))

    def drive(self, x, driver, theta=0.5):
        '''generate the evolution of the system in time

        :param x: float, initial condition

        :param driver: sequence of (float, float) pairs, being (time,
        force); e.g. as generated by the iteritems method of a
        pandas.Series

        :param theta: float, passed on to driven_step

        :rtype: sequence of pairs (float, float), being (time, state)

        '''

        told, fold = next(driver)
        yield told, x

        for t, f in driver:
            x = self.driven_step(told, t, x, fold, f, theta)
            told, fold = t, f
            yield t, x

    def forced_step(self, t, x, h, d, f, theta=0.5):
        '''return the state at t + h given x at t

        :param t: float, time

        :param x: float, initial condition at time t

        :param h: float > 0, time-step

        :param d: dict, discrete dynamical variables

        :param f: (float -> float), time -> rhs

        :param theta: float, in (0, 1), being the parameter of the
        theta method; default 0.5 for Crank-Nicolson, use 1.0 for
        backward Euler

        '''
        
        if h == 0:
            raise ZeroDivisionError
        return (((theta * f(t + h) + (1 - theta) * f(t)) +
                 (self.M / h - (1 - theta) * self.D) * x) /
                (self.M / h + theta * self.D))

    def forced_march(self, h, x, forcing, d=None, theta=0.5):
        '''generate evolution from x due to forcing

        :param h: float > 0, time-step

        :param x: float, initial condition

        :param forcing: (float -> float), (time -> rhs)

        :param d: dict, discrete dynamical variables

        :param theta: float, in (0, 1), being the parameter of the
        theta method; default 0.5 for Crank-Nicolson, use 1.0 for
        backward Euler

        '''        

        t, d = 0., d or {}

        while True:
            yield t, x, d
            t, x = t + h, self.forced_step(t, x, h, d, forcing, theta)

    def equilibrium(self, y0=None):
        '''return eventual steady state

        :param y0: initial guess, ignored

        '''

        return self.f(inf) / self.D

    def step(self, t, h, x, d=None):
        '''estimate the next state using theta method

        :param t: float, time

        :param h: float > 0, time-step

        :param x: float, initial condition at time t

        :param d: dict, discrete dynamical variables, optional
        (default empty)

        If 'master' in d, it should be a dict containing a DySys in
        'system', on the evolution of which the present step depends.
        The right-hand side self.f(t) is replaced by a generalized
        trapezoidal sum of the values of
        d['master']['system']['state'] and where that steps to, both
        mapped with d['master']['f'].

        '''

        try:
            yold = d['master'].pop('state')
            ynew = d['master']['state'] = d['master']['system'].step(
                t, h, yold, d['master'].get('d'))
            fold, fnew = map(d['master']['f'], [yold, ynew])
        except (TypeError, KeyError):
            fold, fnew = map(self.f, [t, t + h])
            
        return ((self.theta * fnew + (1 - self.theta) * fold +
                 (self.M / h - (1 - self.theta) * self.D) * x) /
                (self.M / h + self.theta * self.D))
