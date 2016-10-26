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


class ScalarLinearDySys(LinearDySys):

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
        force)

        :param theta: float, passed on to driven_step

        :rtype: sequence of pairs (float, float), being (time, state)

        '''

        told, fold = next(driver)
        yield told, x

        for t, f in driver:
            x = self.driven_step(told, t, x, fold, f, theta)
            told, fold = t, f
            yield t, x
