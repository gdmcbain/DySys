#!/usr/bin/python
# -*- coding: utf-8 -*-

'''A list of uncoupled dynamical systems

taking synchronized approximate discrete steps in continuous time

:author: G. D. McBain <gmcbain>

:created: 2016-07-21

'''

from __future__ import absolute_import, division, print_function

from .dysys import DySys


class UncoupledDySys(DySys):

    def __init__(self, systems):
        'initialize with a list of DySys'

        self.systems = systems

    def step(self, t, h, yy, dd):
        '''estimate the next states using appropriate methods

        :param t: float, time

        :param h: float > 0, time-step

        :param yy: list of initial conditions, corresponding to
        self.systems

        :param dd: list of discrete states, corresponding to
        self.systems

        '''

        return [s._step(t, h, y, d) for s, y, d in zip(self.systems, yy, dd)]
