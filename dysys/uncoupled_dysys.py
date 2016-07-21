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

    # TODO gmcbain 2016-07-21: Implement, maybe as a list of DySys.

    def __init__(self, *args, **kwargs):
        return NotImplemented

    def step(self, *args, **kwargs):
        return NotImplemented
