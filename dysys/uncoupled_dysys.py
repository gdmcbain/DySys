#!/usr/bin/python

"""A list of uncoupled dynamical systems

taking synchronized approximate discrete steps in continuous time

:author: G. D. McBain <gmcbain>

:created: 2016-07-21

"""

from typing import Any, List, Sequence

from dysys import DySys


class UncoupledDySys(DySys):

    def __init__(self, systems: Sequence[DySys]):
        """initialize with a list of DySys"""

        self.systems = systems

    def step(self,
             t: float,
             h: float,
             yy: Sequence[Any],
             dd: Sequence[Any]) -> List[Any]:
        """estimate the next states using appropriate methods

        :param t: float, time

        :param h: float > 0, time-step

        :param yy: sequence of initial conditions, corresponding to
        self.systems

        :param dd: sequence of discrete states, corresponding to
        self.systems

        """

        return [s._step(t, h, y, d) for s, y, d in zip(self.systems, yy, dd)]
