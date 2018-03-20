#!/usr/bin/env python

'''A path of dependent dynamical systems.

This is a first step towards more general networks of interdependent
dynamical systems.  The approach is inspired by Mason (1953, §II).

In graph theory (Gondran & Minoux 1984, p. 13), 'a path is a chain all
of whose arcs are directed in the same way'.  Further, 'an elementary
path is a path that does not meet the same vertex twice', which is
really what's addressed here.

Really it's a list of systems, with the output of each becoming the
input of the next; the list is the basic Pythonic data structure that
most of the methods of this class work with.

:author: gmcbain

:created: 2016-11-14

'''

from typing import Any, Sequence

import numpy as np

from .dysys import DySys
from .util import autonomous


class SignalFlowPathSys(DySys):

    def __init__(self,
                 systems: Sequence[DySys],
                 functions: Optional[Sequence[Callable[..., Any]]]=None,
                 **kwargs):
        '''construct a SignalFlowPathSys

        :param systems: sequence of DySys

        :param functions: sequence of functions, one shorter than systems,
        being the mappings of (time, state) between systems along the
        list

        '''

        super(SignalFlowPathSys, self).__init__(self, **kwargs)
        self.systems = systems
        self.functions = (functions if functions is not None
                          or ([autonomous()] * (len(self) - 1)))

    def __len__(self):
        return len(self.systems)

    @property
    def zero(self):
        return [s.zero for s in self.systems]

    def step(self,
             t: float,
             h: float,
             x: Sequence[Any],
             d: Any,
             inputs: Optional[Any]=None) -> List[Any]:
        '''estimate the state after a step in time

        '''

        # TODO gmcbain 2016-11-21: Could this be expressed with
        # itertools.accumulate?

        # TOOD gmcbain 2017-10-03: …or toolz.itertoolz.accumulate?

        xnew = [self.systems[0].step(t, h, x[0], d, inputs)]
        for i in range(1, len(self)):
            xnew.append(self.systems[i].step(t, h, x[i], d,
                                             tuple(map(self.functions[i-1],
                                                       (t, t + h),
                                                       (x[i-1], xnew[-1])))))

        return xnew

    def equilibrium(self, x=None, d=None, **kwargs):
        '''return an eventual steady-state solution

        :param x: list of initial guess, optional default self.zero

        :param d: dict, discrete dynamical variables; optional

        Additional keyword arguments are passed on to the equilibrium
        methods of the subsystems.

        '''

        x = x or self.zero

        xoo = [self.systems[0].equilibrium(x[0], d, **kwargs)]
        for i in range(1, len(self)):
            xoo.append(self.systems[i].equilibrium(
                x[i], d, map(self.functions[i-1],
                             (0, np.inf), (x[i-1], xoo[-1])),
                **kwargs))

        return xoo
