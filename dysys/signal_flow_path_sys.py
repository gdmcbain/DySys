#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''A path of dependent dynamical systems.

This is a first step towards more general networks of interdependent
dynamical systems.  The approach is inspired by Mason (1953, §II).

In graph theory (Gondran & Minoux 1984, p. 13), 'a path is a chain all
of whose arcs are directed in the same way'.  Further, 'an elementary
path is a path that does not meet the same vertex twice', which is
really what's addressed here.

:author: gmcbain

:created: 2016-11-14

'''

from __future__ import absolute_import, division, print_function

from dysys import DySys


class SignalFlowPathSys(DySys):

    def __init__(self, systems, functions=None, **kwargs):
        '''construct a SignalFlowPathSys

        :param systems: list of DySys

        :param functions: list of functions to map the output of one
        DySys into the input to the next; this should be one shorter
        that the previous; optional (default: just pass on output as
        input)

        '''

        super(SignalFlowPathSys, self).__init__(self, **kwargs)
        self.systems = systems
        self.functions = functions

        # TODO gmcbain 2016-11-15: Redesign DySys so that simulating a
        # slave system that depends on a master doesn't work by
        # continually modifying attributes of the slave.  This
        # probably means making step et al variadic. Consider
        # scipy.integrate.ode, the constructor of which takes
        # functions for the RHS and its Jacobian, both having two
        # compulsory positional arguments and then an optional list of
        # further arguments which are set with methods
        # set_{f,jac}_params.  That's still 'setting' something, so
        # still not quite 'pure' in the sense of functional
        # programming.

        for s, p in zip(self.systems[1:], self.systems[:-1]):
            s.predecessor = {'system': p}

    def __len__(self):
        return len(self.systems)

    def step(self, t, h, x, d):
        '''estimate the state after a step in time

        :param t: float, time

        :param h: float > 0, time-step

        :param x: state

        :param d: dict, discrete dynamical variables

        '''

        xnew = [self.systems[0].step(t, h, x[0], d)]
        for i in range(1, len(self)):
            self.systems[i].predecessor.update(
                zip(['fold', 'fnew'],
                    map(self.functions[i-1], [x[i-1], xnew[i-1]])))
            xnew.append(self.systems[i].step(t, h, x[i], d))

        return xnew

    def equilibrium(self, x, d=None, **kwargs):
        '''return an eventual steady-state solution

        :param x: list of initial guess

        :param d: dict, discrete dynamical variables

        '''

        xoo = [self.systems[0].equilibrium(x[0], d, **kwargs)]
        for i in range(1, len(self)):
            xoo.append(self.systems[i].equilibrium(x[i], d, (x[i-1], xoo[-1]),
                                                   **kwargs))

        return xoo
