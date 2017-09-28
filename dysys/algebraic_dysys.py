#!/usr/bin/env python

'''A class for algebraic (pseudo) dynamical systems

which just react instantaneously to time and inputs.

:author: G. D. McBain <gmcbain>

:created: 2017-09-26

'''

from typing import Dict

import numpy as np

from .dysys import DySys


class AlgebraicDySys(DySys):

    def __init__(self, zero, f):

        self.f = f              # f(self, t, x, d, inputs)

        # KLUDGE gmcbain 2017-09-28: To avoid "AttributeError: can't
        # set ttribute" [sic]
        
        self._zero = zero       

    @property
    def zero(self):
        return self._zero

    def step(self,
             t: float,
             h: float,
             x,
             d: Dict,
             inputs=None):
        return self.f(self, t + h, x, d,
                      inputs[1] if inputs else None)

    def equilibrium(self, x=None, d=None, **kwargs):
        return self.f(self, np.inf, x or self.zero, d, **kwargs)
