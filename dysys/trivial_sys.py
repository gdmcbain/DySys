#!/usr/bin/env python
"""trivial dynamical system

having no continuous degrees of freedom but arbitrary discrete state

Developed under DySys#44

:author: G. D. McBain <gmcbain>

:created: 2018-03-16

"""

from typing import Any, Optional

import numpy as np

from dysys import DySys


class TrivialSys(DySys):

    def __init__(self):
        pass

    def __len__(self):
        return 0

    def step(self,
             t: float,
             h: float,
             y: np.ndarray,
             d: Optional[Any]=None) -> np.ndarray:
        return self.zero
