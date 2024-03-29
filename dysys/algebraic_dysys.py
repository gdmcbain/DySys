"""A class for algebraic (pseudo) dynamical systems

which just react instantaneously to time and inputs.

"""

from typing import Dict, Union, Callable, Tuple

import numpy as np

from .dysys import DySys


class AlgebraicDySys(DySys):

    def __init__(self,
                 zero: Union[float, np.ndarray],
                 f: Callable[[DySys,
                              float,
                              Union[float, np.ndarray],
                              Dict,
                              Tuple[Union[float, np.ndarray],
                                    Union[float, np.ndarray]]],
                             Union[float, np.ndarray]]):

        self.f = f              # f(self, t, x, d, inputs)

        # KLUDGE gmcbain 2017-09-28: To avoid "AttributeError: can't
        # set ttribute" [sic]

        self._zero = zero

    def __len__(self):
        return 1 if np.isscalar(self._zero) else len(self._zero)

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
        'return the eventual steady state'
        return self.f(self, np.inf,
                      self.zero if x is None else x,
                      d, **kwargs)
