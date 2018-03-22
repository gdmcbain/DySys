#!/usr/bin/env python

"""Example 1 from Driver (1978, p. 215)

as a first example of numerical solution of a differential-difference
equation for DySys#32.

:author: G. D. McBain <gmcbain>

:created: 2018-02-15

"""

from itertools import dropwhile
from os.path import extsep
from typing import Any, List, Optional, Tuple

import attr
from toolz import compose, last

from matplotlib.pyplot import subplots
import numpy as np
from pandas import Series

from dysys import DySys


@attr.s(frozen=True)
class Driver(DySys):
    r = attr.ib(type=float)

    def step(self,
             t: float,
             h: float,
             y: List[Tuple[float, float]],
             d: Optional[Any]=None) -> List[Tuple[float, float]]:
        t1 = t + h
        y1 = list(dropwhile(lambda tx: tx[0] < t1 - self.r,
                            y))
        return y1 + [(t1, y1[-1][1] - h * y1[0][1])]


def main(name: str,
         h: float=0.1) -> None:

    r = np.pi / 2
    t = np.linspace(-r, 0)
    y = list(zip(t, np.sin(t)))
    print(y)
    driver = Driver(r)

    history = Series({t: compose(last, last)(y)
                      for t, y, _
                      in driver.march_till(r, h, y)})
    print(history)

    fig, ax = subplots()
    fig.suptitle("Driver's example 1 delay differential equation")
    history.plot(ax=ax, linestyle='None', marker='o', label='DySys')
    ax.plot(history.index, np.sin(history.index), label='exact')
    ax.legend()
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    fig.savefig(extsep.join([name, 'png']))


if __name__ == '__main__':

    from os.path import basename, splitext
    from sys import argv

    main(splitext(basename(argv[0]))[0])
