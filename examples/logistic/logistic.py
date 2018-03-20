#!/usr/bin/env python

"""Trying out TrivialSys (DySys#44).

:author: G. D. McBain <gmcbain>

:created: 2018-03-16

"""

from itertools import count
from os.path import extsep

import attr
from toolz import sliding_window, take, unique

from matplotlib.pyplot import subplots
import numpy as np

from dysys.trivial_sys import TrivialSys


def main(name: str, mu: float=2.) -> None:
    sys = TrivialSys()

    def logistic_map(x: float, mu: float=mu) -> float:
        return mu * x * (1 - x)

    State = attr.make_class('State',
                            {'x': attr.ib(type=float),
                             'n': attr.ib(type=int, default=0)},
                            frozen=True)

    sequence = unique(d.x for _, __, d in
                      sys.march(np.inf,
                                d=State(x=0.2),
                                events=((n,
                                         lambda _, __, x, d:
                                         (x, attr.evolve(d,
                                                         x=logistic_map(d.x),
                                                         n=d.n + 1)))
                                        for n in count())))

    x = list(take(7, sequence))
    print(x)
    xy = np.array(list(sliding_window(2, np.repeat(x, 2)[1:])))

    fig, ax = subplots()
    fig.suptitle(f'Evolution of the logistic map for µ = {mu}.'
                 f' The equilibrium value is {(mu-1)/mu}.')
    ax.plot(*np.split(xy, 2, 1), 'ro-')
    x = np.linspace(0, 1)
    ax.plot(x, logistic_map(x), 'g')
    ax.plot(x, x, 'k--')
    ax.set_xlabel('$x_n$')
    ax.set_ylabel('$x_{n+1}$')
    fig.savefig(extsep.join([name, 'png']))


if __name__ == '__main__':

    from sys import argv
    from os.path import basename, splitext

    main(splitext(basename(argv[0]))[0])
