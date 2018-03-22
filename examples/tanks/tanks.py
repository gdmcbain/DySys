#!/usr/bin/env python

"""Two-compartment mixing (Fulford et al 1997, p. 359)

Fulford, G., P. Forrester, & A. Jones (1997). *Modelling with
Differential and Difference Equations,* Volume 10 of *Australian
Mathematical Society Lecture Series*. Cambridge University Press

:author: gmcbain

:created: 2016-10-20

"""

from os.path import extsep

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame

from dysys import ScalarLinearDySys

from dysys.signal_flow_path_sys import SignalFlowPathSys


def main(name: str) -> None:

    ic = np.zeros(2)

    c = ic
    dt = 10
    endtime = 240

    sfpsys = SignalFlowPathSys([
        ScalarLinearDySys(1, 0.04, lambda *_: 0.01, 0.5),
        ScalarLinearDySys(1, 0.05, lambda _, __, ___, ____, y: y / 25, 0.5)])

    sfpeqm = sfpsys.equilibrium(ic)

    print('SignalFlowPathSysEquilibrium:', sfpeqm)
    eqm = np.array([0.25, 0.2])
    np.testing.assert_allclose(sfpeqm, eqm)
    ptrajectory = DataFrame({t: x for t, x, _ in
                             sfpsys.march_till(endtime, dt, ic)}).T

    exact = DataFrame({t: np.array([eqm[0] * (1 - np.exp(-t/25)),
                                    (1 - np.exp(-t/25)) -
                                    4/5 * (1 - np.exp(-t/20))])
                       for t in ptrajectory.index}).T
    np.testing.assert_allclose(ptrajectory, exact, atol=1e-2)

    fig, ax = plt.subplots(2, sharex=True)
    fig.suptitle('Concentrations approach the steady-state')
    for i, c in enumerate(ptrajectory):
        ptrajectory[c].plot(ax=ax[i], marker='v', linestyle='None',
                            label='SignalFlowPathSys')
        exact[c].plot(ax=ax[i], label='exact')
        ax[i].axhline(eqm[i], linestyle='--')
        ax[i].set_ylabel('c[{0}]'.format(i))
    ax[1].set_xlabel('time')
    ax[0].legend(loc=4)
    fig.savefig(extsep.join([name, 'png']))


if __name__ == '__main__':
    from os.path import basename, splitext
    from sys import argv
    main(splitext(basename(argv[0]))[0])
