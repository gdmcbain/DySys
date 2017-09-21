#!/usr/bin/env python

'''
:author: G. D. McBain <gmcbain>
:created: 2013-02-07

'''

from __future__ import absolute_import, division, print_function

from unicodedata import lookup

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame, Panel
from scipy.sparse import coo_matrix, csc_matrix, bmat, eye

from dysys import SparseDySys, HilberHughesTaylor
from dysys.newmark import (trapezoidal, fox_goodwin, linear_acceleration,
                           central_difference)


def with_method(m, h, ic):
    return DataFrame({t: {'x': x[0], 'v': v[0]} for t, (x, v), _ in
                      m.march_till(0.2, h, ic)}).T


def exact(L, R, C, Q, t):
    sigma = np.roots([L * C, R * C, 1])
    Delta_sigma = np.diff(sigma)[0]
    return (np.real_if_close(((np.exp(sigma[0] * t) - 1) * sigma[1] -
                              (np.exp(sigma[1] * t) - 1) * sigma[0]) * Q /
                             Delta_sigma),
            np.real_if_close(((np.exp(sigma[0] * t) - 1) -
                              (np.exp(sigma[1] * t) - 1)) /
                             (sigma[1] - sigma[0]) * Q / L / C))


def main(name: str) -> None:
    '''Reproduce the lumped RLC branch with step duty example

    from memjet Swiki "Simulating transient flow with
    Millihydraulics", derived there as a first-order system with two
    degrees of freedom (outlet pressure p and serial branch flowrate
    q) but here recast in second-order form using just q as the
    dependent variable.

    '''

    inertance = 0.073557
    resistance = 4.6809
    compliance = 0.0024519
    duty = -267.48

    M = csc_matrix([inertance])
    C = coo_matrix([resistance])
    K = coo_matrix([1./compliance])

    def f(*_):
        return np.array([-duty/compliance])

    h = 5e-3

    ic = [np.zeros(1), np.zeros(1)]
    trajectories = {'trapezoidal': with_method(trapezoidal(M, K, C, f), h, ic),
                    'Fox-Goodwin': with_method(fox_goodwin(M, K, C, f), h, ic),
                    'linear acceleration':
                    with_method(linear_acceleration(M, K, C, f), h, ic),
                    'central difference':
                    with_method(central_difference(M, K, C, f), h, ic)}
    trajectories.update({u'{0}={1:.1f}'.format(
        lookup('GREEK SMALL LETTER ALPHA'), alpha):
                         with_method(HilberHughesTaylor(M, K, C, f, alpha,
                                                        definite=True),
                                     h, ic)
                         for alpha in [-.15]})
    sys = SparseDySys(bmat([[eye(1), None],
                            [C,      M]]).tocsc(),
                      bmat([[None, -eye(1)],
                            [K,    None]]),
                      lambda *_: np.array([0, -duty/compliance]))
    trajectories['backward Euler'] = DataFrame(
        {t: {'x': x[0], 'v': x[1]} for t, x, _ in
         sys.march_till(0.2, h, sys.zero)}).T

    first_order = trapezoidal(M, K, C, f).to_sparse_dysys(theta=0.5)
    trajectories['first order'] = DataFrame(
        {t: dict(zip('xv', x)) for t, x, _ in
         first_order.march_till(0.2, h, first_order.zero)}).T

    data = Panel(trajectories)

    fig, ax = plt.subplots(2)
    t = np.linspace(0, max(data.major_axis), 999)
    for order, variable in [(0, 'x'), (1, 'v')]:
        for label, series in data.minor_xs(variable).iteritems():
            ax[order].plot(series,
                           marker='o', linestyle='None', label=label)
        ax[order].plot(t, exact(inertance, resistance, compliance, duty,
                                t)[order], label='exact')

    fig.suptitle('step-duty for lumped print-head')
    ax[0].legend(loc=4)
    ax[0].set_ylabel('flowrate, $q$ / [µL/s]')
    ax[1].set_ylabel('rate of change, $\dot q$ / [µL/s²]')
    ax[-1].set_xlabel('time, $t$ / s')
    fig.savefig(name + '.png')


if __name__ == '__main__':

    from os.path import basename, splitext
    from sys import argv

    main(splitext(basename(argv[0]))[0])
