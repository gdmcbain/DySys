#!/usr/bin/env python

'''dysys.post: postprocessing utilities for DySys

:author: G. D. McBain <gmcbain>

:created: 2013-09-04

'''

import itertools as it


def segment(history, period):
    '''break a history into periods

    :param: iterable of events, defined by having their first term as
    time, monotonically nondecreasing

    :param: period, positive float

    :rtype: iterable of iterables

    '''

    for _, group in it.groupby(history, lambda ev: int(ev[0] / period)):
        yield group
