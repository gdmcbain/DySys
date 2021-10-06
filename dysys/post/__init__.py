from toolz import partitionby


def segment(history, period):
    '''break a history into periods

    :param: iterable of events, defined by having their first term as
    time, monotonically nondecreasing

    :param: period, positive float

    :rtype: iterable of iterables

    '''

    # TODO gmcbain 2017-02-24: Reverse order of arguments.

    return partitionby(lambda ev: int(ev[0] / period), history)
