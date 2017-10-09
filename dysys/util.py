#!/usr/bin/env python

from toolz import compose, curry, flip


def autonomous(*funcs):
    '''return a function which ignores its first argument

    assumed to be time, which is appropriate to an autonomous system,

    and then applies the composition of funcs

    '''

    return compose(*funcs, lambda _, x, *args, **kwargs: x)
