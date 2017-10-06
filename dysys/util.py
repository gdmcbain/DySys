#!/usr/bin/env python

from toolz import compose, curry, flip


@curry
def const(x, *_, **__):
    '''like in Haskell 

    but taking and ignoring any further positional or keywords
    arguments.  Oddly missing from toolz.functoolz.

    See also funcy.constantly.

    '''
    
    return x


def autonomous(*funcs):
    '''return a function which ignores its first argument

    assumed to be time, which is appropriate to an autonomous system,

    and then applies the composition of funcs

    '''

    return compose(*funcs, flip(const))
