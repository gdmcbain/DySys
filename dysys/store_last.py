#!/usr/bin/env python

'''
:author: G. D. McBain <gmcbain>, based on code read 2013-06-11
http://www.dabeaz.com/generators-uk/storelast.py, Copyright (C) 2008
David M. Beazley with no licensing information

:created: 2013-06-13

'''

from __future__ import absolute_import, division, print_function


class StoreLast(object):
    'wrap an iterable to remember the last item generated'

    def __init__(self, source):
        self.source = source

    def __next__(self):
        self.last = next(self.source)
        return self.last

    next = __next__             # backward-compatibility with python2

    def __iter__(self):
        return self
