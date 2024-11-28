#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 16:34:55 2022

@author: Danilo Santos

TODO: Implementar o código para um stream gerado automaticamente
"""

import numpy as np

import pandas as pd

from collections.abc import Iterable

import copy as cp


class SlidingWindowErrors(object):
    """
    Sliding window messages errors.

    The concentration messages errors.
    """

    __slots__ = ['E0x000', 'E0x001', 'E0x002', 'E0x003', 'E0x004', 'E0x005',
                 'E0x006', 'E0x007', 'E0x008', 'E0x009']

    E0x000 = "Unrecognized value for parameter 'method_return_window'."
    E0x001 = 'Expected size with one integer type.'
    E0x002 = 'Size not have less than 1.'
    E0X003 = 'Value for step isnot negative or less than 1.'
    E0x004 = 'Value assigned for stream isnot an interable.'
    E0x005 = 'Value assigned for target isnot an interable.'
    E0x006 = 'Unknow type of stream'
    E0x007 = 'Impossible slice stream object.'
    E0x008 = 'Underflow data stream.'
    E0x009 = 'Overflow data stream.'
    E0x010 = 'Type stream must be equal to "static" or "dinamyc"'


class SlidingWindow(object):
    """
    The sliding window implementation.

    An alternative praticle for read stream.
    """

    __slots__ = ['_window', '_stream', 'stream_target', '_step', '_copy',
                 'method_return_window']

    def __init__(self, size, stream, stream_target=None, step=1,
                 method_return_window="deep", type_stream="static"):
        self._window = [0, size+step]
        self._stream = stream
        self._stream_target = stream_target
        self._step = step
        self._copy = None
        self._tstream = str(type_stream).lower()

        method_return_window = str(method_return_window).lower()
        if method_return_window == "deep":
            self._copy = cp.deepcopy
        elif method_return_window == "shallow":
            self._copy = cp.copy
        else:
            raise ValueError(SlidingWindowErrors.E0x000)

        if not(isinstance(size, int)):
            raise RuntimeError(SlidingWindowErrors.E0x001)
        if size < 1:
            raise ValueError(SlidingWindowErrors.E0x002)

        if step < 1:
            raise ValueError(SlidingWindowErrors.E0x003)
        if not(isinstance(stream, Iterable)):
            raise ValueError(SlidingWindowErrors.E0x004)

        validate = all(not(stream_target is None),
                       not(isinstance(stream_target, Iterable)))
        if validate:
            raise ValueError(SlidingWindowErrors.E0x005)

        if not(self._tstream in ['static', 'dynamic']):
            raise ValueError(SlidingWindowErrors.E0x010)

    def _get_size_stream(self):
        """
        Get size stream (return int).

        Return the size of stream according to type of stream.
        """
        if isinstance(self.stream, list):
            return len(self.stream)
        if isinstance(self.stream, np.ndarray):
            return self.stream.shape[0]
        if isinstance(self.stream, pd.core.frame.DataFrame):
            return self.stream.shape[0]
        if isinstance(self.stream, pd.core.series.Series):
            return self.stream.shape[0]
        raise ValueError(SlidingWindowErrors.E0x006)

    def _is_underflow(self):
        """
        Is underflow (return bool).

        This underflow limit of stream.
        """
        return (self._window[0] < 0)

    def _is_overflow(self):
        """
        Is overflow (return bool).

        This overflow limit of stream.
        """
        return (self._window[0] > self._get_size_stream())

    def _slice_window(self, start, stop):
        TStream = type(self._stream)
        if isinstance(TStream, list) or isinstance(TStream, np.ndarray):
            return self._copy(self._stream[start:stop])
        if isinstance(TStream, pd.core.series.Series):
            return self._copy(self._stream[start:stop])
        if isinstance(TStream, pd.core.frame.DataFrame):
            return self._copy(self._stream.iloc[start:stop, :])
        raise RuntimeError(SlidingWindowErrors.E0x007)

    def _increment(self):
        pass

    def _decrement(self):
        pass

    def left_shift(self, size=0):
        """
        Left shift (return window object).

        Drift window in the left.
        """
        while self._is_underflow() is False:
            yield self._slice_window(self._window[0], self._window[1])

        self._window[0] -= self._step + size
        self._window[1] -= self._step + size

    def right_shift(self, size=0):
        """
        Left right (return window object).

        Drift window in the right.
        """
        while self._is_overflow() is False:
            yield self._slice_window(self._window[0], self._window[1])

        self._window[0] += self._step + size
        self._window[1] += self._step + size

    def get_current_window(self):
        """
        Get Current Window (return window object).

        Return the current window.
        """
        if self._is_underflow():
            raise RuntimeError(SlidingWindowErrors.E0x008)
        if self._is_overflow():
            raise RuntimeError(SlidingWindowErrors.E0x009)
        return self._slice_window(self._window[0], self._window[1])

    def get_current_position_window(self):
        """
        Get current position window (return tuple).

        Tuple with current position of window.
        """
        return tuple(self._window)
