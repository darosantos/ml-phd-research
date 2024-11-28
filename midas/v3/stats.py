#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 17:51:42 2022

@author: Danilo Santos

Todo:
    from functools import lru_cache
"""

from os import cpu_count

from joblib import Parallel, delayed

from collections.abc import Iterable
from collections import Counter

import numbers

import numpy as np


class OnlineStats(object):
    """
    On-line statistics package.

    Implement the standar deviation, media and variance for stream.
    This soluction is based in the methology from
    https://stackoverflow.com/questions/5543651/computing-standard-deviation-in-a-stream
    """

    __slots__ = ['_ddof', '_n_observations', '_mean', '_M2', '_delta',
                 '_threshold']

    def __init__(self, X=None, ddof=0, n_observations=0,
                 mean=0.0, threshold=None):
        self._ddof = ddof
        self._n_observations = n_observations
        self._mean = mean
        self._M2 = 0.0
        self._delta = 0.0
        self._threshold = threshold

        if not(X is None):
            if isinstance(X, Iterable):
                _ = [self.update(Xi) for Xi in X]
            elif isinstance(X, numbers.Number):
                _ = self.update(X)
            else:
                raise ValueError('X not is valid value')

        if n_observations < 0:
            raise RuntimeError('N observations cannot go below 0')

        if not(type(n_observations) == int):
            raise ValueError('N observations cannot unknow type')

        if not(self._threshold is None):
            if self._threshold <= 0:
                raise ValueError('Threshold cannot negative value')
            if not(type(self._threshold) == int):
                raise ValueError('Threshold cannot unknow type')

    @property
    def variance(self):
        """
        Return variance.

        Calc and return the variance from current values.
        """
        return self._M2 / (self._n_observations - self._ddof)

    @property
    def std(self):
        """
        Return standard deviation.

        Calc and return the standard deviation from current value.
        """
        return np.sqrt(self.variance)

    @property
    def mean(self):
        """
        Return mean.

        Return mean stored from current value.
        """
        return self._mean

    @property
    def coefficient_variation(self):
        """
        Return the coefficient of variation.

        Calc and return coefficient of variation from current value.
        """
        return (self.mean / abs(self.std))

    def _is_threshold(self):
        if not(self._threshold is None):
            if self._threshold < self._n_observations:
                return True
        return False

    def update(self, X, w=1):
        """
        Update statistics.

        Update statistics for input X.
        """
        if not(isinstance(X, numbers.Number)):
            raise ValueError('Impossible update statistics.')
        if self._is_threshold():
            self.reset()
        self._n_observations += w
        self._delta = X - self._mean
        self._mean += self._delta / self._n_observations
        self._M2 += self._delta * (X - self._mean)

        return self

    def update_many(self, X):
        """
        Update statistics.

        Update many value for statitics.
        """
        if not(isinstance(X, Iterable)):
            raise ValueError('Expected many values to update')

        _ = [self.update(Xi) for Xi in X]
        return self

    def get(self, statistics="mean"):
        """
        Return one of statistics implemented.

        Return the statistics choose from parameter.
        """
        if statistics == 'std':
            return self.std
        if statistics == 'var':
            return self.variance
        return self.mean

    def reset(self):
        """
        Reset statistics for begin.

        Return the initial state class.
        """
        self._n_observations = 0
        self._mean = 0.0
        self._M2 = 0.0
        self._delta = 0.0

        return self


class OnlineEntropy(object):
    """
    Complment of the on-line statistics package.

    Implement the calcu of entropy for instance Xs
    """

    __slots__ = ['_features', '_vars', '_threshold', '_n_jobs', '_verbose']

    def __init__(self, threshold=None, n_jobs=-1, verbose=0):
        self._features = []
        self._vars = {}
        self._threshold = threshold
        self._n_jobs = n_jobs
        self._verbose = verbose

        # validar _threshold

    def _get_n_jobs(self):
        """
        Util function.

        Util for to return get the number of CPUs in the system or
        the number choose from user.
        parameters: None
        """
        return (cpu_count() if self._n_jobs == -1 else self._n_jobs)

    def _parallel_del_counter_negative(self, counter, key):
        """
        Delete key with negative value from counter.

        This function is utility called from parallel execution.
        """
        if counter[key] <= 0:
            del counter[key]

    def _util_del_counter_negative(self, counter):
        """
        Del key from counter with negative value.

        Delete the index negative from counter.
        """
        _ = Parallel(n_jobs=self._get_n_jobs(),
                     verbose=self._verbose, require='sharedmem')(
            delayed(self._parallel_del_counter_negative)(counter, ck)
            for ck in counter.copy()
        )

    def _util_del_counters_negative(self, list_counters):
        """
        Del counters index negative.

        Delete the index negative from counter within counters
        """
        _ = Parallel(n_jobs=self._get_n_jobs(),
                     verbose=self._verbose, require='sharedmem')(
            delayed(self._util_del_counter_negative)(ci)
            for ci in list_counters
        )

    def _parallel_counters_subtract(self, counter, del_counter):
        """
        Delete values from counter.

        Delete values from couter with verification additional.
        """
        valid = {key: del_counter[key]
                 for key in del_counter.keys() if key in counter.keys()}
        counter.subtract(valid)

    def _util_counter_subtract(self, list_counters, value):
        """
        Counter subtract.

        Subtract value dict in list of counters.
        """
        _ = Parallel(n_jobs=self._get_n_jobs(),
                     verbose=self._verbose, require='sharedmem')(
            delayed(self._parallel_counters_subtract)(ci, value)
            for ci in list_counters
        )

    def _util_create_counter(self, feature, value):
        """
        Create the couter for feature.

        Create the couter for feature.
        """
        self._features.append(feature)
        c = Counter([value])
        self._vars[feature] = c
        if not(self._threshold is None):
            self._vars[feature] = []
            self._vars[feature].append(c)

    def _parallel_learn_one(self, ik, Xi):
        """
        Util function.

        Function for parallelism in to learning.
        """
        if not(ik in self._features):
            self._util_create_counter(ik, Xi)
        elif self._threshold is None:
            self._vars[ik].update([Xi])
        else:
            n_counter = len(self._vars[ik])
            if n_counter >= self._threshold:
                p_counter = self._vars[ik].pop(0)
                self._util_counter_subtract(self._vars[ik], dict(p_counter))
                self._util_del_counters_negative(self._vars[ik])
                del p_counter
            current_counter = self._vars[ik][-1]
            new_counter = Counter(dict(current_counter))
            new_counter.update([Xi])
            self._vars[ik].append(new_counter)

    def learn_one(self, X):
        """
        Learn one element.

        X is one dict.
        """
        if not(isinstance(X, dict)):
            raise Exception('X cannot different of dict')

        _ = Parallel(n_jobs=self._get_n_jobs(),
                     verbose=self._verbose, require='sharedmem')(
            delayed(self._parallel_learn_one)(ik, Xi)
            for ik, Xi in X.items()
        )
        return self

    def learn_many(self, X):
        pass

    def getEntropyShannon(self, X):
        """
        entropia de Shannon é uma medida de incerteza
        """
        pass

    def getNormalizedSpecificEntropy(self, X):
        """
        Normalized specific entropy.

        Hn = (H2 * log(2)) / log(n)
        """
        pass

    def getTotalNormalizedExtensiveEntropy(self, X):
        """
        total normalized extensive entropy.

        Sn = (H2N * log(2)) / log(n)
        """
        pass

    def normalized(self, X):
        if normalized:
           if log(n) > 0:
               normalized_ent = entropy / log(n, 2)
