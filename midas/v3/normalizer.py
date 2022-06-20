#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 16:19:43 2022

@author: Danilo Santos
Todo:
"""

from os import cpu_count

import numpy as np

from joblib import Parallel, delayed

from .stats import OnlineStats


class OnlineNormalizer(object):
    """
    Normalizer methods.

    Include z-score and MinMax normalizer.
    """

    __slots__ = ['_cfg_stats', '_vars', '_features', '_n_jobs',
                 '_verbose', '_last_learn']

    def __init__(self, cfg_stats={}, n_jobs=-1, verbose=0):
        self._cfg_stats = cfg_stats
        self._vars = {}
        self._features = []
        self._n_jobs = n_jobs
        self._verbose = verbose
        self._last_learn = {}

    def _get_n_jobs(self):
        """
        Util function.

        Util for to return get the number of CPUs in the system or
        the number choose from user.
        parameters: None
        """
        return (cpu_count() if self._n_jobs == -1 else self._n_jobs)

    def _parallel_learn_one(self, ik, Xi):
        """
        Util function.

        Function for parallelism in to learning.
        """
        if not(ik in self._features):
            self._features.append(ik)
            self._vars[ik] = OnlineStats(**self._cfg_stats)
        self._vars[ik].update(Xi)
        return None

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

    def learn_many(self, X, format="dict"):
        """
        Learn many data.

        learn many data.
        """
        if format == "dict":
            _ = Parallel(n_jobs=self._get_n_jobs(),
                         verbose=self._verbose, require='sharedmem')(
                delayed(self.learn_one)(Xi)
                for Xi in X
            )
        elif format == "table":
            pass
        return self

    def z_score(self, X=None):
        """
        Z-Score.

        (Xi - media) / desvio padrão.
        """
        if not(X is None):
            self.learn_one(X)

        def _func(ik, Xi):
            return (Xi - self._vars[ik].mean) / self._vars[ik].std

        zscore = Parallel(n_jobs=self._get_n_jobs(),
                          verbose=self._verbose, require='sharedmem')(
            delayed(_func)(ik, Xi) for ik, Xi in X.items())
        zscore = dict(zip(self._features, zscore))
        return zscore

    def min_max(self):
        """
        MinMax.

        (Xi - xmin) / (xmax - xmin)
        """
        pass

    def scale(self):
        pass

    def quantile(self):
        pass

    def percentil(self):
        pass

    def get_features(self):
        """
        Get Features fited.

        Return features fited from stream.
        """
        return self._features
