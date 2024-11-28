#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fou Nov 15 11:16:35 2023

@author: Danilo Santos
"""
import gc

from threading import Lock

from collections import deque

from builtins import RuntimeError

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier

from base.dtconfig import DEFAULT_ARGS
from util import CfgParallelBackend
from util import adapt_massive_inputs, check_inputs
from util import estimator_is_fitted
from ensemble import BaseEnsembleTree
from ensemble import build_ensemble, predict_ensemble, voting_majority
from stats import DataStreamMonitorEntropy
from stats import DivergenceMeasures

import numpy as np

if not ("CFG_PARALLEL" in globals()):
    CFG_PARALLEL = CfgParallelBackend()

if not (gc.isenabled()):
    gc.enable()


class KTreeClassifier(BaseEstimator, ClassifierMixin, BaseEnsembleTree):
    """
    Research approach with choose base classifier entropy-based

    """

    __slots__ = ['n_estimators', 'nk_estimators', 'seed',
                 'window_size', 'kl_threshold', 'base_estimator',
                 'param_estimator', 'monitor_entropy_',
                 'data_window', 'divergence_measures']

    def __init__(self, n_estimators=100, k_estimators=25, seed=40,
                 window_size=200, kl_threshold=0.5,
                 base_estimator='DecisionTreeClassifier',
                 param_estimator=DEFAULT_ARGS['dtc'],
                 divergence_measures=DivergenceMeasures(), **kwargs):

        super().__init__(**kwargs)

        self.n_estimators = n_estimators
        self.nk_estimators = k_estimators
        self.seed = seed
        self.window_size = window_size
        self.kl_threshold = kl_threshold
        self.base_estimator = base_estimator
        self.param_estimator = param_estimator
        self.monitor_entropy_ = DataStreamMonitorEntropy()
        self.data_window = deque(maxlen=window_size)
        self.divergence_measures = divergence_measures

    def __del__(self):
        del self.monitor_entropy_

    def fit(self, X, y, **kwargs):
        """
        Util function.

        X -> list of features values
        y -> label for each X(i)

            parameters: [...]
        """
        self.monitor_entropy_.start(X).update(X)

        flag_lock = Lock()
        with flag_lock:
            Xt, yt = adapt_massive_inputs(self, X, y)
            Xt, yt = check_inputs(self, Xt, yt)

            build_ensemble(self.ensemble_, self.package_base_estimator_,
                           self.base_estimator, self.param_estimator,
                           self.n_estimators, Xt, yt, self.seed,
                           self.bootstrap_size_, self.bootstrap_feature_size_,
                           self.sample_n_feature_, self.sample_strategy_,
                           self.method_train_, **kwargs)

            #self.ensemble_ = e
            #self.k_estimators_ = e
            #self.feature_names_ = f
            #self.instances_train_ = i
        gc.collect()

        return self

    def predict(self, X, **kwargs):
        if not (estimator_is_fitted(self)):
            raise RuntimeError("This classifier not is fitted!")

        Xt = adapt_massive_inputs(self, X)
        print('Shape of X: ', np.shape(X))
        flag_lock = Lock()
        with flag_lock:
            # k_estimators is ensemble reduced
            k_estimators, feature_names, _ = self.get_k_estimators(k='all')
            #feature_names = list(map(self.get_name_feature_from_key,
            #                         feature_names))
            y_pred = predict_ensemble(k_estimators, feature_names,
                                      self.method_predict_, Xt, **kwargs)

            print('Y pred shape: ', np.shape(y_pred))

            y_pred_voting = voting_majority(y_pred)

        return y_pred_voting

    def update(self):
        pass
