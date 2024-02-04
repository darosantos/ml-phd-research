#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fou Nov 15 11:16:35 2023

@author: Danilo Santos
"""


from joblib import Parallel, delayed

from threading import Lock

import gc

from sklearn.tree import DecisionTreeClassifier

import ensemble 


class KTreeClassifier(ensemble.RFClassifier):
    """
    K Tree Classifier V4 inspired our function entropy-based
    """
    
    __slots__ = ['_update_strategy']
    
    def __init__(self, update_strategy='entropy', *kwargs):
        super().__init__(kwargs.get('base_estimator',DecisionTreeClassifier),
                         kwargs.get('n_estimators',100),
                         kwargs.get('params_estimators',{'criterion': 'entropy', 'splitter': 'best',
                                                         'max_depth': None, 'min_samples_split': 2,
                                                         'min_samples_leaf': 2,
                                                         'min_weight_fraction_leaf': 0.0,
                                                         'max_features': 'auto', 'random_state': 100,
                                                         'max_leaf_nodes': None,
                                                         'min_impurity_decrease': 0.0,
                                                         'class_weight': None, 'ccp_alpha': 0.0}),
                         kwargs.get('seed',100), kwargs.get('voting',"majority"),
                         kwargs.get('sample_strategy',"numpy.random.Generator.integers").
                         kwargs.get('sample_bootstrap_size',"auto"),
                         kwargs.get('sample_n_feature',"sqrt-"),kwargs.get('parallel_n_jobs',-1),
                         kwargs.get('parallel_verbose',0),
                         kwargs.get('parallel_backend',"'threading'"),
                         kwargs.get('parallel_prefer',"threads"),
                         kwargs.get('parallel_require',None),
                         kwargs.get('enable_logger',False))
        self._update_strategy = update_strategy
    
    def __del__(self):
        pass
    
    def fit(self, X, y):
        super().fit(X, y)
    
    def predict(self, X):
        pass
    
    def _update_strategy_entropy(self):
        pass