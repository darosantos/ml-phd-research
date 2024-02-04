#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fou Jan 15 09:31:35 2024

@author: Danilo Santos
"""
from sklearn.tree import DecisionTreeClassifier


# DTC = DecisionTreeClassifier
DEFAULT_ARGS = {'dtc': {'criterion': 'gini', 'splitter': 'best',
                        'max_depth': None, 'min_samples_split': 2,
                        'min_samples_leaf': 1,
                        'min_weight_fraction_leaf': 0.0,
                        'max_features': None, 'random_state': 100,
                        'max_leaf_nodes': None,
                        'min_impurity_decrease': 0.0,
                        'class_weight': None, 'ccp_alpha': 0.0},
                }
