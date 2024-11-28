#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 17:08:24 2022

@author: Danilo Santos
"""

from joblib import Parallel, delayed

import numpy as np


from sklearn.base import BaseEstimator


class EnsembleTrees(object, BaseEstimator):
    """
    Train um ensemble de trees well under of Random Forest implementation.

    A dummye alternative to Random Forest.
    """

    __slots__ = ['_estimators']

    def __init__(self):
        pass
    
    def fit(self):
        pass
    
    def predict(self):
        pass