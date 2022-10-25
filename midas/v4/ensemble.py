#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 16:11:33 2022

@author: Danilo Santos

@TODO
    Implementar sample_weight nos métodos fit
    Aceitar que o parâmetro seed seja uma instância de um random state
    Introduzir a possibilidade de a sequencia de indices ser passada como parÂmetro opcional para os métodos choice

@DOC-TO-REFERENCE
https://numpy.org/doc/stable/reference/random/legacy.html
https://numpy.org/doc/stable/reference/random/generated/numpy.random.RandomState.randint.html#numpy.random.RandomState.randint
https://numpy.org/doc/stable/reference/random/generated/numpy.random.RandomState.choice.html#numpy.random.RandomState.choice
https://numpy.org/doc/stable/reference/random/generator.html#numpy.random.default_rng
https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.choice.html#numpy.random.Generator.choice
https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.integers.html#numpy.random.Generator.integers
"""
# from sklearn.ensemble import ForestClassifier
# from sklearn.ensemble import BaseEnsemble
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator

import numpy as np

import pandas as pd

import random

import secrets

from joblib import Parallel, delayed


class RandomForestBase(object):
    """
    Standard class for implement random forest.

    A collection of methods for facility the RF implementation.
    """

    def __init__(self):
        pass

    def add_tree(self):
        pass

    def rem_tree(self):
        pass

    def build_tree(self, estimator, params, X, y):
        pass


class RFClassifier(BaseEstimator):
    """
    A simple alternative implementation of the Random Forest algorithm.

    Based in the implementation of the SKLearn.
    """

    __slots__ = ['_base_estimator', '_estimators', '_n_estimators', '_params',
                 '_feature_names', '_n_features', '_classes', '_n_classes',
                 '_sample_strategy', '_seed', '_sample_bootstrap_size',
                 '_sample_n_feature', '_n_jobs', '_verbose']

    def __init__(self,
                 base_estimator=DecisionTreeClassifier,
                 n_estimators=100,
                 params_estimators={'criterion': 'entropy', 'splitter': 'best',
                                    'max_depth': None, 'min_samples_split': 2,
                                    'min_samples_leaf': 2,
                                    'min_weight_fraction_leaf': 0.0,
                                    'max_features': 'auto', 'random_state': 100,
                                    'max_leaf_nodes': None,
                                    'min_impurity_decrease': 0.0,
                                    'class_weight': None, 'ccp_alpha': 0.0},
                 seed=100,
                 sample_strategy='numpy.random.Generator.integers',
                 sample_bootstrap_size='auto',
                 sample_n_feature='sqrt',
                 n_jobs=-1,
                 verbose=0
                 ):
        self._base_estimator = base_estimator
        # self._estimators = np.ndarray(shape=n_estimators, dtype=object)
        self._estimators = []
        self._feature_estimators = []
        self._n_estimators = n_estimators
        self._params = params_estimators
        self._feature_names = []
        self._n_features = 0
        self._classes = []
        self._n_classes = 0
        self._sample_strategy = sample_strategy
        self._seed = seed
        self._sample_bootstrap_size = sample_bootstrap_size
        self._sample_n_feature = sample_n_feature
        self._n_jobs = n_jobs
        self._verbose = verbose

        # self.feature_importances_ = []
        # self.oob_score_ = []

    def fit(self, X, y):
        """
        Fit classifier.

        Trainnig several base classifier to ensemble
        """
        # Filter data
        Xt, yt = self._adapt_massive_inputs(X, y)
        Xt, yt = self._check_inputs(Xt, yt)
        # Build classifiers
        self._estimators = Parallel(n_jobs=self._n_jobs,
                                    verbose=self._verbose,
                                    prefer="threads",
                                    require='sharedmem'
                                    )(
            delayed(self._make_estimator)()
            for i in range(self._n_estimators)
        )
        # Stored feature index name
        self._feature_estimators = np.ndarray(shape=self._n_estimators,
                                              dtype=object)
        # Fit each classifier
        self._estimators = Parallel(n_jobs=self._n_jobs,
                                    verbose=self._verbose,
                                    prefer="threads",
                                    require='sharedmem'
                                    )(
            delayed(self._build_estimator)(e, Xt, yt, i)
            for i, e in enumerate(self._estimators)
        )
        return self

    def _build_estimator(self, estimator, X, y, i):
        """
        Parallel function.

        Build the estimator from the (X,y)
        """
        X_sample, y_sample, ix_feature, _ = self._get_sample_data(X, y)
        estimator = estimator.fit(X_sample, y_sample)
        self._feature_estimators[i] = ix_feature
        return estimator

    def _make_estimator(self):
        """
        Until function.

        Return a instance of the Decision Tree.
        """
        return self._base_estimator(**self._params)

    def _generate_sample_indices(self,
                                 seed,
                                 n_samples,
                                 n_samples_bootstrap,
                                 sample_weight=None,
                                 replace=True,
                                 probabilities=None,
                                 endpoint=False,
                                 axis=0,
                                 shuffle=True):
        """
        Until function.

        Generate random sample acoodily to strategy
        """
        sample_indices = []
        if self._sample_strategy == 'numpy.random.RandomState.randint':
            random_state = np.random.RandomState(seed)
            sample_indices = random_state.randint(0, n_samples,
                                                  n_samples_bootstrap)
        elif self._sample_strategy == 'numpy.random.RandomState.choice':
            random_state = np.random.RandomState(seed)
            a = [i for i in range(n_samples)]
            sample_indices = random_state.choice(a=a, size=n_samples_bootstrap,
                                                 replace=replace,
                                                 p=probabilities)
        elif self._sample_strategy == 'numpy.random.Generator.integers':
            rng = np.random.default_rng(seed)
            sample_indices = rng.integers(low=0, high=n_samples,
                                          size=n_samples_bootstrap,
                                          endpoint=endpoint)
        elif self._sample_strategy == 'numpy.random.Generator.choice':
            rng = np.random.default_rng(seed)
            a = [i for i in range(n_samples)]
            sample_indices = rng.choice(a=a, size=n_samples_bootstrap,
                                        replace=replace,
                                        p=probabilities,
                                        axis=axis,
                                        shuffle=shuffle)
        elif self._sample_strategy == 'random.randrange':
            random.seed(seed)
            sample_indices = [random.randrange(0, n_samples)
                              for i in range(n_samples_bootstrap)]
        elif self._sample_strategy == 'random.choices':
            # High recommend for reprodutibily
            random.seed(seed)
            a = [i for i in range(n_samples)]
            sample_indices = random.choices(population=a,
                                            weights=sample_weight,
                                            k=n_samples_bootstrap)
        elif self._sample_strategy == 'random.sample':
            random.seed(seed)
            a = [i for i in range(n_samples)]
            sample_indices = random.sample(a, k=n_samples_bootstrap)
        elif self._sample_strategy == 'secrets.choice':
            a = [i for i in range(n_samples)]
            sample_indices = [secrets.choice(a)
                              for i in range(n_samples_bootstrap)]
        elif self._sample_strategy == 'secrets.randbelow':
            sample_indices = [secrets.randbelow(n_samples)
                              for i in range(n_samples_bootstrap)]
        else:
            raise ValueError('Random sampling strategy unrecongnized')
        return sample_indices

    def _get_sample_n_feature(self):
        """
        Until function.

        Return the number of features to training.
        """
        if self._sample_n_feature == 'sqrt':
            return int(np.sqrt(self._n_features))
        if self._sample_n_feature == 'log2':
            return int(np.log2(self._n_features))
        if self._sample_n_feature == 'log10':
            return int(np.log10(self._n_features))
        if self._sample_n_feature == 'auto':
            return self._n_features
        if isinstance(self._sample_n_feature, int):
            return self._sample_n_feature
        if isinstance(self._sample_n_feature, float):
            return int(self._sample_n_feature * self._n_features)
        raise ValueError('Impossible determine to sample n feature.')

    def _calc_boostrap_size(self, size):
        """
        Until function.

        Return the size of boostrap sample.
        """
        if self._sample_bootstrap_size == 'auto':
            return size
        if isinstance(self._sample_bootstrap_size, int):
            return self._sample_bootstrap_size
        if isinstance(self._sample_bootstrap_size, float):
            return int(size*self._sample_bootstrap_size)
        raise ValueError('Impossible determine to size bootstrap.')

    def _get_sample_data(self, X, y):
        """
        Until function.

        Making random sample.
        """
        n_sample = X.shape[0]
        ix_features = []
        boostrap_instance_size = self._calc_boostrap_size(n_sample)
        boostrap_feature_size = self._get_sample_n_feature()
        if boostrap_feature_size == self._n_features:
            ix_features = [i for i in range(boostrap_feature_size)]
        else:
            ix_features = random.sample(
                population=[i for i in range(self._n_features)],
                k=boostrap_feature_size
            )
        ix_instance = self._generate_sample_indices(self._seed,
                                                    n_sample,
                                                    boostrap_instance_size)
        X_sample = np.take(np.take(X, ix_instance, axis=0),
                           ix_features,
                           axis=1)
        y_sample = np.take(y, ix_instance, axis=0)

        return (X_sample, y_sample, ix_features, ix_instance)

    def _adapt_massive_inputs(self, X, y=None):
        """
        Until function.

        Return the inputs adapteds.
        """
        Xt = None
        yt = None
        if isinstance(type(X), pd.core.frame.DataFrame):
            self._feature_names = np.array(X.columns)
            self._n_features = len(self._feature_names)
            Xt = X.to_numpy(copy=True)
        else:
            self._feature_names = None
            self._n_features = np.shape(X)[1]
            Xt = np.array(X)

        if (not(y is None)):
            self._classes = np.unique(y)
            self._n_classes = len(self._classes)
            yt = np.array(y)

        return (Xt, yt)

    def _check_inputs(self, X, y):
        """
        Until function.

        Check data input.
        """
        # verificar se é matriz
        # verificar se tem tamanho igual de x e y
        if np.shape(X)[1] < 1:
            raise ValueError('X must be an NxM array.')
        if y is None:
            raise ValueError('Target label is expected.')
        if np.shape(X)[0] != np.shape(y)[0]:
            raise ValueError('X and y must be the same size.')
        return (X, y)


    def predict(X):
        pass

    def predict_log_proba(X):
        pass

    def predict_proba(X):
        pass
