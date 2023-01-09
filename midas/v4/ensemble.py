#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 16:11:33 2022

@author: Danilo Santos

@TODO
    Implementar sample_weight nos métodos fit
    Aceitar que o parâmetro seed seja uma instância de um random state
    Introduzir a possibilidade de a sequencia de indices ser passada como parÂmetro opcional para os métodos choice
    Adicionar njit form numba nas linhas de código numpy

@DOC-TO-REFERENCE
https://numpy.org/doc/stable/reference/random/legacy.html
https://numpy.org/doc/stable/reference/random/generated/numpy.random.RandomState.randint.html#numpy.random.RandomState.randint
https://numpy.org/doc/stable/reference/random/generated/numpy.random.RandomState.choice.html#numpy.random.RandomState.choice
https://numpy.org/doc/stable/reference/random/generator.html#numpy.random.default_rng
https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.choice.html#numpy.random.Generator.choice
https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.integers.html#numpy.random.Generator.integers
"""
from numba import jit

from joblib import Parallel, delayed

from threading import Lock

import gc

# from sklearn.ensemble import ForestClassifier
# from sklearn.ensemble import BaseEnsemble
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from sklearn.utils import parallel_backend

import numpy as np

import pandas as pd

import random

import secrets

from os import cpu_count

import logging

VERSION = (0, 0, 4)

__version__ = ".".join(map(str, VERSION))

if __name__ == "__main__":
    print('Project KTree Version: ', __version__)


class RFClassifier(BaseEstimator):
    """
    A simple alternative implementation of the Random Forest algorithm.

    Based in the implementation of the SKLearn.
    """

    __slots__ = ['_base_estimator', '_estimators', '_n_estimators', '_params',
                 '_feature_names', '_n_features', '_classes', '_n_classes',
                 '_sample_strategy', '_seed', '_sample_bootstrap_size',
                 '_sample_n_feature', '_parallel_n_jobs', '_parallel_verbose',
                 '_parallel_backend', '_parallel_prefer', '_parallel_require','_voting',
                 '_global_lock', '_datalogger']

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
                 voting="majority",
                 sample_strategy='numpy.random.Generator.integers',
                 sample_bootstrap_size='auto',
                 sample_n_feature='sqrt-',
                 parallel_n_jobs=-1,
                 parallel_verbose=0,
                 parallel_backend='threading',
                 parallel_prefer='threads',
                 parallel_require=None,
                 enable_logger=False
                 ):
        self._base_estimator = base_estimator
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
        self._parallel_n_jobs = parallel_n_jobs
        self._parallel_verbose = parallel_verbose
        self._parallel_backend = parallel_backend
        self._parallel_prefer = parallel_prefer
        self._parallel_require = parallel_require
        self._voting = voting
        self._global_lock = Lock()
        self._datalogger = None
        if enable_logger:
            self._setup_logger()

        gc.enable()
        # self.feature_importances_ = []
        # self.oob_score_ = []

    def __del__(self):
        #self._global_lock.release()
        #gc.collect()
        pass
        
    def fit(self, X, y):
        """
        Fit classifier.

        Trainnig several base classifier to ensemble
        """
        # Filter data
        # lock = threading.Lock()
        # with lock:
        self._global_lock.acquire(blocking=True, timeout=- 1)
        Xt, yt = self._adapt_massive_inputs(X, y)
        Xt, yt = self._check_inputs(Xt, yt)
        try:
            # Build classifiers
            self._estimators = Parallel(n_jobs=self._parallel_n_jobs,
                                        verbose=self._parallel_verbose,
                                        backend=self._parallel_backend,
                                        prefer=self._parallel_prefer,
                                        require=self._parallel_require
                                        )(
                delayed(self._make_estimator)()
                for i in range(self._n_estimators)
            )
        except Exception as e:
            print('Error build estimators')
            print(str(e))
        # Stored feature index name
        self._feature_estimators = np.ndarray(shape=self._n_estimators,
                                              dtype=object)
        try:
            # Fit each classifier
            self._estimators = Parallel(n_jobs=self._parallel_n_jobs,
                                        verbose=self._parallel_verbose,
                                        backend=self._parallel_backend,
                                        prefer=self._parallel_prefer,
                                        require=self._parallel_require
                                        )(
                delayed(self._build_estimator)(e, Xt, yt, i)
                for i, e in enumerate(self._estimators)
            )
        except Exception as e:
            print('Error fitting estimators')
            print(str(e))
                                    
        del Xt, yt
        
        self._global_lock.release()
        gc.collect()
        
        return self

    def _build_estimator(self, estimator, X, y, i):
        """
        Parallel function.

        Build the estimator from the (X,y)
        """
        local_lock = Lock()
        local_lock.acquire()
        
        X_sample, y_sample, ix_feature, ix_instances = self._get_sample_data(X, y)
        
        try:
            with parallel_backend(backend=self._parallel_backend,
                                n_jobs=self._parallel_n_jobs):
                handle_fit = estimator.fit(X_sample, y_sample)
                if not(handle_fit is None):
                    estimator = handle_fit
            self._feature_estimators[i] = ix_feature
        except Exception as e:
            print('Error in to parallel build estimator')
            print(str(e))
        
        self._register_log("Tree index [{}] build with the " \
                           "feature index {} and " \
                           "data index {}\n".format(str(i),
                                                    str(ix_feature),
                                                    str(ix_instances)))
        
        local_lock.release()
        del X_sample, y_sample, ix_feature, local_lock
        
        gc.collect()
        
        return estimator

    def _make_estimator(self):
        """
        Until function.

        Return a instance of the Decision Tree.
        """
        local_lock = Lock()
        local_lock.acquire()
        
        try:
            with parallel_backend(backend=self._parallel_backend,
                                n_jobs=self._parallel_n_jobs):
                est_base = self._base_estimator(**self._params)
        except Exception as e:
            print('Error in to parallel make estimator')
            print(str(e))
            
        local_lock.release()
        del local_lock
        
        gc.collect()
        
        return est_base

    @njit(nopython=True, nogil=True, parallel=True, fastmath=True)
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

    #@njit(nopython=True, nogil=True, parallel=True, fastmath=True)
    def _get_sample_n_feature(self):
        """
        Until function.

        Return the number of features to training.
        """
        if self._sample_n_feature == 'sqrt+':
            return int(np.sqrt(self._n_features)+0.5)
        if self._sample_n_feature == 'sqrt-':
            return int(np.sqrt(self._n_features))
        if self._sample_n_feature == 'log2+':
            return int(np.log2(self._n_features)+0.5)
        if self._sample_n_feature == 'log2-':
            return int(np.log2(self._n_features))
        if self._sample_n_feature == 'log10+':
            return int(np.log10(self._n_features)+0.5)
        if self._sample_n_feature == 'log10-':
            return int(np.log10(self._n_features))
        if self._sample_n_feature == 'auto':
            return self._n_features
        if isinstance(self._sample_n_feature, int):
            return self._sample_n_feature
        if isinstance(self._sample_n_feature, float):
            return int(self._sample_n_feature * self._n_features)
        raise ValueError('Impossible determine to sample n feature.')

    #@njit(cache=True, parallel=True, fastmath=True)
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

    #@njit(cache=True, parallel=True, fastmath=True)
    def _get_sample_data(self, X, y):
        """
        Until function.

        Making random sample.
        """
        local_lock = Lock()
        local_lock.acquire()
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

        local_lock.release()
        del n_sample, boostrap_instance_size, boostrap_feature_size, local_lock
        gc.collect()
        
        return (X_sample, y_sample, ix_features, ix_instance)

    def _is_fitted(self):
        """
        Until function.
        
        Return boolean if this classifier is fitted.
        """
        return len(self._estimators) > 0
 
    #@njit(cache=True, parallel=True, fastmath=True)   
    def _adapt_massive_inputs(self, X, y=None):
        """
        Until function.

        Return the inputs adapteds.
        """
        local_lock = Lock()
        local_lock.acquire()
        Xt = None
        yt = None
        if not(self._is_fitted()):
            if isinstance(X, pd.core.frame.DataFrame):
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
        else:
            # verificar se é o mesmo x
            if np.shape(X)[1] != self._n_features:
                raise ValueError("X unknown size feature")
            if isinstance(X, pd.core.frame.DataFrame) and all(X.columns != self._feature_names):
                raise ValueError('X have unknown features.')
            Xt = np.array(X)
            yt = np.array(y)

        local_lock.release()
        del local_lock
        gc.collect()
        
        if (y is None):
            return Xt

        return (Xt, yt)

    #@njit(cache=True, parallel=True, fastmath=True)
    def _check_inputs(self, X, y):
        """
        Until function.

        Check data input.
        """
        # verificar se é matriz
        # verificar se tem tamanho igual de x e y
        local_lock = Lock()
        local_lock.acquire()
        if np.shape(X)[1] < 1:
            raise ValueError('X must be an NxM array.')
        if y is None:
            raise ValueError('Target label is expected.')
        if np.shape(X)[0] != np.shape(y)[0]:
            raise ValueError('X and y must be the same size.')
        local_lock.release()
        del local_lock
        gc.collect()
        
        return (X, y)

    def get_feature_name_from_tree(self, index_tree):
        """
        Get function.
        
        Return the name feature for trained tree.
        """
        if len(self._estimators) == 0:
            raise ValueError('Ensemble not is initialized.')
        if index_tree < len(self._estimators) or index > len(self._estimators):
            raise ValueError('Index out of range.')
        names_feature = []
        if not(self._feature_names is None):
            names_feature = [self._feature_names[i] for i in self._feature_estimators[index_tree]]
        return names_feature

    def set_parallel_options(self, parallel_n_jobs=-1, parallel_verbose=0, 
                             parallel_backend='threading', 
                             parallel_prefer='threads', parallel_require=None):
        self._parallel_n_jobs = parallel_n_jobs
        self._parallel_verbose = parallel_verbose
        self._parallel_backend = parallel_backend
        self._parallel_prefer = parallel_prefer
        self._parallel_require = parallel_require
        
    def _get_n_jobs(self):
        """
        Util function.

        Util for to return get the number of CPUs in the system or
        the number choose from user.
        parameters: None
        """
        return (cpu_count() if self._parallel_n_jobs == -1 else self._parallel_n_jobs)

    def _parallel_predict(self, est, X, ix_feature_name, kwargs={}):
        """
        Util function.

        @Todo Validar 'est' para verificar se é um classificador e se tem o
        método predict

        parameters: [...]
        """
        local_lock = Lock()
        local_lock.acquire()
        y_pred = Parallel(n_jobs=self._parallel_n_jobs,
                          verbose=self._parallel_verbose,
                          backend=self._parallel_backend,
                          prefer=self._parallel_prefer,
                          require=self._parallel_require
                )(
                    delayed(est.predict)(Xi, **dict(kwargs))
                    for Xi in np.array_split(np.take(X, ix_feature_name, axis=1), self._get_n_jobs())
            )
        local_lock.release()
        del local_lock
        gc.collect()
        
        return np.ravel(y_pred)

    def _predict_majority(self, X, kwargs={}):
        """
        Util function.

            X -> unknow instances
        """
        local_lock = Lock()
        local_lock.acquire()
        predictions = Parallel(n_jobs=self._parallel_n_jobs,
                               verbose=self._parallel_verbose,
                               backend=self._parallel_backend,
                               prefer=self._parallel_prefer,
                               require=self._parallel_require
                    )(
                        delayed(self._parallel_predict)(Ei, X, self._feature_estimators[i], kwargs)
                        for i, Ei in enumerate(self._estimators)
        )

        predictions = np.array(predictions, dtype=np.int64)
        predictions = predictions.T

        maj = np.apply_along_axis(lambda x: np.bincount(x).argmax(),
                                  axis=1, arr=predictions)

        local_lock.release()
        del predictions, local_lock
        gc.collect()
        
        return maj

    def predict(self, X, kwargs={}):
        """
        Util function.

        X -> unknow instances
        """
        if not(self._is_fitted()):
            raise Exception("Classifier not is fitted!")
        
        self._global_lock.acquire(blocking=True, timeout=-1)
        
        # chamar adapt
        Xt = self._adapt_massive_inputs(X)

        y_pred = []
        if self._voting == "majority":
            y_pred = self._predict_majority(Xt, kwargs)
        else:
            raise Exception('Method of voting required')

        self._global_lock.release()
        del Xt
        gc.collect()

        return y_pred

    def score(self, X, y):
        """
        Util function.

        @todo implement sample_weight parameter
        """
        y_pred = self.predict(X)
        self._global_lock.acquire(blocking=True, timeout=- 1)
        acc = None
        with parallel_backend(backend=self._parallel_backend, n_jobs=self._parallel_n_jobs):
            acc = accuracy_score(y, y_pred)
        self._global_lock.release()
        
        return acc

    def _parallel_predict_proba(self, est, X, ix_feature_name, check_input=True):
        """
        Util function.

        implement predict proba parallel
        """
        local_lock = Lock()
        local_lock.acquire()
        y_proba = Parallel(n_jobs=self._parallel_n_jobs,
                           verbose=self._parallel_verbose,
                           backend=self._parallel_backend,
                           prefer=self._parallel_prefer,
                           require=self._parallel_require
                )(
                    delayed(est.predict_proba)(Xi, check_input)
                    for Xi in np.array_split(np.take(X, ix_feature_name, axis=1), self._get_n_jobs())
            )

        local_lock.release()
        del local_lock
        gc.collect()
        
        return np.ravel(y_proba)

    def predict_proba(self, X, check_input=True):
        """
        Util function.

        @todo Verificar como a matriz predicitons está send retornada para
        fazer a média porque cada chamada de proba retorna
        ndarray of shape (n_samples, n_classes)
        """
        self._global_lock.acquire(blocking=True, timeout=- 1)
        predictions = Parallel(n_jobs=self._parallel_n_jobs,
                               verbose=self._parallel_verbose,
                               backend=self._parallel_backend,
                               prefer=self._parallel_prefer,
                               require=self._parallel_require
                    )(
                        delayed(self._parallel_predict_proba)(Ei, X, self._feature_estimators[i], check_input)
                        for i, Ei in enumerate(self._estimators)
        )

        predictions = np.array(predictions, dtype=np.int64)
        predictions = predictions.T

        med = np.apply_along_axis(lambda x: np.mean(x),
                                  axis=1, arr=predictions)

        self._global_lock.release()
        del predictions
        gc.collect()
        
        return med

    def _setup_logger(self, name='rflog', log_file='./rf.debuging.log', level=logging.INFO):
        """
        Utily function.

        Function setup as many loggers as you want
        """
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        handler = logging.FileHandler(log_file)        
        handler.setFormatter(formatter)

        self._datalogger = logging.getLogger(name)
        self._datalogger.setLevel(level)
        self._datalogger.addHandler(handler)
    
    def _register_log(self, message="\n"):
        """
        Args:
            level (_type_): _description_
            message (_type_): _description_
        """
        if not(self._datalogger is None):
            local_lock = Lock()
            local_lock.acquire()
            self._datalogger.info(str(message))
            local_lock.release()
        