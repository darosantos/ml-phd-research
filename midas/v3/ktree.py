#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 01:39:37 2022.

@author: Danilo Santos
"""

from os import cpu_count

from joblib import Parallel, delayed

import numpy as np


from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import LabelBinarizer


VERSION = (0, 0, 3)

__version__ = ".".join(map(str, VERSION))


if __name__ == "__main__":
    print('KTree Version: ', __version__)


class KTreeClassifier(object, BaseEstimator):
    """
    K Tree Classifier V3.

    strategy = ['auto', 'score']
    """

    #__slots__ = ['ensemble_base', 'k_tree', 'strategy', 'metric_type',
    #             'metric_param', 'voting', 'multi_class', 'n_jobs',
    #             'verbose']

    def __init__(self,
                 ensemble_base,
                 k_tree=25,
                 strategy="auto",
                 metric_type="roc_auc_score",
                 metric_param=None,
                 voting="majority",
                 multi_class=False,
                 n_jobs=-1,
                 verbose=0):
        self.ensemble = ensemble_base
        self.n_tree = k_tree
        self.strategy = str(strategy).lower()
        self.metric_type = str(metric_type).lower()
        self.metric_param = metric_param
        self.voting = str(voting).lower()
        self.multi_class = multi_class
        self.n_jobs = n_jobs
        self.verbose = verbose

        self.classes_ = None
        self.metric = None
        self.args_metric = None
        self.index_k_tree = []
        self.fit_scores_ = []

        if isinstance(k_tree, float):
            self.n_tree = int(k_tree*self._get_ensemble_size())

    def _is_multi_class(self, y):
        """
        Util function.

        @Todo implementar verificar se o array y é binário de fato
        """
        if np.unique(y).shape[0] == 2:
            if all([e in self.get_classes() for e in [0, 1]]):
                return False
        return True

    def _get_n_jobs(self):
        """
        Util function.

        Util for to return get the number of CPUs in the system or
        the number choose from user.
        parameters: None
        """
        return (cpu_count() if self.n_jobs == -1 else self.n_jobs)

    def get_score_train(self, std=False):
        """
        Util function.

        parameters: none
        """
        if std:
            return (np.mean(self.fit_scores_), np.std(self.fit_scores_))
        return np.mean(self.fit_scores_)

    def get_index_estimators(self):
        """
        Util function.

        parameters: None
        """
        return list(self.index_k_tree)

    def get_nk_tree(self):
        """
        Util function.

        parameters: none
        """
        return self.n_tree

    def _get_ensemble_size(self):
        """
        Return of the numbers DTs in ensemble.

        parameters: None
        """
        return len(self.ensemble.estimators_)

    def _get_metric(self):
        """
        Decode the metric function to use for evaluable classifiers.

        parameters: None
        """
        if self.metric_type == 'accuracy_score':
            return accuracy_score
        if self.metric_type == 'cohen_kappa_score':
            return cohen_kappa_score
        if self.metric_type == 'f1_score':
            return f1_score
        if self.metric_type == 'balanced_accuracy_score':
            return balanced_accuracy_score
        if self.metric_type == 'precision_score':
            return precision_score
        if self.metric_type == 'recall_score':
            return recall_score
        if self.metric_type == 'roc_auc_score':
            return roc_auc_score
        raise Exception('The parameter metric_type can not empty or unknow')

    def _get_metric_params(self):
        dict_params = {}
        smp = self.metric_param
        if self.metric_type == 'accuracy_score':
            dict_params = {'normalize': smp.get('normalize', True),
                           'sample_weight': smp.get('sample_weight', None)
                           }
        elif self.metric_type == 'cohen_kappa_score':
            dict_params = {'labels': smp.get('labels', None),
                           'weights': smp.get('weights', None),
                           'sample_weight': smp.get('sample_weight', None)
                           }
        elif all(self.metric_type == 'f1_score', not(self.multi_class)):
            dict_params = {'labels': smp.get('labels', None),
                           'pos_label': smp.get('pos_label', 1),
                           'average': smp.get('average', 'binary'),
                           'sample_weight': smp.get('sample_weight', None),
                           'zero_division': smp.get('zero_divisiont', 'warn')
                           }
        elif all(self.metric_type == 'f1_score', self.multi_class):
            dict_params = {'labels': smp.get('labels', None),
                           'pos_label': smp.get('pos_label', 1),
                           'average': smp.get('average', 'weighted'),
                           'sample_weight': smp.get('sample_weight', None),
                           'zero_division': smp.get('zero_divisiont', 1)
                           }
        elif self.metric_type == 'balanced_accuracy_score':
            dict_params = {'sample_weigh': smp.get('sample_weigh', None),
                           'adjusted': smp.get('adjusted', False)
                           }
        elif all(self.metric_type == 'precision_score', not(self.multi_class)):
            dict_params = {'labels': smp.get('labels', None),
                           'pos_label': smp.get('pos_label', 1),
                           'average': smp.get('average', 'binary'),
                           'sample_weight': smp.get('sample_weight', None),
                           'zero_division': smp.get('zero_division', 'warn')
                           }
        elif all(self.metric_type == 'precision_score', self.multi_class):
            dict_params = {'labels': smp.get('labels', None),
                           'pos_label': smp.get('pos_label', 1),
                           'average': smp.get('average', 'weighted'),
                           'sample_weight': smp.get('sample_weight', None),
                           'zero_division': smp.get('zero_division', 1)
                           }
        elif all(self.metric_type == 'recall_score', not(self.multi_class)):
            dict_params = {'labels': smp.get('labels', None),
                           'pos_label': smp.get('pos_label', 1),
                           'average': smp.get('average', 'binary'),
                           'sample_weight': smp.get('sample_weight', None),
                           'zero_division': smp.get('zero_division', 'warn')
                           }
        elif all(self.metric_type == 'recall_score', self.multi_class):
            dict_params = {'labels': smp.get('labels', None),
                           'pos_label': smp.get('pos_label', 1),
                           'average': smp.get('average', 'weighted'),
                           'sample_weight': smp.get('sample_weight', None),
                           'zero_division': smp.get('zero_division', 1)
                           }
        elif all(self.metric_type == 'roc_auc_score', not(self.multi_class)):
            dict_params = {'average': smp.get('average', 'macro'),
                           'sample_weight': smp.get('sample_weight', None),
                           'max_fpr': smp.get('max_fpr', None),
                           'multi_class': smp.get('multi_class', 'raise'),
                           'labels': smp.get('labels', None)
                           }
        elif all(self.metric_type == 'roc_auc_score', self.multi_class):
            dict_params = {'average': smp.get('average', 'weightedo'),
                           'sample_weight': smp.get('sample_weight', None),
                           'max_fpr': smp.get('max_fpr', None),
                           'multi_class': smp.get('multi_class', 'raise'),
                           'labels': smp.get('labels', None)
                           }
        else:
            msg = 'Without a known metric_type parameter we cannot determine '
            msg += 'the other auxiliary parameters'
            raise Exception(msg)

        return dict_params

    def get_classes(self):
        """
        Util function.

        Return n classes
        """
        return self.classes_

    def set_classes(self, y, increment=False):
        """
        Util function.

        Set n classes.
        """
        classes_ = np.unique(y)
        if all(increment, not(isinstance(self.classes_, None))):
            self.classes_ = np.unique(np.append(self.classes_, classes_))
        else:
            self.classes_ = classes_

    def _parallel_predict(self, est, X, check_input=True):
        """
        Util function.

        @Todo Validar 'est' para verificar se é um classificador e se tem o
        método predict

        parameters: [...]
        """
        y_pred = Parallel(n_jobs=self._get_n_jobs(),
                          verbose=self.verbose, require='sharedmem'
        )(
            delayed(est.predict)(Xi, check_input)
            for Xi in np.array_split(X, self._get_n_jobs())
        )

        return np.ravel(y_pred)

    def _parallel_predict_proba(self, est, X, check_input=True):
        """
        Util function.

        implement predict proba parallel
        """
        y_proba = Parallel(n_jobs=self._get_n_jobs(),
                           verbose=self.verbose, require='sharedmem'
        )(
            delayed(est.predict_proba)(Xi, check_input)
            for Xi in np.array_split(X, self._get_n_jobs())
        )

        return np.ravel(y_proba)

    def _metric_evaluable(self, index, est, X, y_true, check_input=True):
        """
        Return scored for X from metric with y_true.

        parameters: [...]
        """
        y_pred = self._parallel_predict(est, X, check_input)
        if self.multi_class:
            lb = LabelBinarizer()
            lb = lb.fit(y_true)
            yt = lb.transform(y_true)
            yp = lb.transform(y_pred)
            score = self.metric(yt, yp, **self.args_metric)
        else:
            score = self.metric(y_true, y_pred, **self.args_metric)

        return (index, score)

    def _fit_score(self, X_true, y_true, check_input=True):
        """
        Util function.

        X_true ->
        y_true ->

        parameters: [..]
        """
        self.metric = self._get_metric()
        self.args_metric = self._get_metric_params()

        lst_scores = Parallel(n_jobs=self.n_jobs,
                              verbose=self.verbose,
                              require='sharedmem'
        )(
            delayed(self._metric_evaluable)(xi, est, X_true,
                                            y_true, check_input)
            for xi, est in enumerate(self.ensemble.estimators_)
        )

        ordered_scores = sorted(lst_scores,
                                key=lambda position: position[1], reverse=True)

        self.index_k_tree = np.array(
            [ik[0] for ik in ordered_scores[:self.n_tree]], dtype=np.int64)

        self.fit_scores_ = np.array(
            [ik[1] for ik in ordered_scores[:self.n_tree]], dtype=np.float64)

        del lst_scores, ordered_scores

    def _fit_auto(self, X, y, check_input=True):
        pass

    def fit(self, X, y, check_input=True):
        """
        Util function.

        X -> list of features values
        y -> label for each X(i)

            parameters: [...]
        """
        if len(self.index_k_tree) != 0:
            raise Exception("Classifier is fitted!")

        self.set_classes(y)

        if all(self._is_multi_class(y), not(self.multi_class)):
            msg = 'Inconsistent problem! Multiclass must be explicit'
            raise Exception(msg)
        if all(not(self._is_multi_class(y)), self.multi_class):
            msg = 'Inconsistent problem! Target not is multiclasses'
            raise Exception(msg)

        if self.strategy == 'score':
            self._fit_score(X, y, check_input)
        elif self.strategy == 'auto':
            pass
        else:
            raise Exception('Unknow strategy!')

        return self

    def fit_update(self, X, y, check_input=True):
        """
        Util function.

        X -> list of features values
        y -> label for each X(i)
        """
        if self.index_k_tree.sum() == 0:
            raise Exception("Classifier isn't fitted")

        if self.strategy == 'score':
            self._fit_score(X, y, check_input)
        elif self.strategy == 'auto':
            pass
        else:
            raise Exception('Unknow strategy!')

        return self

    def _predict_majority(self, X, check_input=True):
        """
        Util function.

            X -> unknow instances
        """
        predictions = Parallel(n_jobs=self.n_jobs,
                               verbose=self.verbose,
                               require='sharedmem'
        )(
          delayed(self._parallel_predict)(self.ensemble.estimators_[Ei], X,
                                          check_input)
          for Ei in self.index_k_tree
        )

        predictions = np.array(predictions, dtype=np.int64)
        predictions = predictions.T

        maj = np.apply_along_axis(lambda x: np.bincount(x).argmax(),
                                  axis=1, arr=predictions)

        del predictions
        return maj

    def predict(self, X, check_input=True):
        """
        Util function.

        X -> unknow instances
        """
        if len(self.index_k_tree) == 0:
            raise Exception("Classifier not is fitted!")

        if self.voting == "majority":
            return self._predict_majority(X, check_input)

        raise Exception('Method of voting required')

    def score(self, X, y, use_metric_type=False):
        """
        Util function.

        @todo implement sample_weight parameter
        """
        y_pred = self.predict(X)
        if use_metric_type:
            metric = self._get_metric()
            args_metric = self._get_metric_params()
            return metric(y, y_pred, **args_metric)

        return accuracy_score(y, y_pred)

    def predict_proba(self, X, check_input=True):
        """
        Util function.

        @todo Verificar como a matriz predicitons está send retornada para
        fazer a média porque cada chamada de proba retorna
        ndarray of shape (n_samples, n_classes)
        """
        predictions = Parallel(n_jobs=self.n_jobs,
                               verbose=self.verbose,
                               require='sharedmem'
        )(
          delayed(self._parallel_predict_proba)(self.ensemble.estimators_[Ei],
                                                X, check_input)
          for Ei in self.index_k_tree
        )

        predictions = np.array(predictions, dtype=np.int64)
        predictions = predictions.T

        med = np.apply_along_axis(lambda x: np.mean(x),
                                  axis=1, arr=predictions)

        del predictions
        return med

    def learn_one(self):
        """
        Util function.

        A wrapper for learning compatible with the river plataform, but
        to using lazy classifiers (i. e. sklearn implementation)
        """
        return self

    def predict_one(self):
        """
        Util function.

        A wapprer for predict compatible with the river plataform, but to
        using lazy classifiers (i. e. sklearn implementation)
        """
        return self
