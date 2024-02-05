#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Dom Dez 10 16:05:08 2023

@author: Danilo Santos
"""

import pandas as pd

import numpy as np

from river import stats

from joblib import Parallel, delayed
from threading import Lock
import gc
from builtins import ValueError
from typing import Any, List

import random

import secrets

from util import CfgParallelBackend


if not ('CFG_PARALLEL' in globals()):
    CFG_PARALLEL = CfgParallelBackend()

if (not (gc.isenabled())):
    gc.enable()

# -------------------- Classes --------------------


class DataStreamMonitorEntropy(object):
    """
    Monitoring the entropy for stream object
    Args:
        object (_type_): _description_
    """

    __slots__ = ['_features_name', '_feature_entropy']

    def __init__(self, features_name=[]):
        self._feature_entropy = {}
        self._features_name = []

        self.start(features_name)

    def __del__(self):
        del self._feature_entropy
        del self._features_name
        gc.collect()

    def is_empty(self):
        return (len(self._features_name) == 0)

    def start(self, features_name):
        if self.is_empty():
            self.set_feature_name(features_name)
            self._run_monitor()

        return self

    def reset(self):
        self._feature_entropy = {}
        self._run_monitor()

        return self

    def set_feature_name(self, features_name):
        if isinstance(features_name, list):
            self._features_name = features_name
        elif isinstance(features_name, dict):
            self._features_name = list(features_name.keys())
        elif isinstance(features_name, pd.core.frame.DataFrame):
            self._features_name = features_name.columns.values.tolist()
        else:
            raise Exception('Unable to extract feature names')

    def get_features_name(self):
        return self._features_name

    def _run_monitor(self):
        if not (self.is_empty()):
            local_lock = Lock()
            local_lock.acquire()
            _ = Parallel(n_jobs=CFG_PARALLEL.n_jobs,
                         verbose=CFG_PARALLEL.verbose,
                         backend=CFG_PARALLEL.backend,
                         prefer=CFG_PARALLEL.prefer,
                         require=CFG_PARALLEL.require
                         )(
                             delayed(self._init_entropy)(name)
                             for name in self._features_name
                        )
            local_lock.release()
            del local_lock

    def _init_entropy(self, name):
        self._feature_entropy[name] = stats.Entropy(fading_factor=1)

    def _wrapper_update(self, e: Any) -> List[None]:
        _ = [self._update_one(k, v) for k, v in e.items()]

    def update(self, data, force_parallel=False):
        tp_df = pd.core.frame.DataFrame
        if isinstance(data, tp_df) and not (force_parallel):
            _ = [self._wrapper_update(e)
                 for e in data.to_dict(orient='records')]
        elif isinstance(data, tp_df) and force_parallel:
            self._update_with_pandas(data)
        elif isinstance(data, dict) and not (force_parallel):
            _ = [self._update_one(k, v) for k, v in data.items()]
        elif isinstance(data, dict) and force_parallel:
            self._update_with_dict(data)
        elif isinstance(data, list) and not (force_parallel):
            _ = [self._update_one(n, data[i])
                 for i, n in enumerate(self._features_name)]
        elif isinstance(data, list) and force_parallel:
            self._update_with_list(data)
        else:
            raise Exception('The parameter to update entropy is not of an expected type')

    def _update_with_pandas(self, data):
        global CFG_PARALLEL
        flag_lock = Lock()
        with flag_lock:
            _ = Parallel(n_jobs=CFG_PARALLEL.n_jobs,
                         verbose=CFG_PARALLEL.verbose,
                         backend=CFG_PARALLEL.backend,
                         prefer=CFG_PARALLEL.prefer,
                         require=CFG_PARALLEL.require
                         )(
                            delayed(self._wrapper_update)(e)
                            for e in data.to_dict(orient='records')
                            )

    def _update_with_dict(self, feature_and_value):
        global CFG_PARALLEL
        flag_lock = Lock()
        with flag_lock:
            _ = Parallel(n_jobs=CFG_PARALLEL.n_jobs,
                         verbose=CFG_PARALLEL.verbose,
                         backend=CFG_PARALLEL.backend,
                         prefer=CFG_PARALLEL.prefer,
                         require=CFG_PARALLEL.require
                         )(
                             delayed(self._update_one)(k, v)
                             for k, v in feature_and_value.items()
                             )

    def _update_with_list(self, data):
        global CFG_PARALLEL
        flag_lock = Lock()
        with flag_lock:
            _ = Parallel(n_jobs=CFG_PARALLEL.n_jobs,
                         verbose=CFG_PARALLEL.verbose,
                         backend=CFG_PARALLEL.backend,
                         prefer=CFG_PARALLEL.prefer,
                         require=CFG_PARALLEL.require)(
                             delayed(self._update_one)(n, data[i])
                             for i, n in enumerate(self._features_name))

    def _update_one(self, feature_name, feature_value):
        self._feature_entropy[feature_name].update(feature_value)

    def get_entropy(self, feature_name='all'):
        if feature_name.lower() == 'all':
            return dict([(fname, self._feature_entropy[fname].entropy)
                         for fname in self._features_name])
        # else return one entropy from feature choosed
        return self._feature_entropy[feature_name].entropy

    def get_raw_data_counter(self, feature_name, type_return='dict'):
        dd = self._feature_entropy[feature_name].counter
        type_return = type_return.lower()
        if type_return == 'dict':
            return dict(dd.counter)
        if type_return == 'tuple':
            return list(zip(dd.counter.keys(), dd.counter.values()))
        # else return an object Counter python
        return dd.counter



class DivergenceMeasures:
    def __init__(self):
        pass

    @staticmethod
    def calculate_entropy(window):
        entropies = np.array([entropy(np.bincount(window[:, i]) / len(window)) for i in range(window.shape[1])])
        return entropies

    @staticmethod
    def kl_divergence(p, q):
        return np.sum(np.where(p != 0, p * np.log2(p / q), 0))

    @staticmethod
    def js_divergence(p, q):
        m = 0.5 * (p + q)
        return 0.5 * (DivergenceMeasures.kl_divergence(p, m) + DivergenceMeasures.kl_divergence(q, m))

    @staticmethod
    def hellinger_distance(p, q):
        return np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q))**2))

    @staticmethod
    def total_variation(p, q):
        return 0.5 * np.sum(np.abs(p - q))

    @staticmethod
    def chi_square_divergence(p, q):
        return 0.5 * np.sum(((p - q)**2) / (p + q))


# -------------------- Functions --------------------

def get_sample_data(X, y, seed=100, bootstrap_size=None,
                    feature_size='auto', sample_feature='sqrt+',
                    sample_strategy='numpy.random.Generator.integers'):
    """
    Until function.

    Making random sample.
    """
    flag_lock = Lock()
    with flag_lock:
        n_sample, n_features = np.shape(X)
        
        #print('N Sample, N Features: ', n_sample, n_features)
        #print('Type x: ', type(X))
        
        if bootstrap_size is None:
            bootstrap_size = n_sample

        #print('Bootstrap size: ', bootstrap_size)

        #size_instances = calc_bootstrap_size(bootstrap_size, feature_size)
        #size_features = get_sample_n_feature(n_features, sample_feature)

        #print('Size instances: ', size_instances,
        #      'Size features: ', size_features)

        #chosen_features = list(range(n_features))
        #if (size_features+1) < n_features:
        #    chosen_features = np.random.choice(chosen_features,
        #                                      size=size_features,
        #                                       replace=False)
        #print('Choosen features: ', chosen_features)
        
        bootstrap_instance_size = calc_bootstrap_size(bootstrap_size,
                                                      feature_size)
        bootstrap_feature_size = get_sample_n_feature(n_features,
                                                      sample_feature)
        if (bootstrap_feature_size+1) <= n_features:
            chosen_features = random.sample(population=list(range(n_features)),
                                            k=feature_size)
        else:
            raise ValueError('Number of selected features exceeds dataset size')
        chosen_instances = generate_sample_indices(seed,
                                                   n_sample,
                                                   bootstrap_instance_size,
                                                   sample_strategy=sample_strategy)
        X_sample = np.take(np.take(X, chosen_instances, axis=0),
                           chosen_features, axis=1)
        y_sample = np.take(y, chosen_instances, axis=0)

    return (X_sample, y_sample, chosen_features, chosen_instances)


# The above code is not doing anything. It appears to be a placeholder or
# # a comment indicating that a sample strategy is being implemented.
def deprecated_generate_sample_indices(seed, n_samples, n_samples_bootstrap,
                                       sample_strategy='numpy.random.Generator.integers',
                                       sample_weight=None, replace=True,
                                       probabilities=None, endpoint=False,
                                       axis=0, shuffle=True):
    """
    Until function.

    Generate random sample acoodily to strategy
    """
    sample_indices = []
    if sample_strategy == 'numpy.random.RandomState.randint':
        random_state = np.random.RandomState(seed)
        sample_indices = random_state.randint(0, n_samples,
                                              n_samples_bootstrap)
    elif sample_strategy == 'numpy.random.RandomState.choice':
        random_state = np.random.RandomState(seed)
        a = list(range(n_samples))
        sample_indices = random_state.choice(a=a, size=n_samples_bootstrap,
                                             replace=replace,
                                             p=probabilities)
    elif sample_strategy == 'numpy.random.Generator.integers':
        rng = np.random.default_rng(seed)
        sample_indices = rng.integers(low=0, high=n_samples,
                                      size=n_samples_bootstrap,
                                      endpoint=endpoint)
    elif sample_strategy == 'numpy.random.Generator.choice':
        rng = np.random.default_rng(seed)
        a = list(range(n_samples))
        sample_indices = rng.choice(a=a, size=n_samples_bootstrap,
                                    replace=replace, p=probabilities,
                                    axis=axis, shuffle=shuffle)
    elif sample_strategy == 'random.randrange':
        random.seed(seed)
        lim = n_samples_bootstrap
        sample_indices = [random.randrange(0, n_samples) for _ in range(lim)]
    elif sample_strategy == 'random.choices':
        # High recommend for reprodutibily
        random.seed(seed)
        a = list(range(n_samples))
        sample_indices = random.choices(population=a, weights=sample_weight,
                                        k=n_samples_bootstrap)
    elif sample_strategy == 'random.sample':
        random.seed(seed)
        a = list(range(n_samples))
        sample_indices = random.sample(a, k=n_samples_bootstrap)
    elif sample_strategy == 'secrets.choice':
        a = list(range(n_samples))
        sample_indices = [secrets.choice(a) for _ in range(n_samples_bootstrap)]
    elif sample_strategy == 'secrets.randbelow':
        lim = n_samples_bootstrap
        sample_indices = [secrets.randbelow(n_samples) for _ in range(lim)]
    else:
        raise ValueError('Random sampling strategy unrecognized')

    return sample_indices


def generate_sample_indices(seed, n_samples, n_samples_bootstrap,
                            sample_strategy='numpy.random.Generator.integers',
                            sample_weight=None, replace=True,
                            probabilities=None, axis=0,
                            shuffle=True) -> List[int]:
    sample_indices = []

    # Todo: Implement use the OS Random Generate from random package
    strategies = {
        'numpy.random.RandomState.randint':
            np.random.RandomState(seed).randint,
        'numpy.random.RandomState.choice':
            np.random.RandomState(seed).choice,
        'numpy.random.Generator.integers':
            np.random.default_rng(seed).integers,
        'numpy.random.Generator.choice':
            np.random.default_rng(seed).choice,
        'random.randrange':
            lambda lim: [random.Random(seed).randrange(0, n_samples)
                         for _ in range(lim)],
        'random.choices':
            random.Random(seed).choices,
        'random.sample':
            random.Random(seed).sample,
        'secrets.choice':
            lambda a: [secrets.choice(a)
                       for _ in range(n_samples_bootstrap)],
        'secrets.randbelow':
            lambda lim: [secrets.randbelow(n_samples)
                         for _ in range(lim)],
    }

    args = {
        'numpy.random.RandomState.randint': (0, n_samples, n_samples_bootstrap),
        'numpy.random.RandomState.choice': (list(range(n_samples)),
                                            n_samples_bootstrap, replace,
                                            probabilities),
        'numpy.random.Generator.integers': (0, n_samples, n_samples_bootstrap),
        'numpy.random.Generator.choice': (list(range(n_samples)),
                                          n_samples_bootstrap, replace,
                                          probabilities, axis, shuffle),
        'random.randrange': (n_samples_bootstrap),
        'random.choices': (list(range(n_samples)), sample_weight,
                           n_samples_bootstrap),
        'random.sample': (list(range(n_samples)), n_samples_bootstrap),
        'secrets.choice': (list(range(n_samples))),
        'secrets.randbelow': (n_samples_bootstrap),
    }

    if sample_strategy in strategies:
        sample_indices = strategies[sample_strategy](*args[sample_strategy])
    else:
        raise ValueError('Random sampling strategy unrecognized')

    return sample_indices


def get_sample_n_feature(n_features, sample_n_feature='sqrt+') -> int:
    sample_n_feature_mapping = {
        'sqrt-': int(np.sqrt(n_features)),
        'sqrt+': int(np.sqrt(n_features) + 0.5),
        'log2+': int(np.log2(n_features) + 0.5),
        'log2-': int(np.log2(n_features)),
        'log10+': int(np.log10(n_features) + 0.5),
        'log10-': int(np.log10(n_features)),
        'auto': n_features,
    }

    if isinstance(sample_n_feature, (int, float)):
        return int(sample_n_feature * n_features)
    return sample_n_feature_mapping.get(sample_n_feature, None)


def deprecated_get_sample_n_feature(n_features, sample_n_feature='sqrt-'):
    """
        Until function.

        Return the number of features to training.
    """
    if sample_n_feature == 'sqrt-':
        return int(np.sqrt(n_features))
    if sample_n_feature == 'sqrt+':
        return int(np.sqrt(n_features)+0.5)
    if sample_n_feature == 'log2+':
        return int(np.log2(n_features)+0.5)
    if sample_n_feature == 'log2-':
        return int(np.log2(n_features))
    if sample_n_feature == 'log10+':
        return int(np.log10(n_features)+0.5)
    if sample_n_feature == 'log10-':
        return int(np.log10(n_features))
    if sample_n_feature == 'auto':
        return n_features
    if isinstance(sample_n_feature, int):
        return sample_n_feature
    if isinstance(sample_n_feature, float):
        return int(sample_n_feature * n_features)
    raise ValueError('Impossible determine to sample n feature.')


def calc_bootstrap_size(size, sample_bootstrap_size='auto'):
    """
    Until function.

    Return the size of boostrap sample.
    """
    sample_bootstrap_size = str(sample_bootstrap_size).lower()
    if sample_bootstrap_size == 'auto':
        return size
    if isinstance(sample_bootstrap_size, int):
        return sample_bootstrap_size
    if isinstance(sample_bootstrap_size, float):
        return int(size*sample_bootstrap_size)
    raise ValueError('Impossible determine to size bootstrap.')