#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: Danilo Santos

This file contain the several functions for utility application.
"""

from builtins import AttributeError
from builtins import ValueError, NameError
from threading import Lock
import gc
import pandas as pd
import numpy as np

if not (gc.isenabled()):
    gc.enable()
    

# -------------------- Functions --------------------


def get_the_time():
    from datetime import datetime

    det = datetime.now()

    return det.strftime("%d/%m/%Y %H:%M")


def read_dataset(name, path_to_dataset=""):
    """
    Função para abstrair o procedimento de leitura do conjunto de dados
    """
    return pd.read_csv(
        path_to_dataset + name, engine="c", low_memory=True, memory_map=True
    )


def get_n_jobs(parallel_n_jobs=-1):
    """
    Util function.

    Util for to return get the number of CPUs in the system or
    the number choose from user.
    parameters: None
    """
    from os import cpu_count

    c = cpu_count()
    if parallel_n_jobs == -1:
        return c
    if parallel_n_jobs <= c:
        return parallel_n_jobs
    raise ValueError("Number of cores requested is beyond the machine limit")


def estimator_is_fitted(estimator):
    """
    Until function.

    Return boolean if this classifier is fitted.
    """
    return len(estimator.ensemble_) > 0


def adapt_massive_inputs(estimator, X, y=None):
    """
    Until function.

    Return the inputs adapteds.
    """
    flag_lock = Lock()
    flag_lock.acquire()
    Xt, yt = None, None
    if not (estimator_is_fitted(estimator)):
        if isinstance(X, pd.core.frame.DataFrame):
            estimator.feature_names_ = X.columns
            estimator.n_features_ = X.shape[1]
            Xt = X.to_numpy(copy=True)
        else:
            estimator.feature_names_ = None
            estimator.n_features_ = np.shape(X)[1]
            Xt = np.array(X)

        if not (y is None):
            estimator.classes_ = np.unique(y)
            estimator.n_classes_ = len(estimator.classes_)
            if isinstance(y, pd.core.series.Series):
                yt = y.to_numpy(copy=True)
            else:
                if not (isinstance(y, np.ndarray)):
                    yt = np.array(y)
    else:
        # verificar se é o mesmo x
        if np.shape(X)[1] != estimator.n_features_:
            raise ValueError("X unknown size feature")
        if isinstance(X, pd.core.frame.DataFrame) and all(
            X.columns != estimator.feature_names_
        ):
            raise ValueError("X have unknown features.")
        Xt = np.array(X)
        yt = np.array(y)

    flag_lock.release()

    if y is None:
        return Xt

    return (Xt, yt)


def check_inputs(self, X, y):
    """
    Until function.

    Check data input.
    """
    # verificar se é matriz
    # verificar se tem tamanho igual de x e y
    flag_lock = Lock()
    flag_lock.acquire()
    if np.shape(X)[1] < 1:
        raise ValueError("X must be an NxM array.")
    if y is None:
        raise ValueError("Target label is expected.")
    if np.shape(X)[0] != np.shape(y)[0]:
        raise ValueError("X and y must be the same size.")
    flag_lock.release()

    return (X, y)


def get_instance_of_class(class_name, module_name=None, *args, **kwargs):
    """
    get_class function utility.

    Args:
        class_name (str): the name class will imported

    Returns:
        obj: a instance of class
    """
    # return eval(class_name)
    # return globals()[class_name]
    import importlib
    if not (module_name is None):
        module = importlib.import_module(module_name)
        return getattr(module, class_name)(**kwargs)

    if class_name in globals():
        return globals()[class_name](*args, **kwargs)

    return None


def call_method_from_instance(obj_instance, method_to_call,
                              *args_to_method, **kwargs):
    if method_to_call in dir(obj_instance):
        return getattr(obj_instance, method_to_call)(*args_to_method, **kwargs)
    raise NameError('Method not found in instantiated object')


# -------------------- Classes --------------------

class CfgParallelBackend(object):
    """
    Conteiner for parallell options
    """

    __slots__ = [
        "n_jobs",
        "backend",
        "return_as",
        "prefer",
        "require",
        "verbose",
        "timeout",
        "pre_dispatch",
        "batch_size",
        "temp_folder",
        "max_nbytes",
        "mmap_mode"
    ]

    def __init__(
        self, n_jobs=-1, backend="threading", return_as="list",
        prefer="threads", require=None, verbose=0, timeout=None,
        pre_dispatch='2*n_jobs', batch_size='auto', temp_folder=None,
        max_nbytes='1M', mmap_mode='r'
    ):
        self.n_jobs = get_n_jobs(n_jobs)
        # ["loky", "multiprocessing", "threading"]
        self.backend = backend
        # [‘list’, ‘generator’, ‘generator_unordered’]
        self.return_as = return_as
        # [‘processes’, ‘threads’, None]
        self.prefer = prefer
        # [‘sharedmem’, None]
        self.require = require
        self.verbose = verbose
        self.timeout = timeout
        # ‘all’, integer, or expression, as in ‘3*n_jobs’
        self.pre_dispatch = pre_dispatch
        self.batch_size = batch_size
        self.temp_folder = temp_folder
        # [int, str, None]
        self.max_nbytes = max_nbytes
        # [None, ‘r+’, ‘r’, ‘w+’, ‘c’]
        self.mmap_mode = mmap_mode

    def __del__(self):
        del self.n_jobs
        del self.verbose
        del self.backend
        del self.prefer
        del self.require

    def set_parallel_options(
        self, n_jobs=-1, backend="threading", return_as="list",
        prefer="threads", require='sharedmem', verbose=0, timeout=None,
        pre_dispatch='all', batch_size='auto', temp_folder='/tmp',
        max_nbytes='1M', mmap_mode='r'
    ):
        self.n_jobs = get_n_jobs(n_jobs)
        self.backend = backend
        self.return_as = return_as
        self.prefer = prefer
        self.require = require
        self.verbose = verbose
        self.timeout = timeout
        self.pre_dispatch = pre_dispatch
        self.batch_size = batch_size
        self.temp_folder = temp_folder
        self.max_nbytes = max_nbytes
        self.mmap_mode = mmap_mode

    def get_options(self, name_option):
        if not (hasattr(self, name_option)):
            raise AttributeError("The requested attribute was not found in the object")
        return getattr(self, name_option, "")

    def get_all_options(self):
        all_ = {"n_jobs": self.n_jobs, "backend": self.backend,
                "return_as": self.return_as, "prefer": self.prefer,
                "require": self.require, "verbose": self.verbose,
                "timeout": self.timeout, "pre_dispatch": self.pre_dispatch,
                "batch_size": self.batch_size, "temp_folder": self.temp_folder,
                "max_nbytes": self.max_nbytes, "mmap_mode": self.mmap_mode}
        return all_
