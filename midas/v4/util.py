#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: Danilo Santos

This file contain the several functions for utility application.
"""

##################### Functions #####################
from builtins import ValueError

def get_the_time():
    from datetime import datetime
    
    det = datetime.now()
    
    return det.strftime('%d/%m/%Y %H:%M')

def read_dataset(name, path_to_dataset=''):
  """
  Função para abstrair o procedimento de leitura do conjunto de dados
  """
  import pandas as pd
  return pd.read_csv(path_to_dataset+name, engine='c',
                     low_memory=True, memory_map=True)

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
  raise ValueError('Number of cores requested is beyond the machine limit')



###################### Classes ######################
from builtins import AttributeError

class CfgParallelBackend(object):
  """
  Conteiner for parallell options
  """
  __slots__ = ['n_jobs', 'verbose', 'backend', 'prefer',
               'require',]

  def __init__(self, n_jobs=-1, verbose=0, backend='threading',
               prefer='threads', require=None):
    self.n_jobs = n_jobs
    self.verbose = verbose
    self.backend = backend
    self.prefer = prefer
    self.require = require

  def __del__(self):
    del self.n_jobs
    del self.verbose
    del self.backend
    del self.prefer
    del self.require
  
  def set_parallel_options(self, n_jobs=-1, verbose=0, backend='threading',
                           prefer='threads', require=None):
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.backend = backend
        self.prefer = prefer
        self.require = require

  def get_options(self, name_option):
    if not(hasattr(self, name_option)):
      raise AttributeError('The requested attribute was not found in the object')
    return getattr(self, name_option, '')