#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on ter jan 23 09:51:08 2023

@author: Danilo Santos
"""

import math
import random
import numpy as np
from scipy.stats import entropy
from river import stats

import pandas as pd

import sys
import os

RANDOM_STATE_SEED_ = 100

FOLDER_ROOT = str(os.getcwd()).replace('test', '')
sys.path.append(FOLDER_ROOT)

#pd.options.compute.use_bottleneck = True
pd.options.compute.use_numba = True
pd.options.compute.use_numexpr = True
pd.options.display.memory_usage = True
pd.options.display.precision = 3
pd.options.mode.use_inf_as_na = True

np.random.seed(RANDOM_STATE_SEED_)
np.set_printoptions(precision=3)

from ktree import KTreeClassifier

print('Teste de instancia do KTreeClassifier')
kte = KTreeClassifier()

print('Ossos do Ktree objeto')
print(dir(kte))

print("\n")
print("\n")
print("\n")
print('Teste de treino do KTree')
print('Vamos preparar o dataset')


dataset_file = 'dados/covertype.csv'
wz = 1000
lm = 3000
df = pd.read_csv(dataset_file, engine='c', low_memory=True, memory_map=True)

X_train = df.iloc[:wz,:-1]
y_train = df.iloc[:wz, -1]
X_test = df.iloc[wz:wz+lm, :-1]
y_test = df.iloc[wz:wz+lm, -1]

print('Features de treinamento: ', X_train.columns)

print("Acessar os atributos do KTE")
print('Seed: ', kte.seed)
print('Base Estimator: ', kte.base_estimator)

print('Vamos chamar fit')

kte = kte.fit(X_train, y_train)

print('Acabamos o treino')

print('Tamanho do ensemble: ', len(kte.ensemble_))

print('Acessando um elemento do ensemble: ')
print(kte.ensemble_[5])

print('Vamos decodificaro feature name:')
print(kte.feature_names_)

print('Atributos que treinaram o KTE 5')
for i in kte.ensemble_[5]['feature_name']:
    print('Atributo: ', i, ' = ', kte.feature_names_[i])

print('Testando o match:')
print(kte.match_features(5))