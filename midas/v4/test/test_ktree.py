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
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

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
kte = KTreeClassifier(n_estimators=2)

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
print(kte.ensemble_[1])

print('Vamos decodificaro feature name:')
print(kte.feature_names_)

print('Atributos que treinaram o KTE 5')
for i in kte.ensemble_[1]['feature_name']:
    print('Atributo: ', i, ' = ', kte.feature_names_[i])

print('Testando o match:')
print(kte.match_features(1))

print("Vamos testar o predict")

y_pred = kte.predict(X_test)

print('Essa são as 10 primeiras amostras de predição')
print(y_pred[:10])
print('Esse é os 10 primeiros y_test')
print(list(y_test[:10]))

print('Acurácia: ', accuracy_score(y_test, y_pred))

print('Vamos ver o Random Forest Skelearn')

rf = RandomForestClassifier(random_state=40)
rf = rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print('Essa são as 10 primeiras amostras de predição RF')
print(y_pred[:10])

print('Acurácia: ', accuracy_score(y_test, y_pred))