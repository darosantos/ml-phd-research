#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Dom Dez 29 16:55:08 2023

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

FOLDER_ROOT = str(os.getcwd()).replace('test', '')
sys.path.append(FOLDER_ROOT)

from stats import DataStreamMonitorEntropy


# cria um dataset para testes
print("Criando o dataset para testes...\n")
list_animal = []
for animal, num_val in zip(['cachorro', 'gato', 'passaro'],[301, 401, 601]):
    list_animal += [animal for i in range(num_val)]
random.shuffle(list_animal)

list_nome = []
for nome, num_val in zip(['joão', 'paulo', 'pedro'],[201, 501, 601]):
    list_nome += [nome for i in range(num_val)]
random.shuffle(list_nome)

list_veiculo = []
for veiculo, num_val in zip(['carro', 'moto', 'bicicleta'],[401, 201, 701]):
    list_veiculo += [veiculo for i in range(num_val)]
random.shuffle(list_veiculo)

df_test = pd.DataFrame([], columns=['animal','nome','veiculo'])

df_test['animal'] = list_animal
df_test['nome'] = list_nome
df_test['veiculo'] = list_veiculo

print(df_test.head())

# entropia de cada lista
print("Explora a entropia de cada lista...\n")

entro_animal = stats.Entropy(fading_factor=1)
entro_nome = stats.Entropy(fading_factor=1)
entro_veiculo = stats.Entropy(fading_factor=1)

for animal in list_animal:
    entro_animal.update(animal)

print(f'{entro_animal.get():.6f}')

for nome in list_nome:
    entro_nome.update(nome)

print(f'{entro_nome.get():.6f}')

for veiculo in list_veiculo:
    entro_veiculo.update(veiculo)

print(f'{entro_veiculo.get():.6f}')

# testa a classe para encapsular o monitoramento da entropia por atributo
print("\n")
print("Testa a classe DataStreamMonitorEntropy\n")

monitor = DataStreamMonitorEntropy(df_test)

print("\n")
print('Monitor feature: ', monitor.get_features_name())

print("\n")
print('Monitor entropy before: ')
print(monitor.get_entropy())

print("\n")
print('Update entropy interactive method')
for e in df_test.to_dict(orient='records'):
    #print(e)
    monitor.update(e)

print("\n")
print('Monitor entropy after: ')
print(monitor.get_entropy())

print("\n")
print('Monitor reset')
monitor.reset()

print("\n")
print('Monitor entropy before: ')
print(monitor.get_entropy())

print("\n")
print('Monitor update with dataframe')
monitor.update(df_test)

print('Monitor entropy after: ')
print(monitor.get_entropy())