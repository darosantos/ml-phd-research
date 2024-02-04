#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fou Nov 15 13:55:36 2023

@author: Danilo Santos
"""
def calc_window_size(n_instances, window_train_size=0.1):
  """
  Função para calcular o tamanho da janela de dados para treinamento
  """
  return int(n_instances * window_train_size)

def sliding_window(X, y, window_size=100, step=1, window_return='full', shadow=1):
  """
  Função para percorrer um dataset usando uma janela de tamanho fixo.
  A janela desliza de forma incremental adicionando o novo elemento ao fim da
  janela e removendo o primeiro (mais antigo).
  O retorno da função depende do parâmetro window_return.
    = "full" -> retorna a janela completa
    = "next" -> retorna somente a nova instância
    = "split" -> retorna a uma tupla separando a janela de memória e a nova instância
  """
  contador = 0
  sliding_window.first_window = getattr(sliding_window, 'first_window', True)
  n_instancias = X.shape[0]
  while True:
    inicio = contador
    fim = contador + (window_size - shadow) + (step - shadow)
    contador = contador + step if contador < n_instancias else n_instancias
    if fim >= n_instancias:
      # Condição de pausa adicionada para corrigir um bug de loop avançando mais do que deveria
      del sliding_window.first_window, contador, inicio, fim
      break
    else:
      if sliding_window.first_window or window_return == 'full':
        sliding_window.first_window = False
        yield (X.iloc[inicio:fim+1, :], y.iloc[inicio:fim+1])
      elif window_return == 'next' and not(sliding_window.first_window):
        yield (X.iloc[[fim], :],  y.iloc[fim])
      elif window_return == 'split' and not(sliding_window.first_window):
        yield ((X.iloc[[fim], :], X.iloc[inicio:fim+1, :]),  (y.iloc[fim], y.iloc[inicio:fim+1]))
      else:
        raise StopIteration