FOLDER_ROOT = 'F:\Github\ml-phd-research'
import sys
sys.path.append(FOLDER_ROOT)

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from midas.v4.ensemble import RFClassifier
from midas.v4.lib.ObliqueTree.AndriyMulyar.sklearn_oblique_tree.oblique import ObliqueTree


df_stream = pd.read_csv('https://github.com/scikit-multiflow/streaming-datasets/raw/master/agr_a.csv',
                        engine='c', low_memory=True, memory_map=True)

X = df_stream[df_stream.columns[:-1]]
y = df_stream['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.70, random_state=100)

# param splitter: 'oc1' for stochastic hill climbing, 'cart' for CART multivariate, 'axis_parallel' for traditional.
# 'oc1, axis_parallel' will also consider axis parallel splits when computing best oblique split. Setting 'cart' overrides other options.
# param number_of_restarts: number of times to restart in effort to escape local minimums
# param max_perturbations: number of random vector perturbations
# param random_state: an integer serving as the seed (NOT a numpy random state object)

cfg_base_estimator={'splitter':'oc1',
                    'number_of_restarts':20,
                    'max_perturbations': 5,
                    'random_state':100}

rf = RFClassifier(base_estimator=ObliqueTree,
                  params_estimators=cfg_base_estimator,
                  enable_logger=True)

rf = rf.fit(X_train, y_train)
rf._setup_logger(log_file='./rf.obliquetree.debuging.log')

y_pred = rf.predict(X_test)

print('Acc with splitter = oc1', accuracy_score(y_test, y_pred))

################################
cfg_base_estimator={'splitter':'oc1, axis_parallel',
                    'number_of_restarts':20,
                    'max_perturbations': 5,
                    'random_state':100}

rf = RFClassifier(base_estimator=ObliqueTree,
                  params_estimators=cfg_base_estimator,
                  enable_logger=True)

rf = rf.fit(X_train, y_train)
rf._setup_logger(log_file='./rf.obliquetree.debuging.log')

y_pred = rf.predict(X_test)

print('Acc with splitter = oc1, axis_parallel', accuracy_score(y_test, y_pred))


################################
cfg_base_estimator={'splitter':'cart',
                    'number_of_restarts':20,
                    'max_perturbations': 5,
                    'random_state':100}

rf = RFClassifier(base_estimator=ObliqueTree,
                  params_estimators=cfg_base_estimator,
                  enable_logger=True)

rf = rf.fit(X_train, y_train)
rf._setup_logger(log_file='./rf.obliquetree.debuging.log')

y_pred = rf.predict(X_test)

print('Acc with splitter = cart', accuracy_score(y_test, y_pred))

