from sklearn.base import BaseEstimator
from sklearn.metrics import cohen_kappa_score, f1_score, balanced_accuracy_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score

from joblib import Parallel, delayed

import numpy as np

if __name__ == "__main__":
  print('KTree Version: ', __version__)
  
  
  
class KTreeClassifier(BaseEstimator):
  """K Tree Classifier"""

  def __init__(self,
               ensemble_base,
               k_tree=25,
               strategy="accuracy_mean",
               voting="majority",
               n_jobs=-1,
               verbose=0,
               check_input_predict=True
              ):
    self.ensemble = ensemble_base
    self.n_tree = k_tree
    self.index_k_tree = []
    self.fit_scores_ = []
    self.strategy = str(strategy).lower()
    self.voting = str(voting).lower()
    self.n_jobs = n_jobs
    self.verbose = verbose
    self.check_input_predict = check_input_predict
    
    if isinstance(k_tree, float):
      self.n_tree = int(k_tree*self.get_ensemble_size())
    
    # Utilities lambda functions
    cip = self.check_input_predict
    self.utilities = {'fit_am': None, 
                      'fit_cks': None,
                      'ensemble_predict_majority': None, 
                      'maj_voting': None}
    self.utilities['fit_am'] = lambda index, est, X, y: (index, est.score(X, y))
    self.utilities['fit_cks'] = lambda index, est, X, y: (index, 
                                                          cohen_kappa_score(y, 
                                                                            est.predict(X,check_input=cip)
                                                                           )
                                                         )
    self.utilities['fit_f1'] = lambda index, est, X, y: (index, 
                                                         f1_score(y, 
                                                                  est.predict(X, check_input=cip)
                                                                 )
                                                        )
    self.utilities['fit_bas'] = lambda index, est, X, y: (index, 
                                                          balanced_accuracy_score(y, 
                                                                                  est.predict(X, check_input=cip)
                                                                                 )
                                                         )
    self.utilities['fit_ps'] = lambda index, est, X, y: (index, 
                                                         precision_score(y, 
                                                                         est.predict(X, check_input=cip)
                                                                        )
                                                        )
    self.utilities['fit_rcs'] = lambda index, est, X, y: (index, 
                                                         recall_score(y, 
                                                                      est.predict(X, check_input=cip)
                                                                     )
                                                         )
    self.utilities['fit_ras'] = lambda index, est, X, y: (index, 
                                                         roc_auc_score(y, 
                                                                       est.predict(X, check_input=cip)
                                                                      )
                                                         )
    self.utilities['ensemble_predict_majority'] = lambda i_est, X: self.ensemble.estimators_[i_est].predict(X, check_input=cip)
    self.utilities['maj_voting'] = lambda x: np.bincount(x).argmax()

  def fit(self, X, y):
    """
    X -> list of features values
    y -> label for each X(i)
    """
    if len(self.index_k_tree) != 0:
      raise Exception("Classifier is fitted!")
      
    if self.strategy == "accuracy_mean":
      self._fit_score(X, y, metric=self.utilities['fit_am'])
    elif self.strategy == "cohen_kappa":
      self._fit_score(X, y, metric=self.utilities['fit_cks'])
    elif self.strategy == "f1_score":
      self._fit_score(X, y, metric=self.utilities['fit_f1'])
    elif self.strategy == "balanced_accuracy_score":
      self._fit_score(X, y, metric=self.utilities['fit_bas'])
    elif self.strategy == "precision_score":
      self._fit_score(X, y, metric=self.utilities['fit_ps'])
    elif self.strategy == "recall_score":
      self._fit_score(X, y, metric=self.utilities['fit_rcs'])
    elif self.strategy == "roc_auc_score":
      self._fit_score(X, y, metric=self.utilities['fit_ras'])
    else:
      raise Exception('Strategy not is empty')
      
    return self
  
  def fit_update(self, X, y):
    """
    X -> list of features values
    y -> label for each X(i)
    """
    if self.index_k_tree.sum() == 0:
      raise Exception("Classifier isn't fitted")
      
    if self.strategy == "accuracy_mean":
      self._fit_score(X, y, metric=self.utilities['fit_am'])
    elif self.strategy == "cohen_kappa":
      self._fit_score(X, y, metric=self.utilities['fit_cks'])
    elif self.strategy == "f1_score":
      self._fit_score(X, y, metric=self.utilities['fit_f1'])
    elif self.strategy == "balanced_accuracy_score":
      self._fit_score(X, y, metric=self.utilities['fit_bas'])
    elif self.strategy == "precision_score":
      self._fit_score(X, y, metric=self.utilities['fit_ps'])
    elif self.strategy == "recall_score":
      self._fit_score(X, y, metric=self.utilities['fit_rcs'])
    elif self.strategy == "roc_auc_score":
      self._fit_score(X, y, metric=self.utilities['fit_ras'])
    else:
      raise Exception('Strategy not is empty')
      
    return self

  def _fit_score(self, X_true, y_true, metric):
    """
    X_true -> 
    y_true ->
    metric -> metric for evalute of the list classifiers
    """
    lst_scores = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, 
                          require='sharedmem'
                          )(delayed(metric
                                    )(xi, estimator, X_true, y_true
                                      ) for xi, estimator in enumerate(self.ensemble.estimators_))

    ordered_scores = sorted(lst_scores,
                            key=lambda position: position[1], reverse=True)

    self.index_k_tree = np.array([ik[0] for ik in ordered_scores[:self.n_tree]],
                                 dtype=np.int64)

    self.fit_scores_ = np.array([ik[1] for ik in ordered_scores[:self.n_tree]],
                                dtype=np.float64)
    
    del lst_scores, ordered_scores
  
  def predict(self, X):
    """
    X -> unknow instances
    """
    if self.voting == "majority":
      return self._predict_majority(X)
    else:
      raise Exception('Method of voting required')
  
  def _predict_majority(self, X):
    """
    X -> unknow instances
    """
    predictions = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                           require='sharedmem'
                           )(delayed(self.utilities['ensemble_predict_majority']
                                     )(Ei, X) for Ei in self.index_k_tree)
    predictions = np.array(predictions, dtype=np.int64)
    predictions = predictions.T

    maj = np.apply_along_axis(self.utilities['maj_voting'], axis=1, arr=predictions)
    
    del predictions
    return maj
  
  def get_ensemble_size():
    """
    None
    """
    return len(self.ensemble.estimators_)
