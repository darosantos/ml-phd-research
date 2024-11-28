from .v1.ktree import KTreeClassifier as KTv1

from joblib import Parallel, delayed

import numpy as np


__version__ = (0,0,3)
if __name__ == "__main__":
  print('KTree Version: ', __version__)
  

class KTreeClassifier(KTv1):
  """K Tree Classifier V2"""

  def __init__(self,
               ensemble_base,
               k_tree=25,
               strategy="accuracy_mean",
               voting="majority",
               n_jobs=-1,
               verbose=0,
               check_input_predict=True
              ):
    super().__init__(ensemble_base, k_tree, strategy, voting, n_jobs, verbose, check_input_predict)
    self.utilities = {'exec_predict_incremental': None}
    self.utilities['pair_fit_score'] = lambda est_index, estimator, X_true, y_true: (est_index, estimator.score(X_true, y_true))
    self.utilities['ensemble_predict_majority'] = lambda i_est, X: self.ensemble.estimators_[i_est].predict(X)
    self.utilities['arr_size'] = lambda arr: arr.shape[0] if ('shape' in dir(arr)) else (len(arr) if (arr is list) else -1)
    self.utilities['begin_slice'] = lambda yi, current: 0 if current < self.utilities['arr_size'](yi) else ((current - self.utilities['arr_size'](yi))+1)
    self.utilities['exec_predict_incremental'] = lambda obj, Xi, Xm, yi: obj._predict_update(Xi, Xm, yi)
    self.utilities['util_reshape'] = lambda X, col=1: X.reshape(1,-1) if len(X.shape) == 1 else X.reshape(-1, col)
    
  def predict_incremental(self, X, y_update, partial_fit=None):
    """
    X -> unknow instances
    y_update -> array of array with trues labels for each Xi
    partial_fit -> tuple with index for step by step predict
    """
    # classifica uma instância e se auto atualiza
    if y_update is None:
      return self.predict(X) # não tem como atualizar as arvores sem ter o rotulo verdadeiro então faz só o predict
    
    if partial_fit is None: # add future check_partial_fit_parameter
      partial_fit = (0, X.shape[0]) #default value
    else:
      if (type(partial_fit) is tuple):
        if (len(partial_fit) != 2):
          raise Exception('Partial fit has to be a tuple with two entries')
        if partial_fit[1] > X.shape[0]:
          raise Exception('Partial fit out of range of the shape X')
      else:
        raise Exception('Partial fit has to be a tuple')

    ypred = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, prefer='threads',
                     require='sharedmem'
                     )(delayed(self.utilities['exec_predict_incremental']
                               )(self,self.utilities['util_reshape'](Xi),
                                 self.utilities['util_reshape'](X[self.utilities['begin_slice'](yUi, (partial_fit[0]+i)):(partial_fit[0]+i+1)], X.shape[1]), 
                                 yUi
                                 ) for i, (Xi, yUi) in enumerate(zip(X[partial_fit[0]:partial_fit[1]], 
                                                                     y_update[partial_fit[0]:partial_fit[1]])))
    
    return np.ravel(ypred)
  
  def _predict_update(self, Xi, Xm=None, yi=None):
    """
    Xi -> unknow instance
    Xm -> memory instance X(i-len(yi), i)
    yi -> label true for Xm
    """
    y_pred = self.predict(Xi)
    if not(yi is None): # update k tree
      self.fit_update(Xm,  yi)
    return y_pred
  
  def utility_slice_predict_memory(y_true, slide_memory=100):
    """
    y_true -> array of labels of X_test
    slide_memory -> number of old elements from X_test
    """
    y_memory = []
    inicio = 0
    fim = 0

    for i in range(y_true.shape[0]):
      if i < slide_memory:
        inicio = i // slide_memory
      else:
        inicio = (i - slide_memory) + 1
      fim = i + 1
      y_memory.append(y_test[inicio:fim])

    return np.asarray(y_memory, dtype=object)