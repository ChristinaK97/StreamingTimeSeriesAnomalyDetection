# -*- coding: utf-8 -*-
"""Notebook_test_iForestASD_Scikitmultiflow.ipynb
Automatically generated by Colaboratory.
Original file is located at
    https://colab.research.google.com/drive/1APZBgZ0fuHufYWM5QKqFhR7cpPs0bF2T
# iForestASD :  Unsupervised Anomaly Detection with Scikit-MultiFlow
An Implementation of Unsupervised Anomaly Detection with Isolation Forest in Scikit-MultiFlow with Sliding Windows \& drift detection
## References :
 - An Anomaly Detection Approach Based on Isolation Forest  for Streaming Data using Sliding Window (Ding \& Fei, 2013) https://www.sciencedirect.com/science/article/pii/S1474667016314999
 
 - Isolation-based Anomaly Detection (Liu, Ting \& Zhou, 2011) https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/tkdd11.pdf
 - Scikit MultiFlow HalfSpace Trees Implementation - “Fast anomaly detection for streaming data,” in IJCAI Proceedings - S.C.Tan, K.M.Ting, and T.F.Liu, 
 https://scikit-multiflow.github.io/scikit-multiflow/_autosummary/skmultiflow.anomaly_detection.HalfSpaceTrees.html#id2
 - Original implementation of Isolation Forest (not the one in SK-learn) https://github.com/Divya-Bhargavi/isolation-forest
"""



"""# Notebook File Structure  is the following
Part 1 - Main Class contians
  - Init,
  - Partial_fit,
  -  Update_model,
  - Predict methods which use the anomaly_score methods of the iForest class
Part 2 - Isolation Forest class (re-used) and main functions
 - 
Part 3 - Testing some examples and comparison of HS-Trees and IsolatationForestStream 
- on synthetic 
- on Real (HTTP) data.
## Import lib and packages
"""



"""## Install Cyphion then load the Scikit-multiflow latest release from Github"""

## !pip install scikit-multiflow



#!pip install -U git+https://github.com/scikit-multiflow/scikit-multiflow

from skmultiflow.core import BaseSKMObject, ClassifierMixin

from skmultiflow.utils import check_random_state

import numpy as np

import random

#from skmultiflow.drift_detection.adwin import ADWIN


"""# Part 1 - Main class - IsolationForestStream"""

## To implement this class, we took inspiration from Scikit-MultiFLow HSTrees implementation to follow its requirements.

class IsolationForestStream(BaseSKMObject, ClassifierMixin):

  """
  This code implements  Anomaly Detection Approach Based on Isolation Forest Algorithm for Streaming Data Using Sliding Window (Ding \& Fei, 2013) [3]
    Each sample has an anomaly score is computed based on Isolation Forest anomaly based approach [2]. The concept of Isolation forest [1]
    consists on  isolating observations by randomly selecting a feature
    and then randomly selecting a split value between the maximum and minimum
    values of the selected feature.
    
    Model is updated of a Drift has been detected based on a input drift threshold. The drift detection approach is proposed by [2] 
    and works as follow : if the averaged anomaly score between two successive sliding windows is highter than the drift threshold (u), 
    then the previous model is completely discarded and a new model is build as an isolation forest on latest sliding windows stream.
  Parameters
    ---------
    n_estimators: int, optional (default=25)
       Number of trees in the ensemble.
       't' in the original paper.
    window_size: int, optional (default=100)
        The window size of the stream.
        ψ, 'Psi' in the original paper.   
## Optional       
    anomaly_threshold: double, optional (default=0.5)
        The threshold for declaring anomalies.
        Any instance prediction probability above this threshold will be declared as an anomaly.
    drift_threshold: double, optional (default=0.5)
        The threshold for detecting Drift and update the model.
       If the averaged anomaly score between two successive sliding windows is highter than the threshold (u), 
    then the previous model is completely discarded and a new model is build as an isolation forest on latest sliding windows stream.
    This parameters is supposed to be know by an expert domain, depending on data set.
## Other Attributes
    ensemble : Isolation Tree Ensemble
        Contain an Isolation Tree Ensemble object, current model for   IsolationForestStream
    sample_size : int
        Number of sample seen since the update
    anomaly_rate : float
        Rate of the anomalies in the previous sliding window (AnomalyRate in the original paper iForestASD)
    prec_window & window : numpy.ndarray of shape (n_samples, self.window_size)
        The previous and current window of data
    cpt : int
        Counter, if the n_estimator is higher than its, it will fit
    References
    ----------
    [1] Liu, Fei Tony, Ting, Kai Ming and Zhou, Zhi-Hua.        
“Isolation forest.” Data Mining, 2008. ICDM’08. Eighth IEEE International Conference on.
    [2] Liu, Fei Tony, Ting, Kai Ming and Zhou, Zhi-Hua. “Isolation-based anomaly detection.” ACM Transactions on Knowledge Discovery from Data (TKDD) 6.1 (2012): 
self.n_estimators
    [3] Ding, Zhiguo. (2013) An Anomaly Detection Approach Based on Isolation Forest Algorithm for Streaming Data Using Sliding Window. 12-17. 10.3182/20130902-3-CN-3020.00044. 
    """ 

  def __init__(self, window_size=100, n_estimators=25, anomaly_threshold=0.5, 
               drift_threshold=0.5, random_state=None, version="AnomalyRate",
               #Parameters for partial model update
               n_estimators_updated=0.5, updated_randomly=True,
               #Parameters for NDKSWIN
               alpha=0.01, data=None, n_dimensions=1, n_tested_samples=0.1,
               fixed_checked_dimension = False, fixed_checked_sample=False):
    
        super().__init__()

        self.n_estimators = n_estimators

        self.ensemble = None
      
        self.random_state = random_state

        self.window_size = window_size

        self.samples_seen = 0

        self.anomaly_rate = 0.30

        self.anomaly_threshold = anomaly_threshold

        self.drift_threshold = drift_threshold

        self.window = None

        self.prec_window = None

        self.cpt = 0
        self.version = version
        self.model_update = [] #To count the number of times the model have been updated 0 Not updated and 1 updated
        self.model_update_windows = [] #To count the number of times the model have been updated 0 Not updated and 1 updated
        self.model_update.append(version) #Initialisation to know the concerned version of IForestASD
        self.model_update_windows.append("samples_seen_"+version) #Initialisation to know the number of data seen in the window
        self.n_estimators_updated=int(self.n_estimators*n_estimators_updated) # The percentage of new trees to compute when update on new window
        if n_estimators_updated <= 0.0 or n_estimators_updated > 1.0 :
            raise ValueError("n_estimators_updated must be > 0 and <= 1")
            
        self.updated_randomly=updated_randomly # If we will choose randomly the trees: True for randomly, 
                        # False to pick the first (n_estimators- int(n_estimators*n_estimators_updated)) trees

        self.alpha=alpha
        self.n_dimensions=n_dimensions
        self.n_tested_samples=n_tested_samples
        self.fixed_checked_dimension =fixed_checked_dimension
        self.fixed_checked_sample=fixed_checked_sample
        self.first_time_fit = True
        
        # TODO Maurras 27112020: Find a way to optimize the use of ADWIN()
        #self.adwin = ADWIN()
        
  def partial_fit(self, X, y, classes=None, sample_weight=None):

          """ Partially (incrementally) fit the model.
          Parameters
          ----------
          X : numpy.ndarray of shape (n_samples, n_features)
              The features to train the model.
          y: numpy.ndarray of shape (n_samples)
              An array-like with the class labels of all samples in X.
          classes: None
              Not used by this method.
          sample_weight: None
              Not used by this method.
          Returns
          -------
          self
          """

          ## get the number of observations
          number_instances, _ = X.shape
          
          if(self.samples_seen==0):
            ## ToDo ? Give a sample of self.window_size in attribute of iForest
            iforest = IsolationTreeEnsemble(self.window_size,self.n_estimators,self.random_state)
            self.ensemble = iforest

          for i in range(number_instances):
              self._partial_fit(X[i], y[i])

          return self


  def _partial_fit(self, X, y):

          """ Trains the model on samples X and corresponding targets y.
          Private function where actual training is carried on.
          Parameters
          ----------
          X: numpy.ndarray of shape (1, n_features)
              Instance attributes.
          y: int
              Class label for sample X. Not used in this implementaion which is Unsupervised
          """ 
          
          """
          Reshape X and add it to our window if it isn't full.
          If it's full, give window to our precedent_window.
          If we are at the end our window, fit if we're learning 
          Check the anomaly score of our window 
          Update if self.anomaly_rate > self.drift_threshold
          """
          X = np.reshape(X,(1,len(X)))

          if self.samples_seen % self.window_size == 0:
            ## Update the two windows (precedent one and current windows)
            self.prec_window = self.window
            self.window = X
          else:
            self.window = np.concatenate((self.window,X))
          
          if self.samples_seen % self.window_size == 0 and self.samples_seen !=0:
              
              #Fit the ensemble if it's not empty
              #if(self.cpt<self.n_estimators):
              #  self.ensemble.fit(self.prec_window)
              #  self.cpt += 1  
              if self.first_time_fit: #It is the first window 
                  self.ensemble.fit(self.prec_window)
                  self.first_time_fit = False

              if(self.version == "AnomalyRate"):
                  #print('start AnomalyRate version')
                  ## Update the current anomaly score
                  self.anomaly_rate = self.anomaly_scores_rate(self.prec_window) ## Anomaly rate
                  #print(self.anomaly_rate) ## 
                  ## Update the model if the anomaly rate is greater than the threshold (u in the original paper [3])
                  if self.anomaly_rate > self.drift_threshold: ## Use Anomaly RATE ?
                    self.model_update.append(1)
                    self.model_update_windows.append(self.samples_seen)
                    self.update_model(self.prec_window) # This function will discard completly the old model and create a new one
                  else:
                      self.model_update.append(0)
                      self.model_update_windows.append(self.samples_seen)
                        
          self.samples_seen += 1
  
          
  def update_model(self,window):
    """ Update the model (fit a new isolation forest) if the current anomaly rate (in the previous sliding window)
     is higher than self.drift_threshold
        Parameters: 
          window: numpy.ndarray of shape (self.window_size, n_features)
        Re-Initialize our attributes and our ensemble, fit with the current window
    """

    ## ToDo ? Give a sample of self.window_size in attribute of iForest
    #MAJ Maurras 03112020 : No, Leave it like that. Must give all the window to tt construct the forest of itrees.
    self.is_learning_phase_on = True
    iforest = IsolationTreeEnsemble(self.window_size,self.n_estimators,self.random_state)
    self.ensemble = iforest
    self.ensemble.fit(window)
    #self.nb_update = self.nb_update + 1
    print("")
    print("The model was updated by training a new iForest with the version : "+self.version)
    
          
  def anomaly_scores_rate(self, window):
    """
    Given a 2D matrix of observations, compute the anomaly rate 
    for all instances in the window and return an anomaly rate of the given window.
    Parameters :
    window: numpy.ndarray of shape (self.window_size, n_features)
    """

    score_tab = 2.0 ** (-1.0 * self.ensemble.path_length(window) / c(len(window)))
    score = 0
    for x in score_tab:
      if x > self.anomaly_threshold:
        score += 1
    return score / len(score_tab)
  
  '''
      MAJ : 21112020
      By : Maurras
      Add new function to classify instances (anomaly or normal)
  ''' 
  def predict_simple(self, X):
    """
    Given a window, Predict the instance class (1 or 0) by using predict_from_instances_scores on our model
    """
    #print('predict_simple')
    prediction =  self.ensemble.predict_from_instances_scores(self.ensemble.anomaly_score(X),
                                                            self.anomaly_threshold) ## return prediction of all instances

    #print('end predict_simple')
    return prediction
    
  def predict(self, X):
    """
    Given an instance, Predict the anomaly (1 or 0) based on the last sample of the window by using predict_proba if our model have fit, 
    else return None
    """
    if(self.samples_seen <= self.window_size):

      return [-1] ## Return the last element

    X = np.reshape(X,(1,len(X[0])))
    self.prec_window = np.concatenate((self.prec_window ,X)) ## Append the instances in the sliding window

    prediction =  self.ensemble.predict_from_anomaly_scores(self.predict_proba(self.prec_window),self.anomaly_threshold) ## return 0 or 1

    return [prediction]
              
  def predict_proba(self, X):
    """
    Calculate the anomaly score of the window if our model have fit, else return None
    Parameters :
    X: numpy.ndarray of shape (self.window_size, n_features)   
    """
    if(self.samples_seen <= self.window_size):
        return [-1]
    return self.ensemble.anomaly_score(self.prec_window)[-1] # Anomaly return an array with all scores of each data, taking -1 return the last instance (X) anomaly score

"""# Part 2- IsolationTreeEnsemble  Class (iForest in the original paper)"""

# Follows original paper algo from https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf
# Original Source re-used and adpted to our project from https://github.com/Divya-Bhargavi/isolation-forest 
class IsolationTreeEnsemble:
    def __init__(self, sample_size, n_trees, random_state):
        self.sample_size = sample_size
        self.n_trees = n_trees
        self.depth = np.log2(sample_size)
        self.trees = []
        self.random_state = random_state
        self._random_state = check_random_state(self.random_state)
        self.is_learning_phase_on = True 

    def fit(self, X:np.ndarray):
        """
        Given a 2D matrix of observations, create an ensemble of IsolationTree
        objects and store them in a list: self.trees.  Convert DataFrames to
        ndarray objects.
        """
        len_x = len(X)

        for i in range(self.n_trees):
            sample_idx = random.sample(list(range(len_x)), self.sample_size )
            temp_tree = IsolationTree(self.depth, 0).fit(X[sample_idx])
            self.trees.append(temp_tree)

        return self
   
    def path_length(self, X:np.ndarray):
        """
        Given a 2D matrix of observations, X, compute the average path length
        for each observation in X.  Compute the path length for x_i using every
        tree in self.trees then compute the average for each x_i.  Return an
        ndarray of shape (len(X),1).
        """
        pl_vector = []

        for x in (X):
            pl = np.array([path_length_tree(x, t, 0) for t in self.trees])
            pl = pl.mean()

            pl_vector.append(pl)

        pl_vector = np.array(pl_vector).reshape(-1, 1)

        return pl_vector

    def anomaly_score(self, X:np.ndarray):
        """
        Given a 2D matrix of observations, X, compute the anomaly score
        for each x_i observation, returning an ndarray of them.
        """
        return 2.0 ** (-1.0 * self.path_length(X) / c(len(X)))

    def predict_from_anomaly_scores(self, scores:int, threshold:float):
        """
        Given an array of scores and a score threshold, return an array of
        the predictions: 1 for any score >= the threshold and 0 otherwise.
        """
        predictions = 1 if scores >= threshold else 0

        return predictions
    
    '''
          MAJ : 21112020
          By : Maurras
          Add new function to classify instances (anomaly or normal)
    ''' 
    def predict_from_instances_scores(self, scores:np.ndarray, threshold:float) -> np.ndarray:
            """
            Given an array of scores and a score threshold, return an array of
            the predictions: -1 for any score >= the threshold and 1 otherwise.
            """
    
            predictions = [1 if p[0] >= threshold else 0 for p in scores]
    
            return predictions

class IsolationTree:
    def __init__(self, height_limit, current_height):

        self.depth = height_limit
        self.current_height = current_height
        self.split_by = None
        self.split_value = None
        self.right = None
        self.left = None
        self.size = 0
        self.exnodes = 0
        self.n_nodes = 1

    def fit(self, X:np.ndarray):
        """
        Given a 2D matrix of observations, create an isolation tree. Set field
        self.root to the root of that tree and return it.
        If you are working on an improved algorithm, check parameter "improved"
        and switch to your new functionality else fall back on your original code.
        """

        if len(X) <= 1 or self.current_height >= self.depth:
            self.exnodes = 1
            self.size = X.shape[0]

            return self

        split_by = random.choice(np.arange(X.shape[1]))
        X_col = X[:, split_by]
        min_x = X_col.min()
        max_x = X_col.max()

        #TODO MAJ: MAurras 03112020 = Revoir ce bout de code : ça pourrait créer des problèmes
        if min_x == max_x:
            self.exnodes = 1
            self.size = len(X)

            return self

        else:

            split_value = min_x + random.betavariate(0.5, 0.5) * (max_x - min_x)

            w = np.where(X_col < split_value, True, False)
            del X_col

            self.size = X.shape[0]
            self.split_by = split_by
            self.split_value = split_value

            self.left = IsolationTree(self.depth, self.current_height + 1).fit(X[w])
            self.right = IsolationTree(self.depth, self.current_height + 1).fit(X[~w])
            self.n_nodes = self.left.n_nodes + self.right.n_nodes + 1

        return self

def c(n):
    if n > 2:
        return 2.0*(np.log(n-1)+0.5772156649) - (2.0*(n-1.)/(n*1.0))
    elif n == 2:
        return 1
    if n == 1:
        return 0

def path_length_tree(x, t,e):
    e = e
    if t.exnodes == 1:
        e = e+ c(t.size)
        return e
    else:
        a = t.split_by
        if x[a] < t.split_value :
            return path_length_tree(x, t.left, e+1)
        if x[a] >= t.split_value :
            return path_length_tree(x, t.right, e+1)