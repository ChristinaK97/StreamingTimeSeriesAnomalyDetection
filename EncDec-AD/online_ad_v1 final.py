from collections import deque
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import confusion_matrix

from TSB_UAD.utils.slidingWindows import find_length
from torch.cuda import current_device, is_available
from scipy.stats import ks_2samp
from tqdm import tqdm

from enc_dec_ad import EncDecAD
from tstENCDEC import separate_sets, ECG, load_dataset, base_path

import random
import torch
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

"""
Base version
------------

stream_batch_size: int = 10000,
Wtrain: int = 1000, Wdrift: int = 200,
incremental_cutoff: int = 50,
percentile_cutoff: int = 90,
ks_significance_level: float = 0.01

Accuarcy: 0.88405
AUC: 0.7175235688772724
AUC PR: 0.13145938724812903
[[68761  7564]
 [ 1712  1963]]
              precision    recall  f1-score   support

      normal     0.9757    0.9009    0.9368     76325
     anamoly     0.2060    0.5341    0.2974      3675

    accuracy                         0.8841     80000
   macro avg     0.5909    0.7175    0.6171     80000
weighted avg     0.9404    0.8841    0.9074     80000
"""

"""
Added MAX BUFFER SIZE 2000
stream_batch_size: int = 10000,
Wtrain: int = 1000, Wdrift: int = 200,
incremental_cutoff: int = 50,
percentile_cutoff: int = 90,
ks_significance_level: float = 0.01

Accuarcy: 0.8866875
AUC: 0.7181287949179016
AUC PR: 0.13348443490035738
[[68978  7347]
 [ 1718  1957]]
              precision    recall  f1-score   support

      normal     0.9757    0.9037    0.9383     76325
     anamoly     0.2103    0.5325    0.3016      3675

    accuracy                         0.8867     80000
   macro avg     0.5930    0.7181    0.6200     80000
weighted avg     0.9405    0.8867    0.9091     80000

---------------------------------
Wtrain: int = 500, Wdrift: int = 500,

Accuarcy: 0.893975
AUC: 0.7032994868435419
AUC PR: 0.1293361091347484
[[69705  6620]
 [ 1862  1813]]
              precision    recall  f1-score   support

      normal     0.9740    0.9133    0.9426     76325
     anamoly     0.2150    0.4933    0.2995      3675

    accuracy                         0.8940     80000
   macro avg     0.5945    0.7033    0.6211     80000
weighted avg     0.9391    0.8940    0.9131     80000

"""


device = current_device() if is_available() else None
# device = None
print(f"Device = {device}")


class CapacityQueue:
    def __init__(self, queue_capacity, with_replacement=False):
        self.replaced_elements = None
        self.total_elements = None
        self.queue = None

        self.queue_capacity = queue_capacity  # Maximum capacity of the queue
        self.with_replacement = with_replacement
        self.reset()

    def __len__(self):
        return len(self.queue)

    def reset(self):
        self.queue = deque(maxlen=self.queue_capacity if self.with_replacement else None)  # Initialize deque with a fixed size
        self.total_elements = 0  # Total elements added
        self.replaced_elements = 0  # Number of elements replaced

    def push(self, x):
        if self.with_replacement and len(self.queue) == self.queue_capacity:
            # If the queue is full, increment replaced_elements
            self.replaced_elements += 1
        self.queue.append(x)
        self.total_elements += 1

    def isFull(self):
        return len(self.queue) >= self.queue_capacity

    def percentage_replaced(self):
        if self.total_elements < self.queue_capacity:
            return 0  # No replacement has occurred if the total elements added is less than capacity
        return (self.replaced_elements / self.total_elements) * 100

    def copy_from(self, other):
        self.queue = deque(other.queue, maxlen=other.queue_capacity)
        self.total_elements = other.total_elements
        self.replaced_elements = other.replaced_elements





def create_stream(X, Y, stream_batch_size: int = 10000) -> List[Tuple[pd.DataFrame, pd.Series]]:
    # first stream batch boundaries
    start = 0
    end = min(stream_batch_size, X.shape[0])

    stream = []
    while start < X.shape[0] and end <= X.shape[0]:
        X_stream_batch = X.iloc[start:end].reset_index(drop=True)
        Y_stream_batch = Y.iloc[start:end].reset_index(drop=True)
        stream.append((X_stream_batch, Y_stream_batch))

        # move on to the next batch of the stream
        start += stream_batch_size
        end += stream_batch_size
        end = min(end, X.shape[0])
    return stream



class OnlineAD:

    def __init__(self,
                 df: pd.DataFrame,
                 Wincrem: int = 1000, Wdrift: int = 200,
                 max_buffer_len: int = 2000,
                 incremental_cutoff: int = 50,
                 percentile_cutoff: int = 90,
                 ks_significance_level: float = 0.001,
                 training_epochs: int = 40,
                 incremental_training_epochs: int = 10,
                 use_increm_learning: bool = True,
                 use_concept_drift: bool = True,
                 verbose: int = 0
                 ):
        self.name = "OnlineEncDecAD"
        self.v = verbose
        self.training_epochs = training_epochs
        self.incremental_training_epochs = incremental_training_epochs
        self.max_buffer_len = max_buffer_len

        self.Wincrem = Wincrem
        self.Wdrift = Wdrift
        self.incremental_cutoff = incremental_cutoff
        self.percentile_cutoff = percentile_cutoff
        self.ks_significance_level = ks_significance_level

        self.X, self.Y = None, None
        self.sequence_length = None
        self.n_features = None
        self.model = None
        self.theta_t = None
        self.drift_idxs = []
        self.increm_idxs = []
        self.initialize(df)

        self.errors = []
        self.Y_hat = []
        self.pred_buffer= CapacityQueue(self.max_buffer_len)
        self.wnd_increm = CapacityQueue(self.Wincrem, True)
        self.wnd_drift  = CapacityQueue(self.Wdrift)
        self.wnd_ref    = CapacityQueue(self.Wdrift)

        self.parse_stream(use_increm_learning, use_concept_drift)
        self.get_predictions()

    # -----------------------------------------------------------------------------------------------

    def initialize(self, df: pd.DataFrame):
        X = df[[0]]
        Y = df[1]
        self.n_features = X.shape[1]
        self.sequence_length = find_length(X[0])
        X_init_train, Y_init_train, _, _, self.X, self.Y = separate_sets(X, Y, train_perc=0.2, val_perc=0)
        if self.v>=1: print(f"Seq len = {self.sequence_length}\nInitializing model with {X_init_train.shape[0]} samples...")
        self.fit_model(X_init_train, True, 0)


    def fit_model(self, X: Union[pd.DataFrame, CapacityQueue], reset_model: bool, t:int):

        if reset_model:
            self.model = EncDecAD(
                n_features = self.n_features,
                hidden_size = 32,
                sequence_length = self.sequence_length,
                gpu = device)
            epochs = self.training_epochs
        else:
            epochs = self.incremental_training_epochs

        X = self.toDF(X)

        if self.v>=1: print(f"\nTraining model ({'from scratch' if reset_model else 'incremental'}) with {X.shape} samples")
        if reset_model: self.drift_idxs.append(t)
        else: self.increm_idxs.append(t)

        self.model.fit(X, epochs, batch_size = self.get_batch_size(X), show_progress_bar=self.v>=2)
        _, train_loss = self.model.predict(X)
        self.theta_t = self.calc_percentile(train_loss)

        if self.v>=1: print(f"New theta_t = {self.theta_t}\n{'-'*30}")


    def calc_percentile(self, loss):
        return np.percentile(loss, self.percentile_cutoff, interpolation='linear')



    def predict_model(self, X: Union[pd.DataFrame, CapacityQueue], is_output: bool):
        X = self.toDF(X)

        _, loss = self.model.predict(X, batch_size = self.get_batch_size(X))
        Y_hat = (loss > self.theta_t).astype(int)

        if is_output:
            self.errors.append(loss)
            self.Y_hat.append(Y_hat)

        return loss


    # -------------------------------------------------------------------------------------------------
    def parse_stream(self, use_increm_learning: bool, use_concept_drift: bool):

        print(f"Parse stream with #{self.X.shape[0]} data points...")
        for (t, xt) in self.X.iterrows():
            if self.v == 0 and t % 5000 == 0: print(f"{t} ->", end='')

            self.pred_buffer.push(xt)

            if self.pred_buffer.isFull():
                if self.v >= 2: print(f"Predict buffer (# {len(self.pred_buffer)}) since full")
                self.predict_model(self.pred_buffer, True)
                self.pred_buffer.reset()

            if self.X.shape[0] - t > self.sequence_length:  # at least one sequence should remain in the stream
                if use_increm_learning:
                    self.incremental_training(t, xt)
                if use_concept_drift:
                    self.concept_drift_detection(t, xt)

        if len(self.pred_buffer) > 0:
            print(f"\nPredict last buffer (# {len(self.pred_buffer)}) before finish")
            self.predict_model(self.pred_buffer, True)
            self.pred_buffer.reset()



    def incremental_training(self, t, xt):
        self.wnd_increm.push(xt)
        if self.wnd_increm.isFull() or self.wnd_increm.percentage_replaced() >= self.incremental_cutoff:

            if self.v>=2: print(f"Incremental training (# {len(self.wnd_increm)}) at t = {t}"); self.printObjects()

            self.fit_model(self.wnd_increm, False, t)
            self.wnd_increm.reset()



    def concept_drift_detection(self, t, xt):
        if not self.wnd_ref.isFull():
            self.wnd_ref.push(xt)
        elif not self.wnd_drift.isFull():
            self.wnd_drift.push(xt)
        else:
            ref_loss = self.predict_model(self.wnd_ref, False)
            drift_loss = self.predict_model(self.wnd_drift, False)
            drift_detected = self.check_for_drift(ref_loss, drift_loss, t)

            if drift_detected:
                if self.v>=2:
                    print(f"Drift detected at t = {t}")
                    self.printObjects()

                if len(self.pred_buffer) > self.sequence_length:
                    if self.v>=2: print(f"Predict buffer (# {len(self.pred_buffer)})  before drift reset step")
                    self.predict_model(self.pred_buffer, True)
                    self.pred_buffer.reset()

                self.fit_model(self.wnd_drift, True, t)
                self.wnd_ref.reset()
                self.wnd_drift.reset()
                self.wnd_increm.reset()
            else:
                self.wnd_ref.copy_from(self.wnd_drift)
                self.wnd_drift.reset()
                if self.v>=2:
                    print(f"No Drift at t = {t}"); self.printObjects()



    def check_for_drift(self, ref_loss: np.ndarray, drift_loss: np.ndarray, t:int):
        """:return: 
            True:  The two distributions are significantly different (reject H0).
            False: The two distributions are not significantly different (fail to reject H0).
        """""
        ks_statistic, p_value = ks_2samp(ref_loss.ravel(), drift_loss.ravel())
        if self.v>=2: print(f"Checked for drift p-value = {p_value} at t = {t}")
        return p_value < self.ks_significance_level


    # -------------------------------------------------------------------------------------------------

    def toDF(self, X):
        if isinstance(X, pd.DataFrame):
            return X
        elif isinstance(X, CapacityQueue):
            return pd.DataFrame(X.queue)
        else:
            ValueError(f"Invalid x type: {type(X)}")


    def get_batch_size(self, X):
        data_len = X.shape[0] if isinstance(X, pd.DataFrame) else len(X)
        p = len(str(self.sequence_length)) - 1
        p = pow(10, p)
        batch_size = min(int(4 * (self.sequence_length // p) * p), data_len)
        return batch_size

    def get_predictions(self):
        self.errors = np.concatenate(self.errors, axis=1).ravel()
        self.Y_hat  = np.concatenate(self.Y_hat, axis=1).ravel()
        
        
    def results(self):
        
        target_names = ['normal', 'anamoly']
        print("Accuarcy: " + str(accuracy_score(self.Y, self.Y_hat)))
        print("AUC: " + str(roc_auc_score(self.Y, self.Y_hat)))
        print("AUC PR: " + str(average_precision_score(self.Y, self.Y_hat)))
        print(confusion_matrix(self.Y, self.Y_hat))
        print(classification_report(self.Y, self.Y_hat, target_names=target_names, digits=4))

        return {'name': self.name,
                'Accuarcy': str(accuracy_score(self.Y, self.Y_hat)),
                'AUC': str(roc_auc_score(self.Y, self.Y_hat)),
                'AUC PR': str(average_precision_score(self.Y, self.Y_hat)),
                'confusion matrix': confusion_matrix(self.Y, self.Y_hat),
                'report': (classification_report(self.Y, self.Y_hat, target_names=target_names, digits=4, output_dict=True))}

    # -------------------------------------------------------------------------------------------------

    def printObjects(self, message= None):
        print(f"\tref_drift : # {len(self.wnd_ref)}\t {message}")
        print(f"\tmov_drift : # {len(self.wnd_drift)}")
        print(f"\tmov_incre : # {len(self.wnd_increm)}")
        print(f"\tpred_buff : # {len(self.pred_buffer)}")





def load_tst_set():
    import os
    current_script_path = os.path.abspath(__file__)
    print("Absolute path of the current script:", current_script_path)
    i = current_script_path.find("EncDec-AD")
    current_script_path = current_script_path[0:i]

    SMD = 'SMD/machine-1-1.test.csv@1.out'
    filenames = [str(Path(current_script_path, base_path, SMD))]
    return load_dataset(filenames, sample_size=100000)


df, _, _ = load_tst_set()
print("Loaded dataset ", df.shape)
model = OnlineAD(df, verbose=2)
print(model.results())
print(len(model.drift_idxs), model.drift_idxs)


