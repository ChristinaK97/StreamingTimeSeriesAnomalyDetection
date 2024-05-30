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

from OnlineEncDec_AD.OnlineEncDecAD import OnlineEncDecAD
from tstENCDEC import separate_sets, ECG, load_dataset, base_path

from scipy.stats import ks_2samp


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
Version 2.0: Predict only ref_drift with outdated model
------------
Worst results than Base Version 1

stream_batch_size: int = 10000,
Wtrain: int = 1000, Wdrift: int = 200,
incremental_cutoff: int = 50,
percentile_cutoff: int = 90,
ks_significance_level: float = 0.01

Accuarcy: 0.8806125
AUC: 0.6378904657178954
AUC PR: 0.08760713692453725
[[69087  7238]
 [ 2313  1362]]
              precision    recall  f1-score   support

      normal     0.9676    0.9052    0.9353     76325
     anamoly     0.1584    0.3706    0.2219      3675

    accuracy                         0.8806     80000
   macro avg     0.5630    0.6379    0.5786     80000
weighted avg     0.9304    0.8806    0.9026     80000


ks_significance_level: float = 0.001

Accuarcy: 0.8751625
AUC: 0.6624889982196613
AUC PR: 0.09752498723590916
[[68439  7886]
 [ 2101  1574]]
              precision    recall  f1-score   support

      normal     0.9702    0.8967    0.9320     76325
     anamoly     0.1664    0.4283    0.2397      3675

    accuracy                         0.8752     80000
   macro avg     0.5683    0.6625    0.5858     80000
weighted avg     0.9333    0.8752    0.9002     80000

"""

device = current_device() if is_available() else None
print(f"Device = {device}")

class CapacityQueue:
    def __init__(self, queue_capacity, with_replacement=False):
        self.replaced_elements = None
        self.total_elements = None
        self.queue = None
        self.index_queue = None

        self.queue_capacity = queue_capacity  # Maximum capacity of the queue
        self.with_replacement = with_replacement
        self.reset()

    def __len__(self):
        return len(self.queue)

    def reset(self):
        self.queue = deque(maxlen=self.queue_capacity if self.with_replacement else None)  # Initialize deque with a fixed size
        self.index_queue = deque(maxlen=self.queue_capacity if self.with_replacement else None)
        self.total_elements = 0  # Total elements added
        self.replaced_elements = 0  # Number of elements replaced

    def push(self, x, t):
        if self.with_replacement and len(self.queue) == self.queue_capacity:
            # If the queue is full, increment replaced_elements
            self.replaced_elements += 1
        self.queue.append(x)
        self.index_queue.append(t)
        self.total_elements += 1

    def isFull(self):
        return len(self.queue) >= self.queue_capacity

    def percentage_replaced(self):
        if self.total_elements < self.queue_capacity:
            return 0  # No replacement has occurred if the total elements added is less than capacity
        return (self.replaced_elements / self.total_elements) * 100

    def copy_from(self, other):
        self.queue = deque(other.queue, maxlen=other.queue_capacity)
        self.index_queue = deque(other.index_queue, maxlen=other.queue_capacity)
        self.total_elements = other.total_elements
        self.replaced_elements = other.replaced_elements

    def extend(self, other):
        if isinstance(other, CapacityQueue):
            self.queue.extend(other.queue)
            self.index_queue.extend(other.index_queue)
            self.total_elements += other.total_elements
            self.replaced_elements += self.replaced_elements

        elif isinstance(other, pd.DataFrame):
            for (t, xt) in other.iterrows():
                self.push(xt, t)
            self.total_elements += other.shape[0]


    def show_indexes(self):
        return show_indexes(self.index_queue)




def show_indexes(l):
    if len(l) > 0:
        return f"[{l[0]} - {l[-1]}]"
    else:
        return "Empty"

def check_consistency(l):
    is_consistent = True
    for i in range(1, len(l)):
        if l[i] != l[i - 1] + 1:
            is_consistent = False
            print(f"i = {i} [i-1] = {l[i - 1]} [i] = {l[i]}")
    return is_consistent


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

    TRAINING_EPOCHS = 40
    INCREMENTAL_TRAINING_EPOCHS = 10
    MAX_BUFFER_LEN = 2000

    def __init__(self,
                 df: pd.DataFrame,
                 Wincrem: int = 1000, Wdrift: int = 200,
                 incremental_cutoff: int = 50,
                 percentile_cutoff: int = 90,
                 ks_significance_level: float = 0.001
                 ):
        self.name = "OnlineEncDecAD"
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
        self.initialize(df)

        self.errors = []
        self.Y_hat = []
        self.predicted_idxs = []
        self.pred_buffer= CapacityQueue(OnlineAD.MAX_BUFFER_LEN)
        self.mov_increm = CapacityQueue(self.Wincrem, True)
        self.mov_drift  = CapacityQueue(self.Wdrift)
        self.ref_drift  = CapacityQueue(self.Wdrift)

        self.parse_stream()
        self.get_predictions()
        self.results()

        print()

    # -----------------------------------------------------------------------------------------------

    def initialize(self, df: pd.DataFrame):
        X = df[[0]]
        Y = df[1]
        self.n_features = X.shape[1]

        self.sequence_length = find_length(X[0])
        print("Seq len = ", self.sequence_length)
        X_init_train, Y_init_train, _, _, self.X, self.Y = separate_sets(X, Y, train_perc=0.2, val_perc=0)

        print(f"Initializing model with {X_init_train.shape[0]} samples...")
        self.fit_model(X_init_train, True)


    def fit_model(self, X: Union[pd.DataFrame, CapacityQueue], reset_model: bool):

        if reset_model:
            self.model = OnlineEncDecAD(
                n_features = self.n_features,
                hidden_size = 32,
                sequence_length = self.sequence_length,
                gpu = device)
            epochs = OnlineAD.TRAINING_EPOCHS
        else:
            epochs = OnlineAD.INCREMENTAL_TRAINING_EPOCHS

        X = self.toDF(X)
        print(f"\nTraining model ({'from scratch' if reset_model else 'incremental'}) with {X.shape} samples")
        self.model.fit(X, epochs, batch_size = self.get_batch_size(X))
        _, train_loss = self.model.predict(X)
        self.theta_t = self.calc_percentile(train_loss)
        print(f"New theta_t = {self.theta_t}\n{'-'*30}")

    def calc_percentile(self, loss):
        return np.percentile(loss, self.percentile_cutoff, interpolation='linear')



    def predict_model(self, X: Union[pd.DataFrame, CapacityQueue], is_output: bool):
        if is_output:
            idxs = X.index_queue

        X = self.toDF(X)

        _, loss = self.model.predict(X, batch_size = self.get_batch_size(X))
        Y_hat = (loss > self.theta_t).astype(int)

        if is_output:
            self.errors.append(loss)
            self.Y_hat.append(Y_hat)
            self.predicted_idxs.extend(idxs)

        return loss


    # -------------------------------------------------------------------------------------------------
    def parse_stream(self):

        for (t, xt), (_, yt) in zip(self.X.iterrows(), self.Y.items()):
            if self.X.shape[0] - t > self.sequence_length:  # at least one sequence should remain in the stream
                self.incremental_training(t, xt)
                self.concept_drift_detection(t, xt)
            else:
                print("\nProcessing last sequence at t = ", t)
                self.pred_buffer.extend(self.ref_drift)
                self.pred_buffer.extend(self.mov_drift)
                self.pred_buffer.extend(self.X.iloc[t:])
                break

        if len(self.pred_buffer) > 0:
            print(f"Predict last buffer {self.pred_buffer.show_indexes()} {str(check_consistency(self.pred_buffer.index_queue))}  (# {len(self.pred_buffer)}) before finish")
            self.predict_model(self.pred_buffer, True)
            self.pred_buffer.reset()



    def incremental_training(self, t, xt):
        self.mov_increm.push(xt, t)
        if self.mov_increm.isFull() or self.mov_increm.percentage_replaced() >= self.incremental_cutoff:

            print(f"Incremental training {self.mov_increm.show_indexes()} (# {len(self.mov_increm)}) at t = {t}")
            self.printObjects()

            self.fit_model(self.mov_increm, False)
            self.mov_increm.reset()



    def concept_drift_detection(self, t, xt):
        if not self.ref_drift.isFull():
            self.ref_drift.push(xt, t)
        elif not self.mov_drift.isFull():
            self.mov_drift.push(xt, t)
        else:
            ref_loss = self.predict_model(self.ref_drift, False)
            mov_loss = self.predict_model(self.mov_drift, False)
            drift_detected = self.check_for_drift(ref_loss, mov_loss, t)

            if drift_detected:

                print(f"Drift detected at t = {t}")

                self.pred_buffer.extend(self.ref_drift)
                print(f"Predict buffer {self.pred_buffer.show_indexes()} {str(check_consistency(self.pred_buffer.index_queue))}  (# {len(self.pred_buffer)})  before drift reset step")
                self.predict_model(self.pred_buffer, True)

                self.pred_buffer.reset()
                self.pred_buffer.extend(self.mov_drift)
                self.pred_buffer.push(xt, t)


                self.fit_model(self.mov_drift, True)
                self.ref_drift.reset()
                self.mov_drift.reset()
                self.mov_increm.reset()
            else:
                prev = self.ref_drift.show_indexes() + " " + str(check_consistency(self.ref_drift.index_queue))

                self.pred_buffer.extend(self.ref_drift)
                if self.pred_buffer.isFull():
                    print(f"Predict buffer {self.pred_buffer.show_indexes()}  (# {len(self.pred_buffer)}) since full")
                    self.predict_model(self.pred_buffer, True)
                    self.pred_buffer.reset()

                self.ref_drift.copy_from(self.mov_drift)
                self.mov_drift.reset()
                self.mov_drift.push(xt, t)

                print(f"No Drift at t = {t}")
                self.printObjects("Before: " + prev)



    def check_for_drift(self, ref_loss: np.ndarray, mov_loss: np.ndarray, t:int):
        """:return: 
            True:  The two distributions are significantly different (reject H0).
            False: The two distributions are not significantly different (fail to reject H0).
        """""
        ks_statistic, p_value = ks_2samp(ref_loss.ravel(), mov_loss.ravel())
        print(f"Checked for drift p-value = {p_value} at t = {t}")
        return p_value < self.ks_significance_level


    # -------------------------------------------------------------------------------------------------

    def toDF(self, X):
        if isinstance(X, pd.DataFrame):
            return X
        elif isinstance(X, CapacityQueue):
            return pd.DataFrame(X.queue)
        else:
            ValueError(f"Invalid x type: {type(X)}")\


    def get_batch_size(self, X):
        data_len = X.shape[0] if isinstance(X, pd.DataFrame) else len(X)
        return min(
            int(4 * (self.sequence_length // 100) * 100), data_len)

    def get_predictions(self):
        self.errors = np.concatenate(self.errors, axis=1).ravel()
        self.Y_hat  = np.concatenate(self.Y_hat, axis=1).ravel()

        print(len(self.errors), len(self.Y_hat), len(self.predicted_idxs))
        print(show_indexes(self.predicted_idxs), check_consistency(self.predicted_idxs))

        
        
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
        print(f"\tref_drift : {self.ref_drift.show_indexes()} {str(check_consistency(self.ref_drift.index_queue))} \t# {len(self.ref_drift)} \t{message}")
        print(f"\tmov_drift : {self.mov_drift.show_indexes()} {str(check_consistency(self.mov_drift.index_queue))} \t# {len(self.mov_drift)}")
        print(f"\tmov_incre : {self.mov_increm.show_indexes()} {str(check_consistency(self.mov_increm.index_queue))} \t# {len(self.mov_increm)}")
        print(f"\tpred_buff : {self.pred_buffer.show_indexes()} {str(check_consistency(self.pred_buffer.index_queue))}\t# {len(self.pred_buffer)}")






def load_tst_set():
    import os
    current_script_path = os.path.abspath(__file__)
    print("Absolute path of the current script:", current_script_path)
    i = current_script_path.find("OnlineEncDec_AD")
    current_script_path = current_script_path[0:i]

    filenames = [str(Path(current_script_path, base_path, ECG))]
    return load_dataset(filenames, sample_size=100000)


df, _, _ = load_tst_set()
print("Loaded dataset ", df.shape)
OnlineAD(df)




