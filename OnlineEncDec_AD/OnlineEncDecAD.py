import pandas as pd
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import multivariate_normal
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import trange

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'EncDec-AD')))
from algorithm_utils import Algorithm, PyTorchUtils
from lstm_enc_dec_axl import LSTMEDModule



class OnlineEncDecAD(Algorithm, PyTorchUtils):

    def __init__(self, name: str = 'LSTM-ED', lr: float = 1e-3,
                 n_features: int = 1, hidden_size: int = 5, sequence_length: int = 30, train_gaussian_percentage: float = 0.25,
                 n_layers: tuple = (1, 1), use_bias: tuple = (True, True), dropout: tuple = (0, 0),
                 seed: int = 42, gpu: int = None, details=True):

        Algorithm.__init__(self, __name__, name, seed, details=details)
        PyTorchUtils.__init__(self, seed, gpu)
        self.lr = lr

        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.train_gaussian_percentage = train_gaussian_percentage

        self.n_layers = n_layers
        self.use_bias = use_bias
        self.dropout = dropout

        self.mean, self.cov = None, None

        self.lstmed = LSTMEDModule(n_features, self.hidden_size,
                                   self.n_layers, self.use_bias, self.dropout,
                                   seed=self.seed, gpu=self.gpu)
        self.to_device(self.lstmed)



    def fit(self, X: pd.DataFrame, num_epochs: int = 10, batch_size: int = 20):

        X.interpolate(inplace=True)
        X.bfill(inplace=True)
        data = X.values
        sequences = [data[i:i + self.sequence_length] for i in range(data.shape[0] - self.sequence_length + 1)]
        indices = np.random.permutation(len(sequences))
        split_point = int(self.train_gaussian_percentage * len(sequences))

        train_loader = DataLoader(dataset=sequences, batch_size=batch_size, drop_last=False,
                                  sampler=SubsetRandomSampler(indices[:-split_point]), pin_memory=True)
        train_gaussian_loader = DataLoader(dataset=sequences, batch_size=batch_size, drop_last=False,
                                           sampler=SubsetRandomSampler(indices[-split_point:]), pin_memory=True)

        optimizer = torch.optim.Adam(self.lstmed.parameters(), lr=self.lr)

        self.lstmed.train()
        for epoch in trange(num_epochs):
            logging.debug(f'Epoch {epoch+1}/{num_epochs}.')
            for ts_batch in train_loader:
                output = self.lstmed(self.to_var(ts_batch))
                loss = nn.MSELoss(size_average=False)(output, self.to_var(ts_batch.float()))
                self.lstmed.zero_grad()
                loss.backward()
                optimizer.step()

        self.lstmed.eval()
        error_vectors = []
        for ts_batch in train_gaussian_loader:
            output = self.lstmed(self.to_var(ts_batch))
            error = nn.L1Loss(reduce=False)(output, self.to_var(ts_batch.float()))
            error_vectors += list(error.view(-1, X.shape[1]).data.cpu().numpy())

        self.mean = np.mean(error_vectors, axis=0)
        self.cov = np.cov(error_vectors, rowvar=False)




    def predict(self, X: pd.DataFrame, batch_size: int = 20):
        # dont calculate grad during evaluation
        with torch.no_grad():
            X.interpolate(inplace=True)
            X.bfill(inplace=True)
            data = X.values

            sequences = [data[i : i + self.sequence_length] for i in range(data.shape[0] - self.sequence_length + 1)]
            data_loader = DataLoader(dataset=sequences, batch_size=batch_size, shuffle=False, drop_last=False)

            self.lstmed.eval()

            mvnormal = multivariate_normal(self.mean, self.cov, allow_singular=True)

            scores = []
            outputs = []
            errors = []
            for idx, ts in enumerate(data_loader):
                output = self.lstmed(self.to_var(ts))
                error = nn.L1Loss(reduce=False)(output, self.to_var(ts.float()))
                score = -mvnormal.logpdf(error.view(-1, X.shape[1]).data.cpu().numpy())
                scores.append(score.reshape(ts.size(0), self.sequence_length))
                if self.details:
                    # send to cpu
                    outputs.append(output.data.cpu().numpy())
                    errors.append(error.data.cpu().numpy())

            # stores seq_len-many scores per timestamp and averages them
            scores = np.concatenate(scores)
            lattice = np.full((self.sequence_length, data.shape[0]), np.nan)
            for i, score in enumerate(scores):
                lattice[i % self.sequence_length, i:i + self.sequence_length] = score
            scores = np.nanmean(lattice, axis=0)

            if self.details:
                outputs = np.concatenate(outputs)
                lattice = np.full((self.sequence_length, X.shape[0], X.shape[1]), np.nan)
                for i, output in enumerate(outputs):
                    lattice[i % self.sequence_length, i:i + self.sequence_length, :] = output
                self.prediction_details.update({'reconstructions_mean': np.nanmean(lattice, axis=0).T})

                errors = np.concatenate(errors)
                lattice = np.full((self.sequence_length, X.shape[0], X.shape[1]), np.nan)
                for i, error in enumerate(errors):
                    lattice[i % self.sequence_length, i:i + self.sequence_length, :] = error
                self.prediction_details.update({'errors_mean': np.nanmean(lattice, axis=0).T})

        return scores, self.prediction_details['errors_mean']
