import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import minmax_scale
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt


class BAFFLE:
    def __init__(self, lag_time, window_size, alpha=0.5, k=0.1, pca_mode='normal', voting_mode='majority',
                 explained_variance_ratio=0.7):
        self.normalize_ratio = 25
        self.lag_time = lag_time
        self.window_size = window_size
        self.alpha = alpha
        self.k = k
        self.pca_mode = pca_mode
        if pca_mode == 'normal':
            self.pca = PCA(n_components=explained_variance_ratio)
        elif pca_mode == 'partial':
            self.pca = IncrementalPCA(n_components=explained_variance_ratio, batch_size=window_size)
            self.dimensions = explained_variance_ratio
            self.init_variables()
        elif pca_mode == 'none':
            self.dimensions = 3
            self.init_variables()
        self.voting_mode = voting_mode

    def init_variables(self):
        self.E = np.zeros(shape=self.dimensions)
        self.W = np.zeros(shape=self.dimensions)
        self.result = np.zeros(shape=self.dimensions)
        self.mean = np.zeros(shape=self.dimensions)
        self.std = np.zeros(shape=self.dimensions)
        self.y = np.zeros(shape=self.dimensions)

    def fit(self, lag_data):
        self.pca.fit(lag_data)
        self.dimensions = self.pca.n_components_
        self.init_variables()

    def fit_transform(self, lag_data):
        normalized_lag_data = lag_data/self.normalize_ratio
        self.fit(normalized_lag_data)
        return self.pca.transform(normalized_lag_data)

    def partial_fit(self, lag_data):
        self.pca.partial_fit(lag_data/self.normalize_ratio)

    def calculate_result(self, W_arr):
        if self.voting_mode == 'majority':
            if self.dimensions == 1:
                self.result = W_arr[:, 0]
            elif self.dimensions == 2:
                self.result = np.logical_and(W_arr[:, 0], W_arr[:, 1])
            else:
                self.result = np.logical_or(np.logical_and(W_arr[:, 0], W_arr[:, 1]),
                                       np.logical_and(W_arr[:, 0], W_arr[:, 2]),
                                       np.logical_and(W_arr[:, 2], W_arr[:, 1]))
        elif self.voting_mode == 'or':
            result = W_arr[:, 0]
            for i in range(self.dimensions):
                result = np.logical_or(result, W_arr[:, i])

    def update_e_w(self, y):
        self.W = (np.abs(y - self.mean) > (self.k+3)*self.std).reshape(self.dimensions)
        self.E = (np.abs(y - self.mean) > 3 * self.std).reshape(self.dimensions)

    def calculate_new_y(self, y, b):
        for i in range(self.dimensions):
            if self.E[i] == 0 and self.W[i] == 0:
                self.y[i] = self.alpha * y[i] + (1-self.alpha)*self.y[i]
            elif self.E[i] == 0 and self.W[i] == 1:
                index = np.random.randint(self.lag_time, size=1)[0]
                self.y[i] = self.alpha * y[i] + (1 - self.alpha) * b[index][i]
            elif self.E[i] == 1 and self.W[i] == 1:
                index = np.random.randint(self.lag_time, size=1)[0]
                self.y[i] = b[index][i]

    # TODO:
    def fit_and_predict(self, data):
        lag_data = data[:self.lag_time]/25
        decomposed_data = self.fit(lag_data)
        self.mean = np.apply_over_axes(np.mean, decomposed_data, 0).reshape(self.dimensions)
        self.std = np.apply_over_axes(np.std, decomposed_data, 0).reshape(self.dimensions)
        # E_results_table = np.zeros(shape=(test_length, self.dimensions))
        # W_results_table = np.zeros(shape=(test_length, self.dimensions))
        E_results = []
        W_results = []
        std_results = [self.std]
        mean_results = [self.mean]
        projection_results = np.copy(decomposed_data)
        b = decomposed_data
        new_data = np.copy(decomposed_data)
        for i in range(self.lag_time, len(data)):
            # new_sample = data[i].reshape(self.dimensions)/25
            new_sample = data[i].reshape(-1,1).T/25
            self.pca.partial_fit(new_sample)
            sample_reduced = (self.pca.transform(new_sample)).reshape(self.dimensions)
            # sample_reduced = new_sample

            self.update_e_w(sample_reduced)
            self.calculate_new_y(sample_reduced, b)

            new_data = np.concatenate((new_data, self.y.reshape(self.dimensions, 1).T))
            projection_results = np.concatenate((projection_results, sample_reduced.reshape(self.dimensions, 1).T))
            self.mean = np.apply_over_axes(np.mean, new_data[i - self.window_size+1:i], 0).reshape(self.dimensions)

            self.std = np.apply_over_axes(np.std, new_data[i - self.window_size+1:i], 0).reshape(self.dimensions)
            std_results.append(self.std)
            mean_results.append(self.mean)
            E_results.append(self.E)
            W_results.append(self.W)

        return E_results, W_results, new_data, std_results, mean_results, projection_results