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
        if self.pca_mode == 'partial':
            self.pca.partial_fit(lag_data)
        else:
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
                result = W_arr[:, 0]
            elif self.dimensions == 2:
                result = np.logical_and(W_arr[:, 0], W_arr[:, 1])
            else:
                result = np.logical_or(np.logical_and(W_arr[:, 0], W_arr[:, 1]),
                                       np.logical_and(W_arr[:, 0], W_arr[:, 2]),
                                       np.logical_and(W_arr[:, 2], W_arr[:, 1]))
        elif self.voting_mode == 'or':
            result = W_arr[:, 0]
            for i in range(self.dimensions):
                result = np.logical_or(result, W_arr[:, i])
        return result

    def update_e_w(self, y):
        self.W = (np.abs(y - self.mean) > (self.k+4)*self.std).reshape(self.dimensions)
        self.E = (np.abs(y - self.mean) > 4 * self.std).reshape(self.dimensions)

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

    def project_new_observation(self, new_sample):

        if self.pca_mode == 'partial':
            new_sample_norm = new_sample.reshape(-1, 1).T / self.normalize_ratio
            self.pca.partial_fit(new_sample_norm)
            sample_reduced = (self.pca.transform(new_sample_norm)).reshape(self.dimensions)
        elif self.pca_mode == 'normal':
            new_sample_norm = new_sample.reshape(-1, 1).T / self.normalize_ratio
            sample_reduced = (self.pca.transform(new_sample_norm)).reshape(self.dimensions)
        elif self.pca_mode == 'none':
            sample_reduced = new_sample.reshape(self.dimensions) / self.normalize_ratio

        return sample_reduced

    # TODO:
    def fit_and_predict(self, data):
        lag_data = data[:self.lag_time]
        decomposed_data = self.fit_transform(lag_data)
        self.mean = np.apply_over_axes(np.mean, decomposed_data, 0).reshape(self.dimensions)
        self.std = np.apply_over_axes(np.std, decomposed_data, 0).reshape(self.dimensions)

        test_size = len(data) - self.lag_time
        E_results = np.zeros(shape=(len(data), self.dimensions))
        W_results = np.zeros(shape=(len(data), self.dimensions))
        std_results = np.tile(self.std, (len(data), 1))
        mean_results = np.tile(self.mean, (len(data), 1))

        b = np.copy(decomposed_data)
        new_data = np.copy(decomposed_data)
        all_data = np.concatenate((new_data, np.zeros((test_size, self.dimensions))))
        projection_results = np.concatenate((new_data, np.zeros((test_size, self.dimensions))))

        for i in range(self.lag_time, len(data)):

            sample_reduced = self.project_new_observation(data[i])
            self.update_e_w(sample_reduced)
            self.calculate_new_y(sample_reduced, b)

            all_data[i] = self.y.reshape(self.dimensions, 1).T
            projection_results[i] = sample_reduced.reshape(self.dimensions, 1).T

            self.mean = np.apply_over_axes(np.mean, all_data[i - self.window_size+1:i], 0).reshape(self.dimensions)
            self.std = np.apply_over_axes(np.std, all_data[i - self.window_size+1:i], 0).reshape(self.dimensions)

            std_results[i] = self.std
            mean_results[i] = self.mean
            E_results[i] = self.E
            W_results[i] = self.W

        # std_results = np.concatenate((np.tile(std_results[0,:], (self.lag_time,1)), std_results))
        # mean_results = np.concatenate((np.tile(mean_results[0,:], (self.lag_time,1)), mean_results))
        results = self.calculate_result(W_results)

        return results, E_results, W_results, all_data, std_results, mean_results, projection_results
