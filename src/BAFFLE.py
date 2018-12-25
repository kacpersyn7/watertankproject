import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale
import matplotlib.pyplot as plt

class BAFFLE:
    def __init__(self, lag_time, window_size, alpha=0.5, k=0.1, dimensions=2):
        self.lag_time = lag_time
        self.window_size = window_size
        self.alpha = alpha
        self.dimensions = dimensions
        self.pca = PCA(n_components=dimensions)
        self.E = np.zeros(shape=dimensions)
        self.W = np.zeros(shape=dimensions)
        self.mean = np.zeros(shape=dimensions)
        self.std = np.zeros(shape=dimensions)
        self.y = np.zeros(shape=dimensions)
        self.k = k

    def update_e_w(self, y):
        self.W = np.squeeze(np.abs(y - self.mean) > (self.k+3)*self.std)
        self.E = np.squeeze(np.abs(y - self.mean) > 3 * self.std)

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

    #     b is a random sample
    #     yes xD

    def fit_and_predict(self, data):

        lag_data = minmax_scale(data[:self.lag_time])
        decomposed_data = self.pca.fit_transform(lag_data)

        # plt.plot(decomposed_data)
        # plt.show()
        # a = self.pca.components_
        self.mean = np.apply_over_axes(np.mean, decomposed_data, 0)
        self.std = np.apply_over_axes(np.std, decomposed_data, 0)
        test_length = (len(data) - self.lag_time)
        # E_results_table = np.zeros(shape=(test_length, self.dimensions))
        # W_results_table = np.zeros(shape=(test_length, self.dimensions))
        E_results = []
        W_results = []
        y_results = []
        b = lag_data
        new_data = np.copy(data)
        new_data = np.copy(decomposed_data)
        for i in range(self.lag_time, test_length-1):
            new_sample = minmax_scale(data[i])
            new_sample = new_sample.reshape(3, 1)
            sample_reduced = (new_sample.T @ self.pca.components_.T)
            new_data = np.concatenate((new_data, sample_reduced))
            self.update_e_w(new_data[i])
            self.calculate_new_y(new_data[i], b)

            y_results.append(np.copy(self.y))
            self.mean = np.apply_over_axes(np.mean, new_data[i - self.window_size+1:i], 0)

            self.std = np.apply_over_axes(np.std, new_data[i - self.window_size+1:i], 0)
            # E_results_table[i-self.lag_time]=self.E
            # W_results_table[i - self.lag_time] = self.W

            E_results.append(self.E)
            W_results.append(self.W)

        return E_results, W_results, y_results