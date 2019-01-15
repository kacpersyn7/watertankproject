import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.spatial import distance
# from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import minmax_scale
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt

class MahClassifer:
    def __init__(self, lag_time, window_size):
        self.lag_time = lag_time
        self.window_size = window_size
        self.dimensions = 3

    def fit(self, lag_data):
        self.cov_matrix = lag_data.cov()
        self.mean = np.mean(lag_data)
        self.std = np.std(lag_data)

    def fit_and_predict(self, data):
        lag_data = data[:self.lag_time]/25
        self.fit(lag_data)
        results = np.zeros(len(data.values))
        new_data = data.values/25
        new_mean = np.mean(lag_data)
        for i in range(self.lag_time, len(data)):

            # self.mean = np.apply_over_axes(np.mean, new_data[i - self.window_size+1:i], 0).reshape(self.dimensions)
            old_mean = new_mean
            new_mean = np.apply_over_axes(np.mean, new_data[i - self.window_size+1:i], 0).reshape(self.dimensions)
            dist = distance.mahalanobis(new_mean, old_mean, self.cov_matrix)
            # print(dist)
            if dist > 10e-6:
                results[i] = 1

        return results