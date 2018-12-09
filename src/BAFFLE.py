import numpy as np
import pandas as pd

class BAFFLE:
    def __init__(self, lag_time, window_size, alpha, k, dimensions=3):
        self.lag_time = lag_time
        self.window_size = window_size
        self.alpha = alpha
        self.dimensions = dimensions
        self.E = np.zeros(shape=dimensions)
        self.W = np.zeros(shape=dimensions)
        self.mean = np.zeros(shape=dimensions)
        self.std = np.zeros(shape=dimensions)
        self.y = np.zeros(shape=dimensions)
        self.k = k
        self.lag_data = np.zeros(shape=dimensions)

    def normalize_df(self, df):
        normalized_df = (df - df.min()) / (df.max() - df.min())
        return normalized_df

    def update_e_w(self, y):
        self.W = (np.abs(y - self.mean) > (self.k+3)*self.std)
        self.E = (np.abs(y - self.mean) > 3 * self.std)

    def predict(self, y):
        for i in range(self.dimensions):
            if self.E[i] == 0 and self.W[i] == 0:
                self.y[i] = self.alpha * y[i] + (1-self.alpha)*self.y[i]
            elif self.E[i] == 0 and self.W[i] == 1:
                self.y[i] = self.alpha * y[i] + (1 - self.alpha) * b[i]
            if self.E[i] == 1 and self.W[i] == 1:
                self.y[i] = np.random(self.lag_data[i])
    #     b is a random sample
    #     yes xD

    def fit_and_predict(self, data):
        self.lag_data = data[:self.lag_time]
        self.mean = self.lag_data.mean()
        self.std = self.lag_data.std()


    def get_gaussian(self):
        pass

    def learn(self):
        pass

    def classify(self):