import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
import os


class LoadData:
    def __init__(self, path='../experiments'):
        self.path = path
        self.experiments_paths = []
        self.get_experiments_paths()

    def get_experiments_paths(self):
        for filename in os.listdir(self.path):
            if '.' not in filename:
                for mat_file in os.listdir(self.path+filename):
                    if mat_file == 'experiment_results.mat':
                        self.experiments_paths.append(self.path+filename+'/'+mat_file)

    def load_mat_file(self, file_path):
        split = file_path.split('Exp_')
        number = split[1]
        number = number.split('_')
        number = number[0]

        data = loadmat(file_path)

        t = data['t'].reshape(-1)
        h1 = data['h1'].reshape(-1)
        h2 = data['h2'].reshape(-1)
        h3 = data['h3'].reshape(-1)

        u = data['u'].reshape(-1)
        v1 = data['v1'].reshape(-1)
        v1 = 2*np.maximum(np.zeros(v1.shape), v1-0.5)

        v2 = data['v2'].reshape(-1)
        v2 = 2*np.maximum(np.zeros(v2.shape), v2-0.5)

        v3 = data['v3'].reshape(-1)
        v3 = 2*np.maximum(np.zeros(v3.shape), v3-0.5)

        error = np.logical_or(v1 > 0, v3 > 0, v2 > 0)

        data_merged = np.array([h1, h2, h3, v1, v2, v3, u, error])

        return pd.DataFrame(data_merged.T, index=t, columns=['h1', 'h2', 'h3', 'v1', 'v2', 'v3', 'u', 'error']), number

    def build_data_frame(self, experiments_to_exclude=[]):
        # frames = [self.load_mat_file(file_path) for file_path in self.experiments_paths]
        frames = [file for file in map(self.load_mat_file, self.experiments_paths)
                  if file not in experiments_to_exclude[-1]]

        frames_data = [frame[0] for frame in frames]
        frames_numbers = [frame[1] for frame in frames]

        return pd.concat(frames_data, axis=1, keys=frames_numbers)

    def build_data_frame_without_numbers(self):
        frames = [self.load_mat_file(file_path) for file_path in self.experiments_paths]
        frames_data = [frame[0] for frame in frames]
        return pd.concat(frames_data)
