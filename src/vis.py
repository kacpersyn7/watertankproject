import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
import os

path = '/home/kacper/experiments/'

experiments_paths = []

for filename in os.listdir(path):
    if not '.' in filename:
        for mat_file in os.listdir(path+filename):
            if mat_file == 'experiment_results.mat':
               experiments_paths.append(path+filename+'/'+mat_file)


def load_mat_file(file_path):
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
    v2 = data['v2'].reshape(-1)
    v3 = data['v3'].reshape(-1)

    data_merged = np.array([h1, h2, h3, v1, v2, v3, u])

    return pd.DataFrame(data_merged.T, index=t, columns=['h1', 'h2', 'h3', 'v1', 'v2', 'v3', 'u']), number

frames = [load_mat_file(file_path) for file_path in experiments_paths]
frames_data = [frame[0] for frame in frames]
frames_numbers = [frame[1] for frame in frames]
result = pd.concat(frames_data, axis=1, keys=frames_numbers)

plot_res = result['004'].plot(subplots=True, figsize=(10, 10))
