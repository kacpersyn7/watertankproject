import LoadData
from BAFFLE import BAFFLE
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix


def verif(dtest_y, predictions):
    plt.figure()
    plt.subplot(221)
    conf_mat = sns.heatmap(confusion_matrix(dtest_y, predictions), annot=True, fmt="d", linecolor="k", linewidths=3)
    plt.title("Macierz pomyłek", fontsize=20)


experiments = LoadData.LoadData()
data = experiments.build_data_frame()
k = 0.9
alpha = 0.5
lag_size = 2000
window_size = 2000
edz = 1000
baffle_alg = BAFFLE(lag_size, window_size, k=k, alpha=alpha, explained_variance_ratio=0.7)

with open('time_intervals.json', 'r') as fp:
    time_intervals_dict = json.load(fp)
features = ['h1', 'h2', 'h3']
# column_names = list(set([column[0] for column in data]))
column_names = list(time_intervals_dict.keys())
column_names.sort()
column_names.insert(0, column_names.pop(column_names.index('014')))
column_names.insert(0, column_names.pop(column_names.index('013')))
column_names.insert(0, column_names.pop(column_names.index('012')))
column_names.insert(0, column_names.pop(column_names.index('011')))
column_names.insert(0, column_names.pop(column_names.index('009')))
column_names.insert(0, column_names.pop(column_names.index('008')))
# column_names.insert(0, column_names.pop(column_names.index('006')))
column_names.insert(0, column_names.pop(column_names.index('004')))

# stationary_data.index = (stationary_data.index - min(stationary_data.index))
i = 0
data_list = []
metrics = []
times_real = []
times_predicted = []
column_names = ['024']
for name in column_names:
    for elem in time_intervals_dict[name]:
        three_min = data[name][elem[0]:elem[1]]
        three_min = three_min.fillna(0)
        three_min.index = (three_min.index - min(three_min.index))
        a = three_min.error
        k_er = 6000
        if name in ['004', '006', '008', '009', '011', '012', '013', '014']:
            if elem[0] == 200:
                k_er = 4000
            elif elem[0] == 700:
                k_er = 7000
        elif name == '007':
            if elem[0] == 1060:
                k_er = 1500
            else:
                k_er = 4000
        elif name == '019':
            if elem[0] == 1000:
                k_er = 8000
        elif name == '020':
            if elem[0] == 450:
                k_er = 7000
            elif elem[0] == 1090:
                k_er = 4000
        elif name == '027':
            if elem[0] == 1150:
                k_er = 3000
            elif elem[0] == 1450:
                k_er = 2500
        elif name == '028':
            if elem[0] == 200:
                k_er = 5000
            elif elem[0] == 1500:
                k_er = 2000
        elif name == '029':
            if elem[0] == 200:
                k_er = 3000
            elif elem[0] == 1500:
                k_er = 3500
            elif elem[0] == 1600:
                k_er = 2000
        elif name == '030':
            if elem[0] == 1600:
                k_er = 2000
        three_min.error = ((a == 1) & (a.shift(15) == 1) | (a.shift(k_er)) | (a.shift(k_er // 2)))
        even_plot = three_min.plot(subplots=True, legend=False, figsize=(10, 10), grid=True, lw=3)
        plt.figure()
        plt.xlabel('czas (s)')
        features_to_show = ['h1 (cm)', 'h2 (cm)', 'h3 (cm)', 'v1', 'v2', 'v3', 'u', 'error']

        for i, namel in enumerate(features_to_show):
            even_plot[i].set_ylabel(namel)

        for i in range(3, 8):
            even_plot[i].set_ylim(0, 1)


        dtest_y = three_min['error'].values[lag_size:]

        E, W, y, std_results, mean_results, projection_results = baffle_alg.fit_and_predict(three_min[features].values)
        E_arr = np.array(E)
        W_arr = np.array(W)
        std_results = np.array(std_results)
        mean_results = np.array(mean_results)
        dim = E_arr.shape[1]
        E_arr = np.concatenate((np.zeros((lag_size, dim)), E_arr))
        W_arr = np.concatenate((np.zeros((lag_size, dim)), W_arr))
        std_results = np.concatenate((np.tile(std_results[0, :], (lag_size, 1)), std_results))
        mean_results = np.concatenate((np.tile(mean_results[0, :], (lag_size, 1)), mean_results))
        # result = W_arr[:, 0]
        if dim == 1:
            result = W_arr[:, 0]
        elif dim == 2:
            result = np.logical_and(W_arr[:, 0], W_arr[:, 1])
        else:
            result = np.logical_or(np.logical_and(W_arr[:, 0], W_arr[:, 1]),
                                   np.logical_and(W_arr[:, 0], W_arr[:, 2]),
                                   np.logical_and(W_arr[:, 2], W_arr[:, 1]))

        # for i in range(dim):
            # result = np.logical_or(result, W_arr[:, i])
        plt.figure(figsize=(10, 3))
        plt.xlabel("czas(ms)")
        plt.plot(result)
        result = result[lag_size:]

        plt.figure()
        plt.xlabel("czas(ms)")
        for i in range(dim):
            plt.subplot(3, 1, i + 1)
            plt.plot(E_arr[:, i])

        plt.figure()
        plt.xlabel("czas(ms)")
        for i in range(dim):

            plt.subplot(3, 1, i + 1)
            plt.plot(W_arr[:, i])

        for i in range(dim):
            plt.figure(figsize=(10, 6))
            plt.xlabel("czas(ms)")
            plt.plot(projection_results[:, i])

            plt.plot(3 * std_results[:, i] + mean_results[:, i])
            plt.plot(mean_results[:, i] - 3 * std_results[:, i])
            plt.plot((3+k) * std_results[:, i] + mean_results[:, i])
            plt.plot(mean_results[:, i] - (3+k) * std_results[:, i])
            # plt.plot(3 * (std_results[:,i]+0.9))
            plt.plot(mean_results[:, i])
            plt.legend(('y', 'mean + 3*std', 'mean - 3*std', 'mean + (3+k)*std', 'mean + (3+k)*std', 'mean'),
                      loc='lower left')

        verif(dtest_y, result)
        print(elem[0])
        print(name)
        plt.show()

        input("next")
