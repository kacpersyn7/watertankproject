import LoadData
from BAFFLE import BAFFLE
import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import json

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
path = '../results/'


def create_confusion_from_times(times_real, times_predicted, edz=1000, name=''):
    pn=0
    pp=0
    fp=0
    fn=0
    for prediction, real in zip(times_predicted, times_real):
        if prediction == 0.0 and real == 0.0:
            pn += 1
        elif prediction != 0.0 and real == 0.0:
            fp += 1
        elif prediction == 0.0 and real != 0.0:
            fn += 1
        else:
            if abs(prediction - real) < edz:
                pp += 1
            else:
                fp += 1

    my_confusion_matrix = np.array([[pn, fp],
                                 [fn, pp]])
    acc = (pp+pn)/(pn+pp+fp+fn)
    prec = pp/(pp+fp)
    rec = pp/(pp+fn)
    f1 = (2*pp)/(2*pp + fn + fp)
    results = (' '.join(["{0:.2f}".format(acc), "{0:.2f}".format(prec),
                         "{0:.2f}".format(rec),
                         "{0:.2f}".format(f1)]))
    plt.figure()
    conf_mat=sns.heatmap(my_confusion_matrix, annot=True, fmt="d", linecolor="k", linewidths=3)
    conf_mat = conf_mat.get_figure()
    conf_mat.savefig(path + name + '_macierzpomylekedz.png')
    return results

def verif(dtest_y, predictions, name):
    acc = accuracy_score(dtest_y, predictions)
    print("\naccuracy_score :", acc)
    report = classification_report(dtest_y, predictions)
    print(report)
    results = (' '.join(["{0:.2f}".format(acc), "{0:.2f}".format(precision_score(dtest_y, predictions)),
                         "{0:.2f}".format(recall_score(dtest_y, predictions)),
                         "{0:.2f}".format(f1_score(dtest_y, predictions))]))
    plt.figure()
    plt.subplot(221)
    conf_mat = sns.heatmap(confusion_matrix(dtest_y, predictions), annot=True, fmt="d", linecolor="k", linewidths=3)
    conf_mat = conf_mat.get_figure()
    conf_mat.savefig(path + name + '_macierzpomylek.png')
    return results

experiments = LoadData.LoadData()
data = experiments.build_data_frame()
params = [{'k': 0.9, 'a': 0.25, 'var': 0.7},
          {'k': 0.9, 'a': 0.5, 'var': 0.7},
          {'k': 0.9, 'a': 0.75, 'var': 0.7},
          {'k': 0.3, 'a': 0.5, 'var': 0.7},
          {'k': 5, 'a': 0.5, 'var': 0.7},
          {'k': 0.9, 'a': 0.5, 'var': 0.95}]
params2 = [{'k': 10, 'a': 0.25, 'var': 0.7},
           {'k': 10, 'a': 0.25, 'var': 0.95}]
k = 10
alpha = 0.25
lag_size = 2000
window_size = 2000
edz = 2000
variance_ratio=0.7

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
# i = 0
# data_list = []
# metrics = []
# times_real = []
# times_predicted = []
# all_test = []
# all_results = []
# for par in params2:
data_list = []
metrics = []
times_real = []
times_predicted = []
all_test = []
all_results = []
# k = par['k']
# alpha = par['a']
# variance_ratio = par['var']
baffle_alg = BAFFLE(lag_size, window_size, k=k, alpha=alpha, explained_variance_ratio=variance_ratio)

for name in column_names:
    for elem in time_intervals_dict[name]:
        three_min = data[name][elem[0]:elem[1]]
        three_min = three_min.fillna(0)
        three_min.index = (three_min.index - min(three_min.index))
        a = three_min.error
        k_er = 5500
        if name in ['004', '006', '008', '009', '011', '012', '013', '014']:
            if elem[0] == 200:
                k_er = 3500
            elif elem[0] == 700:
                k_er = 6500
        elif name == '007':
            if elem[0] == 1060:
                k_er = 1500
            else:
                k_er = 3500
        elif name == '019':
            if elem[0] == 1000:
                k_er = 7500
        elif name == '020':
            if elem[0] == 450:
                k_er = 6500
            elif elem[0] == 1090:
                k_er = 3500
        elif name == '027':
            if elem[0] == 1150:
                k_er = 3000
            elif elem[0] == 1450:
                k_er = 2500
        elif name == '028':
            if elem[0] == 200:
                k_er = 4500
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
        three_min.error = ((a==1) & (a.shift(15)==1) | (a.shift(k_er)) | (a.shift(k_er//2)))
#        three_min.plot(figsize=(10,6))
         # plt.plot(three_min)
        dtest_y = three_min['error'].values[lag_size:]

        E, W, y, std_results, mean_results, projection_results = baffle_alg.fit_and_predict(three_min[features].values)
        E_arr = np.array(E)
        W_arr = np.array(W)
        std_results = np.array(std_results)
        mean_results = np.array(mean_results)
        dim = E_arr.shape[1]
        E_arr = np.concatenate((np.zeros((lag_size, dim)), E_arr))
        W_arr = np.concatenate((np.zeros((lag_size, dim)), W_arr))
        std_results = np.concatenate((np.tile(std_results[0,:], (lag_size,1)), std_results))
        mean_results = np.concatenate((np.tile(mean_results[0,:], (lag_size,1)), mean_results))
#        plt.figure()
#         result = W_arr[:, 0]
        if dim == 1:
            result = W_arr[:, 0]
        elif dim == 2:
            result = np.logical_and(W_arr[:, 0], W_arr[:, 1])
        else:
            result = np.logical_or(np.logical_and(W_arr[:, 0], W_arr[:, 1]),
                                   np.logical_and(W_arr[:, 0], W_arr[:, 2]),
                                   np.logical_and(W_arr[:, 2], W_arr[:, 1]))
        # for i in range(dim):
        #     result = np.logical_or(result, W_arr[:, i])
        result = result[lag_size:]
        first_real = np.argmax(dtest_y == True)
        first_predicted = np.argmax(result == True)
        times_real.append(first_real)
        times_predicted.append(first_predicted)
        acc = accuracy_score(dtest_y, result)
        print("\naccuracy_score :", acc)
        metrics.append(' '.join(["{0:.2f}".format(acc), "{0:.2f}".format(precision_score(dtest_y, result)), "{0:.2f}".format(recall_score(dtest_y, result)), "{0:.2f}".format(f1_score(dtest_y, result))]))
        all_results.append(result)
        all_test.append(dtest_y)
#
#        plt.plot(result)
        # for i in range(dim):
        #     plt.subplot(3, 1, i + 1)
        #     plt.plot(E_arr[:, i])
        #
        # plt.figure()

        #     plt.subplot(3, 1, i + 1)
        #     plt.plot(W_arr[:, i])

#        for i in range(dim):
#            plt.figure(figsize=(10, 6))
#            plt.xlabel("czas(s)")
#            plt.plot(projection_results[:, i])
#
#            plt.plot(3 * std_results[:, i] + mean_results[:, i])
#            plt.plot(mean_results[:, i] - 3 * std_results[:, i])
#            plt.plot((3+k) * std_results[:, i] + mean_results[:, i])
#            plt.plot(mean_results[:, i] - (3+k) * std_results[:, i])
#            # plt.plot(3 * (std_results[:,i]+0.9))
#            plt.plot(mean_results[:, i])
#            plt.legend(('y', 'mean + 3*std', 'mean - 3*std', 'mean + (3+k)*std', 'mean + (3+k)*std', 'mean'),
#                       loc='lower left')

#        plt.show()
#        input("next")
#        print(elem[0])
#        print(name)
#        plt.show()

#        input("next")

name = '_'.join([str(k), str(alpha), str(variance_ratio)])
detection_time_results = create_confusion_from_times(times_real, times_predicted, edz, name)
summary = verif(np.concatenate(all_test, axis=None), np.concatenate(all_results, axis=None), name)
with open(path+'_summary_' + name+'_.txt', 'w') as f:
    f.writelines(summary)
with open(path+'_edz_' + name+'_.txt', 'w') as f:
    f.writelines(detection_time_results)
with open(path+name+'_.txt', 'w') as f:
    for line in metrics:
        f.writelines(line)
        f.writelines("\n")
with open(path+'_dt_' + name+'_.txt', 'w') as f:
    for x, y in zip(times_real, times_predicted):
        f.writelines(' '.join([str(x), str(y), str(abs(x-y))]))
        f.writelines("\n")